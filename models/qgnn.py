import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

def normalize_directed_adj(adj, mode='row'):
    """
    Нормализация для ориентированного графа.
    
    Args:
        adj (torch.Tensor): Матрица смежности (N x N).
        mode (str): 'row' для нормализации по строкам, 'col' для нормализации по столбцам.
    
    Returns:
        torch.Tensor: Нормализованная матрица смежности.
    """
    adj = adj + torch.eye(adj.size(0)).to(adj.device)
    if mode == 'row':
        degree = torch.sum(adj, dim=1, keepdim=True)
    elif mode == 'col':
        degree = torch.sum(adj, dim=0, keepdim=True)
    else:
        raise ValueError("Mode must be 'row' or 'col'.")
    
    degree_inv = torch.where(degree > 0, 1.0 / degree, torch.zeros_like(degree))
    return adj * degree_inv

def normalize_adj(adj):
    """
    Нормализует матрицу смежности симметричным способом.
    
    Args:
        adj (torch.Tensor): Матрица смежности (N x N).
    
    Returns:
        torch.Tensor: Нормализованная матрица смежности.
    """
    # Добавляем петли (self-loops) к матрице смежности
    adj = adj + torch.eye(adj.size(0)).to(adj.device)
    
    # Вычисляем степени узлов (degree matrix)
    degree = torch.sum(adj, dim=1)
    
    # Симметричная нормализация: D^(-1/2) * A * D^(-1/2)
    degree_inv_sqrt = torch.diag(torch.pow(degree, -0.5))
    adj_normalized = torch.mm(torch.mm(degree_inv_sqrt, adj), degree_inv_sqrt)
    
    return adj_normalized


# Функция для создания матрицы Гамильтона
def make_quaternion_mul(kernel):
    dim = kernel.size(1) // 4
    r, i, j, k = torch.split(kernel, [dim, dim, dim, dim], dim=1)
    hamilton = torch.cat([
        torch.cat([r, -i, -j, -k], dim=0),
        torch.cat([i, r, -k, j], dim=0),
        torch.cat([j, k, r, -i], dim=0),
        torch.cat([k, -j, i, r], dim=0)
    ], dim=1)
    return hamilton

# Слой QGNN
class QGNNLayer(nn.Module):
    def __init__(self, in_features, out_features, num_nodes, space, dropout=0.5, act=F.tanh):
        super(QGNNLayer, self).__init__()
        self.num_nodes = num_nodes
        self.space = space
        self.act = act
        self.dropout = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(out_features)

        # Проверяем кратность in_features
        if self.space == 'q':
            assert in_features % 4 == 0, "in_features must be divisible by 4 for quaternion operations"
            self.weight = Parameter(torch.FloatTensor(in_features // 4, out_features))
        else:
            self.weight = Parameter(torch.FloatTensor(in_features, out_features))

        self.dim = self.weight.size(1) // 4
        self.reset_parameters()

    def reset_parameters(self):
        stdv = math.sqrt(6.0 / (self.weight.size(0) + self.weight.size(1)))
        self.weight.data.uniform_(-stdv, stdv)

    def forward(self, x, adj):
        x = self.dropout(x)
        if self.space == 'q':
            hamilton = make_quaternion_mul(self.weight)
            support = torch.mm(x, hamilton)  # Hamilton product
        else:
            support = torch.mm(x, self.weight)
        
        B_N, hidden_dim = support.shape
        B = B_N // self.num_nodes
        N = self.num_nodes
        support = support.view(B, N, hidden_dim)

        # Обработка каждого батча с использованием torch.bmm
        output = torch.bmm(adj.unsqueeze(0).expand(B, -1, -1), support)
        output = output.view(B * N, hidden_dim)
        output = self.bn(output)
        output = self.act(output)

        return output


# Основная модель для прогнозирования трафика
class QGNNTrafficPredictor(nn.Module):
    def __init__(self, adj, 
                 num_nodes, 
                 input_dim, 
                 hidden_dim, 
                 output_dim, 
                 num_layers, 
                 pre_len, 
                 space, 
                 dropout=0.5, 
                 directed=False):
        super(QGNNTrafficPredictor, self).__init__()
        self.adj = normalize_directed_adj(adj) if directed else normalize_adj(adj)
        self.pre_len = pre_len
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout = dropout

        # Убедимся, что input_dim кратно 4
        assert input_dim % 4 == 0, "input_dim must be divisible by 4 for quaternion operations"

        # Инициализация эмбеддингов
        self.T_i_D_emb = nn.Embedding(288, 5)  # Эмбеддинг для T_i_D (288 вариантов)
        self.D_i_W_emb = nn.Embedding(7, 5)    # Эмбеддинг для D_i_W (7 вариантов)

        # QGNN слои
        self.qgnn_layers = nn.ModuleList([
            QGNNLayer(input_dim if i == 0 else hidden_dim, hidden_dim, self.num_nodes, space, dropout)
            for i in range(num_layers)
        ])

        # Временной слой (GRU)
        self.temporal_layer = nn.GRU(hidden_dim + 11, hidden_dim, batch_first=True)
        self.relu = F.relu
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, history_data):
        B, L, N, C = history_data.shape
        history_data = history_data.permute(0, 2, 1, 3)  # [B, N, L, C]
        speed = history_data[:, :, :, :1]  # [B, N, L, input_dim]

        # Восстановление целочисленных значений для T_i_D и D_i_W
        T_i_D_indices = torch.round(history_data[:, :, :, 1] * 287).type(torch.LongTensor).to(history_data.device)  # 288 вариантов (0-287)
        D_i_W_indices = torch.round(history_data[:, :, :, 2] * 6).type(torch.LongTensor).to(history_data.device)    # 7 вариантов (0-6)

        # Применение эмбеддингов
        T_D = self.T_i_D_emb(T_i_D_indices)  # [B, N, L, hidden_dim]
        D_W = self.D_i_W_emb(D_i_W_indices)  # [B, N, L, hidden_dim]

        outputs = []
        for t in range(L):
            x = history_data[:, :, t, :]  # [B, N, C]
            x = x.reshape(B * N, -1)  # [B * N, C]

            for i, qgnn_layer in enumerate(self.qgnn_layers):
                x = qgnn_layer(x, self.adj)  # [B * N, hidden_dim]

            x = x.reshape(B, N, -1)  # [B, N, hidden_dim]
            outputs.append(x)

        # Объединяем выходы по временным шагам
        outputs = torch.stack(outputs, dim=2)  # [B, N, L, hidden_dim]
        outputs = torch.cat([outputs, speed, T_D, D_W], dim=-1)  # [B, N, L, hidden_dim + input_dim + hidden_dim + hidden_dim]

        # Применяем временной слой (GRU)
        outputs = outputs.reshape(B * N, L, -1)  # [B * N, L, hidden_dim + input_dim + hidden_dim + hidden_dim]
        outputs, _ = self.temporal_layer(outputs)  # [B * N, L, hidden_dim]
        outputs = outputs.reshape(B, N, L, -1)  # [B, N, L, hidden_dim]
        outputs = outputs[:, :, -self.pre_len:, :]  # [B, N, pre_len, hidden_dim]
        outputs = self.relu(outputs)

        # Применяем выходной слой
        outputs = self.fc(outputs)  # [B, N, pre_len, output_dim]
        return outputs.permute(0, 2, 1, 3)  # [B, pre_len, N, output_dim]


# Пример использования
if __name__ == "__main__":
    # Параметры модели
    pre_len = 12
    num_nodes = 207  # Количество узлов (датчиков) в PEMS08
    input_dim = 4  # Количество признаков (должно быть кратно 4)
    hidden_dim = 64  # Размер скрытого слоя
    output_dim = 1  # Прогнозируемое значение (например, скорость)
    num_layers = 1  # Количество QGNN слоев
    dropout = 0.5  # Dropout

    # Создаем модель

    # Пример входных данных
    L = 12  # Длина временного ряда (история)
    B = 32  # Размер батча
    N = num_nodes  # Количество узлов
    C = input_dim  # Количество признаков
    history_data = torch.randn(B, L, N, C)  # [B, L, N, C]
    print(f"history_data.shape = {history_data.shape}")

    # Матрица смежности (пример)
    adj = torch.zeros(N, N)  # Инициализируем нулевую матрицу смежности
    for i in range(N):
        for j in range(N):
            if i != j and torch.rand(1) < 0.1:  # 10% вероятность наличия ребра
                adj[i, j] = 1  # Устанавливаем связь между узлами i и j

    model = QGNNTrafficPredictor(adj, num_nodes, input_dim, hidden_dim, output_dim, num_layers, pre_len, space='q', dropout=dropout, directed=True)
    
    # Прогноз
    output = model(history_data)
    print(output.shape)  # [B, N, output_dim]