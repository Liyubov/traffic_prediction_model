import numpy as np
import networkx as nx
from sklearn.cluster import KMeans
import plotly.graph_objects as go


class GraphClusterProcessor:
    def __init__(self, adj, data):
        """
        Инициализация класса.

        Параметры:
            adj (numpy.ndarray): Матрица смежности графа.
            data (numpy.ndarray): Исходные данные [time_steps, num_nodes, num_features].
        """
        self.adj = adj.copy()
        self.data = data
        np.fill_diagonal(self.adj, 0)  # Убираем диагональные элементы
        self.graph = nx.from_numpy_array(self.adj)

    def elbow_method(self, metric='betweenness', max_clusters=20):
        """
        Метод локтя для определения оптимального числа кластеров.

        Параметры:
            metric (str): Метрика графа ('betweenness', 'degree', 'closeness', 'eigenvector').
            max_clusters (int): Максимальное количество кластеров для анализа.

        Выводит график локтя.
        """
        values = self._get_graph_metric(metric)

        distortions = []
        for k in range(1, max_clusters + 1):
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(np.array(values).reshape(-1, 1))
            distortions.append(kmeans.inertia_)

        # Построение графика локтя
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(1, max_clusters + 1)),
            y=distortions,
            mode='lines+markers',
            name='Distortion'
        ))
        fig.update_layout(
            title='Метод локтя',
            xaxis_title='Число кластеров',
            yaxis_title='Инерция (Distortion)',
            template='plotly_white'
        )
        fig.show()

    def cluster_and_add_channel(self, metric='betweenness', n_clusters=5, one_hot=False):
        """
        Кластеризация узлов и добавление индексов кластеров как нового канала.

        Параметры:
            metric (str): Метрика графа ('betweenness', 'degree', 'closeness', 'eigenvector').
            n_clusters (int): Количество кластеров для K-Means.
            one_hot (bool): Если True, преобразовать индексы кластеров в формат one-hot.

        Возвращает:
            numpy.ndarray: Тензор данных с добавленным каналом [time_steps, num_nodes, num_features + 1] или 
                           [time_steps, num_nodes, num_features + n_clusters] в случае one-hot.
        """
        values = self._get_graph_metric(metric)

        # Кластеризация
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(np.array(values).reshape(-1, 1))

        if one_hot:
            # Преобразование в формат one-hot
            labels_one_hot = np.eye(n_clusters)[labels]  # [num_nodes, n_clusters]
            labels_expanded = np.expand_dims(labels_one_hot, axis=0)  # [1, num_nodes, n_clusters]
        else:
            # Преобразование в обычный канал
            labels_expanded = np.expand_dims(labels, axis=0)  # [1, num_nodes]
            labels_expanded = np.expand_dims(labels_expanded, axis=-1)  # [1, num_nodes, 1]

        # Дублирование меток по временным шагам
        labels_expanded = np.repeat(labels_expanded, self.data.shape[0], axis=0)  # [time_steps, num_nodes, *]

        # Добавление меток кластеров к данным
        data_with_clusters = np.concatenate([self.data, labels_expanded], axis=-1)  # [time_steps, num_nodes, num_features + *]
        return data_with_clusters

    def plot_group_average_speeds(self, labels, n_clusters):
        """
        Построение графика средней скорости по группам.

        Параметры:
            labels (numpy.ndarray): Метки кластеров узлов.
            n_clusters (int): Количество кластеров.
        """
        groups = [[] for _ in range(n_clusters)]
        for node, label in enumerate(labels):
            groups[label].append(node)

        # Вычисляем средние скорости по группам
        group_average_speeds = np.zeros((self.data.shape[0], n_clusters))
        for i, group in enumerate(groups):
            group_data = self.data[:, group, 0]  # Используем первый канал для вычислений
            group_average_speeds[:, i] = np.mean(group_data, axis=1)

        # Построение графика
        time_steps = np.arange(self.data.shape[0])
        fig = go.Figure()
        for i in range(n_clusters):
            fig.add_trace(go.Scatter(
                x=time_steps,
                y=group_average_speeds[:, i],
                mode='lines',
                name=f'Группа {i + 1}'
            ))

        fig.update_layout(
            title='Средняя скорость по группам',
            xaxis_title='Временные шаги',
            yaxis_title='Средняя скорость',
            legend_title='Группы',
            template='plotly_white'
        )
        fig.show()

    def _get_graph_metric(self, metric):
        """
        Получение значений указанной метрики графа.

        Параметры:
            metric (str): Метрика графа ('betweenness', 'degree', 'closeness', 'eigenvector').

        Возвращает:
            list: Значения метрики для узлов.
        """
        if metric == 'betweenness':
            return list(nx.betweenness_centrality(self.graph).values())
        elif metric == 'degree':
            return [val for _, val in self.graph.degree()]
        elif metric == 'closeness':
            return list(nx.closeness_centrality(self.graph).values())
        elif metric == 'eigenvector':
            return list(nx.eigenvector_centrality(self.graph, max_iter=1000).values())
        else:
            raise ValueError(f"Метрика '{metric}' не поддерживается. Выберите из: 'betweenness', 'degree', 'closeness', 'eigenvector'.")
