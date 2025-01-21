import numpy as np
import pandas as pd

def add_features(data, feature_list, feature_functions, graph_feature_functions=None, index=None, adj_matrix=None, **kwargs):
    """
    Добавляет фичи в данные, поддерживая канал измерений и графовые признаки.

    Args:
        data (np.ndarray): Исходные данные, shape: [time, nodes, channels] или [time, nodes].
        feature_list (list): Список названий признаков для добавления.
        feature_functions (dict): Словарь функций для создания признаков.
        graph_feature_functions (dict): Словарь функций для графовых признаков.
        index (pd.DatetimeIndex): Индекс времени для временных признаков.
        adj_matrix (np.ndarray): Матрица смежности для графовых признаков.
        **kwargs: Дополнительные аргументы для функций.

    Returns:
        np.ndarray: Обновленные данные с дополнительными фичами, shape: [time, nodes, channels+N].
    """
    if len(data.shape) == 2:  # [time, nodes]
        data = data[..., np.newaxis]  # Преобразуем в [time, nodes, 1]

    time, nodes, channels = data.shape
    df = pd.DataFrame(data[:, :, :1].reshape(time, -1))

    new_features = []

    # Генерация временных или статистических признаков
    for feature_name in feature_list:
        if feature_name in feature_functions:
            func = feature_functions[feature_name]
            if feature_name in ['weekday', 'hour', 'weekday', 'hour', 'peak_hours', 'is_holiday', 'season']:
                feature = func(df.values, index)
            else:
                feature = func(df.values, **kwargs)

            if feature.ndim == 1:
                feature = np.tile(feature, (time, 1))
            elif feature.ndim == 2 and feature.shape[0] != time:
                raise ValueError(f"Признак '{feature_name}' имеет несовместимую форму: {feature.shape}")

            new_features.append(feature[..., np.newaxis])
        else:
            raise ValueError(f"Функция для признака '{feature_name}' не найдена.")

    # Генерация графовых признаков, если задана матрица смежности
    if adj_matrix is not None and graph_feature_functions is not None:
        for name, func in graph_feature_functions.items():
            graph_feature = func(adj_matrix)
            graph_feature = np.array([graph_feature[node] for node in range(nodes)])
            graph_feature = np.tile(graph_feature, (time, 1))
            new_features.append(graph_feature[..., np.newaxis])

    new_features = np.concatenate(new_features, axis=-1)
    data = np.concatenate([data, new_features], axis=-1)
    return data