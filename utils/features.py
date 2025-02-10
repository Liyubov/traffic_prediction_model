# import pywt
import numpy as np
import pandas as pd
import networkx as nx
from scipy.fft import fft, rfft, ifft, irfft
from sklearn.cluster import KMeans
from scipy.stats import kurtosis, skew

def mean_feature(data):
    """Среднее значение."""
    return np.mean(data, axis=0)

def median_feature(data):
    """Медиана."""
    return np.median(data, axis=0)

def std_feature(data):
    """Стандартное отклонение."""
    return np.std(data, axis=0)

def min_feature(data):
    """Минимум."""
    return np.min(data, axis=0)

def max_feature(data):
    """Максимум."""
    return np.max(data, axis=0)

def derivative_feature(data):
    """Производная."""
    return np.diff(data, axis=0)

def lagged_feature(data, lags=[1,2,3]):
    """Добавляет лаггированные признаки для данных."""
    data_lagged = []
    for lag in lags:
        lagged_data = np.roll(data, shift=lag, axis=0)  # Сдвигаем данные на заданное количество шагов
        lagged_data[:lag] = 0  # Заполняем пропуски нулями
        data_lagged.append(lagged_data)
    
    # Объединяем оригинальные данные с лагами
    data_lagged = np.concatenate(data_lagged, axis=-1)
    
    return data_lagged

def kurtosis_feature(data):
    """Эксцесс."""
    return kurtosis(data, axis=0)

def skew_feature(data):
    """Асимметрия."""
    return skew(data, axis=0)

def quantile_feature(data, q=0.25):
    """Квантиль."""
    return np.quantile(data, q, axis=0)

def weekday_feature(data, index):
    """День недели."""
    return np.array([index.weekday] * data.shape[1]).T

def hour_feature(data, index):
    """Час дня."""
    return np.array([index.hour] * data.shape[1]).T

def minute_feature(data, index):
    """Минута дня."""
    return np.array([index.minute] * data.shape[1]).T

def minute_index_feature(data, index):
    """
    Возвращает индекс минуты в диапазоне 0-287 (группировка по 5 минут).
    """
    # Вычисляем общее количество минут с начала дня
    total_minutes = index.hour * 60 + index.minute
    # Группируем по 5 минут и получаем индекс
    minute_index = total_minutes // 5
    return np.array([minute_index] * data.shape[1]).T

def weekday_index_feature(data, index):
    """
    Возвращает индекс дня недели в диапазоне 0-6.
    """
    # День недели уже в диапазоне 0-6
    weekday_index = index.weekday
    return np.array([weekday_index] * data.shape[1]).T


def peak_hours_feature(data, index):
    """Пиковые часы."""
    nodes = data.shape[1]
    peak_hours = np.isin(index.hour, [7, 8, 9, 17, 18, 19, 20]).astype(int)
    return np.tile(peak_hours[:, np.newaxis], (1, nodes))

def is_holiday_feature(data, index, holidays):
    """Выходные/праздники."""
    return np.array([1 if index.date() in holidays else 0] * data.shape[1]).T

def season_feature(data, index):
    """Сезонность (время года)."""
    month = index.month
    season = (month % 12) // 3
    return np.array([season] * data.shape[1]).T

def rolling_mean_feature(data, window=3):
    """Скользящее среднее."""
    return pd.DataFrame(data).rolling(window=window, min_periods=1).mean().values

def rolling_std_feature(data, window=3):
    """Скользящее стандартное отклонение."""
    return pd.DataFrame(data).rolling(window=window, min_periods=1).std().fillna(0).values

def rolling_min_feature(data, window=3):
    """Скользящий минимум."""
    return pd.DataFrame(data).rolling(window=window, min_periods=1).min().values

def rolling_max_feature(data, window=3):
    """Скользящий максимум."""
    return pd.DataFrame(data).rolling(window=window, min_periods=1).max().values

def fft_feature(data):
    """Преобразование Фурье (FFT)."""
    return np.abs(rfft(data, axis=0)) # type: ignore

def fft_denoise_feature(data, threshold=0.001):
    """Очищает данные от шума с использованием FFT."""
    fft_data = fft(data, axis=0)
    fft_abs = np.abs(fft_data)
    fft_abs_normalized = fft_abs / np.max(fft_abs)
    fft_data[fft_abs_normalized < threshold] = 0
    denoised_data = ifft(fft_data, axis=0)
    return np.real(denoised_data)

def dwt_feature(data, wavelet='db1'):
    """Вейвлет-преобразование (DWT)."""
    coeffs = pywt.wavedec(data, wavelet, axis=0)
    result = np.concatenate(coeffs, axis=0)
    return result[:data.shape[0]]

def diff_feature(data, periods=1):
    """Разность значений."""
    return pd.DataFrame(data).diff(periods=periods).fillna(0).values


def degree_feature(adj_matrix):
    """Степень узла."""
    G = nx.from_numpy_array(adj_matrix)
    return dict(G.degree())

def weighted_degree_feature(adj_matrix):
    """Взвешенная степень узла."""
    G = nx.from_numpy_array(adj_matrix)
    return dict(G.degree(weight='weight'))

def degree_centrality_feature(adj_matrix):
    """Степень центральности."""
    G = nx.from_numpy_array(adj_matrix)
    return nx.degree_centrality(G)

def closeness_centrality_feature(adj_matrix):
    """Близость центральности."""
    G = nx.from_numpy_array(adj_matrix)
    return nx.closeness_centrality(G)

def betweenness_centrality_feature(adj_matrix):
    """Посредничество центральности."""
    G = nx.from_numpy_array(adj_matrix)
    return nx.betweenness_centrality(G)

def clustering_coefficient_feature(adj_matrix):
    """Коэффициент кластеризации для каждого узла."""
    G = nx.from_numpy_array(adj_matrix)
    clustering = nx.clustering(G) 
    return clustering

def node_indices_feature(adj_matrix):
    """Создает индексы узлов графа от 0 до N-1."""
    G = nx.from_numpy_array(adj_matrix)  # Создаем граф
    return {node: node for node in G.nodes}


