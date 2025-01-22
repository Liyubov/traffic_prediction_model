import numpy as np
import pickle
import json
from pathlib import Path

def load_memmap(data_path, shape, dtype='float32', mode='r'):
    """Загружает данные из файла np.memmap."""
    return np.memmap(data_path, dtype=dtype, mode=mode, shape=shape)

def save_memmap(data, data_path, dtype='float32'):
    """Сохраняет данные в файл np.memmap."""
    data_path = Path(data_path)
    if not data_path.parent.exists():
        data_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Папка {data_path.parent} создана.")
    memmap = np.memmap(data_path, dtype=dtype, mode='w+', shape=data.shape)
    memmap[:] = data[:]
    memmap.flush()
    print(f"Данные сохранены в {data_path}.")

def load_pkl(pickle_file: str) -> object:
    """Загружает данные из pickle файла."""
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print(f'Unable to load data from {pickle_file}: {e}')
        raise
    return pickle_data

def load_metadata(metadata_path: str) -> dict:
    """Загружает метаданные из JSON файла."""
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    return metadata

def load_all_data(data_dir: str):
    """Загружает все данные: metadata, data, adj."""
    data_dir = Path(data_dir)
    data_path = data_dir / 'data.dat'
    metadata_path = data_dir / 'desc.json'
    adj_path = data_dir / 'adj_mx.pkl'

    # Загрузка метаданных
    metadata = load_metadata(metadata_path)
    
    # Загрузка данных
    data_shape = (metadata['num_time_steps'], metadata['num_nodes'], metadata['num_features'])
    data = load_memmap(data_path, shape=data_shape)
    
    # Загрузка adjacency matrix
    try:
        _, _, adj_mx_pb = load_pkl(adj_path)
    except ValueError:
        adj_mx_pb = load_pkl(adj_path)
    
    return metadata, data, adj_mx_pb