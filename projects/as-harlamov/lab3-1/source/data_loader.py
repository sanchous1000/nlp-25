"""
Модуль для загрузки векторных представлений документов и меток классов.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple


def load_vectors_from_tsv(tsv_path: Path) -> np.ndarray:
    """
    Загружает векторные представления из TSV файла.
    
    Формат: doc_id\tvec1\tvec2\t...\tvecN
    """
    vectors = []
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) > 1:
                # Пропускаем doc_id (первый элемент) и берем вектор
                vector = [float(x) for x in parts[1:]]
                vectors.append(vector)
    return np.array(vectors)


def load_labels_from_csv(csv_path: Path) -> np.ndarray:
    """
    Загружает метки классов из CSV файла.
    
    Формат: class,title,text
    
    Метки нормализуются так, чтобы начинаться с 0.
    """
    df = pd.read_csv(csv_path, header=None, names=['class', 'title', 'text'])
    # Преобразуем метки классов в числовой формат (если они строковые)
    labels = df['class'].values
    # Если метки строковые, преобразуем их в числовые
    if labels.dtype == object:
        unique_labels = np.unique(labels)
        label_to_int = {label: idx for idx, label in enumerate(unique_labels)}
        labels = np.array([label_to_int[label] for label in labels])
    else:
        # Нормализуем числовые метки, чтобы они начинались с 0
        unique_labels = np.unique(labels)
        min_label = np.min(unique_labels)
        if min_label != 0:
            labels = labels - min_label
    return labels


def load_train_test_data(
    train_csv_path: Path,
    test_csv_path: Path,
    train_vectors_path: Path = None,
    test_vectors_path: Path = None,
    lab2_path: Path = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Загружает обучающую и тестовую выборки.
    
    Если векторы не предоставлены, они будут сгенерированы используя lab2.
    """
    # Загружаем метки
    y_train = load_labels_from_csv(train_csv_path)
    y_test = load_labels_from_csv(test_csv_path)
    
    # Загружаем или генерируем векторы
    if train_vectors_path and train_vectors_path.exists():
        X_train = load_vectors_from_tsv(train_vectors_path)
    else:
        # Нужно будет сгенерировать, используя lab2
        raise ValueError(f"Train vectors not found at {train_vectors_path}. Need to generate them.")
    
    if test_vectors_path and test_vectors_path.exists():
        X_test = load_vectors_from_tsv(test_vectors_path)
    else:
        raise ValueError(f"Test vectors not found at {test_vectors_path}. Need to generate them.")
    
    # Проверяем соответствие размеров
    if len(X_train) != len(y_train):
        raise ValueError(f"Mismatch: X_train has {len(X_train)} samples, y_train has {len(y_train)}")
    if len(X_test) != len(y_test):
        raise ValueError(f"Mismatch: X_test has {len(X_test)} samples, y_test has {len(y_test)}")
    
    return X_train, y_train, X_test, y_test

