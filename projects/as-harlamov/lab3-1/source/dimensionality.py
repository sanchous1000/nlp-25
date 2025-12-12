"""
Модуль для экспериментов с изменением размерности векторных представлений.
"""
import numpy as np
from typing import Tuple
from sklearn.decomposition import PCA


def random_dimension_dropout(X: np.ndarray, num_drop: int, random_state: int = 42) -> np.ndarray:
    """
    Отбрасывает несколько случайно выбранных размерностей из векторных представлений.
    
    Args:
        X: Матрица векторов (n_samples, n_features)
        num_drop: Количество размерностей для отбрасывания
        random_state: Seed для генератора случайных чисел
    
    Returns:
        Матрица с отброшенными размерностями
    """
    if num_drop >= X.shape[1]:
        raise ValueError(f"Cannot drop {num_drop} dimensions from vectors with {X.shape[1]} dimensions")
    
    rng = np.random.RandomState(random_state)
    indices_to_keep = rng.choice(X.shape[1], size=X.shape[1] - num_drop, replace=False)
    indices_to_keep = np.sort(indices_to_keep)  # Сохраняем порядок
    
    return X[:, indices_to_keep]


def reduce_dimension_pca(X: np.ndarray, n_components: int) -> Tuple[np.ndarray, PCA]:
    """
    Сокращает размерность векторных представлений с помощью PCA.
    
    Args:
        X: Матрица векторов (n_samples, n_features)
        n_components: Новая размерность
    
    Returns:
        Преобразованная матрица и обученная PCA модель
    """
    pca = PCA(n_components=n_components, random_state=42)
    X_reduced = pca.fit_transform(X)
    return X_reduced, pca


def add_mathematical_features(X: np.ndarray) -> np.ndarray:
    """
    Добавляет дополнительные размерности с использованием математических функций.
    
    Добавляет:
    - log(1 + |x|) для каждой исходной размерности
    - cos(x) для каждой исходной размерности
    - sin(x) для каждой исходной размерности
    """
    features_list = [X]
    
    # log(1 + |x|)
    log_features = np.log1p(np.abs(X))
    features_list.append(log_features)
    
    # cos(x)
    cos_features = np.cos(X)
    features_list.append(cos_features)
    
    # sin(x)
    sin_features = np.sin(X)
    features_list.append(sin_features)
    
    return np.hstack(features_list)


def experiment_dimension_dropout(
    X_train: np.ndarray,
    X_test: np.ndarray,
    num_drops_list: list
) -> list:
    """
    Проводит серию экспериментов с отбрасыванием размерностей.
    
    Returns:
        Список кортежей (num_drop, X_train_new, X_test_new)
    """
    results = []
    for num_drop in num_drops_list:
        X_train_new = random_dimension_dropout(X_train, num_drop)
        X_test_new = random_dimension_dropout(X_test, num_drop)
        results.append((num_drop, X_train_new, X_test_new))
    return results


def experiment_pca_reduction(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_components_list: list
) -> list:
    """
    Проводит серию экспериментов с сокращением размерности через PCA.
    
    Returns:
        Список кортежей (n_components, X_train_new, X_test_new, pca_model)
    """
    results = []
    for n_components in n_components_list:
        X_train_new, pca = reduce_dimension_pca(X_train, n_components)
        X_test_new = pca.transform(X_test)
        results.append((n_components, X_train_new, X_test_new, pca))
    return results


def experiment_mathematical_features(
    X_train: np.ndarray,
    X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Добавляет математические признаки к векторным представлениям.
    
    Returns:
        (X_train_new, X_test_new)
    """
    X_train_new = add_mathematical_features(X_train)
    X_test_new = add_mathematical_features(X_test)
    return X_train_new, X_test_new

