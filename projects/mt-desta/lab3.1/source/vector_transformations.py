"""
Vector transformation methods for experiments.
"""

import numpy as np
from typing import Tuple


def drop_random_dimensions(X: np.ndarray, num_drop: int, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Drop randomly selected dimensions from vectors.
    
    Args:
        X: Input vectors of shape (n_samples, n_features)
        num_drop: Number of dimensions to drop
        random_state: Random seed
        
    Returns:
        Tuple of (transformed_vectors, dropped_indices)
    """
    np.random.seed(random_state)
    n_features = X.shape[1]
    
    if num_drop >= n_features:
        raise ValueError(f"Cannot drop {num_drop} dimensions from {n_features}-dimensional vectors")
    
    # Randomly select dimensions to drop
    dropped_indices = np.random.choice(n_features, size=num_drop, replace=False)
    kept_indices = np.setdiff1d(np.arange(n_features), dropped_indices)
    
    # Keep only non-dropped dimensions
    X_transformed = X[:, kept_indices]
    
    return X_transformed, dropped_indices


def reduce_dimensionality(X: np.ndarray, target_dim: int) -> np.ndarray:
    """
    Reduce dimensionality using PCA (simplified - just takes first N dimensions).
    For proper PCA, use sklearn.decomposition.PCA, but this is a simple version.
    
    Args:
        X: Input vectors of shape (n_samples, n_features)
        target_dim: Target dimensionality
        
    Returns:
        Transformed vectors of shape (n_samples, target_dim)
    """
    if target_dim >= X.shape[1]:
        return X
    
    # Simple approach: take first target_dim dimensions
    # For proper PCA, would need to compute principal components
    return X[:, :target_dim]


def add_math_features(X: np.ndarray) -> np.ndarray:
    """
    Add additional dimensions using standard mathematical functions.
    
    Args:
        X: Input vectors of shape (n_samples, n_features)
        
    Returns:
        Extended vectors with additional features
    """
    new_features = []
    
    # Add log features (with handling for negative/zero values)
    X_abs = np.abs(X) + 1e-10  # Add small epsilon to avoid log(0)
    log_features = np.log(X_abs)
    new_features.append(log_features)
    
    # Add cosine features
    cos_features = np.cos(X)
    new_features.append(cos_features)
    
    # Add sine features
    sin_features = np.sin(X)
    new_features.append(sin_features)
    
    # Add square root features (absolute value to handle negatives)
    sqrt_features = np.sqrt(X_abs)
    new_features.append(sqrt_features)
    
    # Add squared features
    squared_features = X ** 2
    new_features.append(squared_features)
    
    # Concatenate all new features
    X_extended = np.concatenate([X] + new_features, axis=1)
    
    return X_extended


def apply_pca_reduction(X: np.ndarray, target_dim: int, random_state: int = 42) -> np.ndarray:
    """
    Apply proper PCA dimensionality reduction using sklearn.
    
    Args:
        X: Input vectors of shape (n_samples, n_features)
        target_dim: Target dimensionality
        random_state: Random seed
        
    Returns:
        Transformed vectors of shape (n_samples, target_dim)
    """
    from sklearn.decomposition import PCA
    
    if target_dim >= X.shape[1]:
        return X
    
    pca = PCA(n_components=target_dim, random_state=random_state)
    X_transformed = pca.fit_transform(X)
    
    return X_transformed

