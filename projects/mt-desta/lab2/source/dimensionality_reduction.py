"""
Task 5 (Optional): Dimensionality reduction for basic vectorization methods.
"""

import numpy as np
from sklearn.decomposition import PCA
from typing import List


class DimensionalityReducer:
    """
    Apply dimensionality reduction to basic vectorization methods.
    """
    
    def __init__(self, target_dim: int = 100):
        """
        Initialize dimensionality reducer.
        
        Args:
            target_dim: Target dimensionality after reduction
        """
        self.target_dim = target_dim
        self.pca = None
        self.is_fitted = False
    
    def fit(self, vectors: np.ndarray):
        """
        Fit PCA on vectors.
        
        Args:
            vectors: Array of shape (n_samples, n_features)
        """
        print(f"Fitting PCA: {vectors.shape[1]} -> {self.target_dim} dimensions")
        self.pca = PCA(n_components=self.target_dim)
        self.pca.fit(vectors)
        self.is_fitted = True
        print(f"Explained variance ratio: {self.pca.explained_variance_ratio_.sum():.4f}")
    
    def transform(self, vectors: np.ndarray) -> np.ndarray:
        """
        Transform vectors to reduced dimensionality.
        
        Args:
            vectors: Array of shape (n_samples, n_features)
            
        Returns:
            Reduced vectors of shape (n_samples, target_dim)
        """
        if not self.is_fitted:
            raise ValueError("PCA not fitted. Call fit() first.")
        
        return self.pca.transform(vectors)
    
    def fit_transform(self, vectors: np.ndarray) -> np.ndarray:
        """
        Fit and transform in one step.
        
        Args:
            vectors: Array of shape (n_samples, n_features)
            
        Returns:
            Reduced vectors of shape (n_samples, target_dim)
        """
        self.fit(vectors)
        return self.transform(vectors)

