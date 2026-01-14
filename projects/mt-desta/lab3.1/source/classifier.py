"""
SVM classifier with multiple kernel functions and comparison with other models.
"""

import time
import numpy as np
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional


class SVMClassifier:
    """
    SVM classifier wrapper with support for multiple kernels and epoch tracking.
    """
    
    def __init__(self, kernel: str = 'linear', C: float = 1.0, max_iter: int = 1000, 
                 gamma: str = 'scale', degree: int = 3, scale_data: bool = True):
        """
        Initialize SVM classifier.
        
        Args:
            kernel: Kernel type ('linear', 'rbf', 'poly', 'sigmoid')
            C: Regularization parameter
            max_iter: Maximum number of iterations (epochs)
            gamma: Kernel coefficient ('scale', 'auto', or float)
            degree: Degree for polynomial kernel
            scale_data: Whether to scale/normalize input data
        """
        self.kernel = kernel
        self.C = C
        self.max_iter = max_iter
        self.gamma = gamma
        self.degree = degree
        self.scale_data = scale_data
        self.scaler = StandardScaler() if scale_data else None
        
        self.model = SVC(
            kernel=kernel,
            C=C,
            max_iter=max_iter,
            gamma=gamma,
            degree=degree if kernel == 'poly' else 3,
            random_state=42,
            decision_function_shape='ovr'  # One-vs-rest for multiclass
        )
        self.is_trained = False
        self.training_time = 0.0
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """
        Train the SVM classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Training time in seconds
        """
        # Scale data if enabled
        if self.scale_data and self.scaler is not None:
            X_train = self.scaler.fit_transform(X_train)
        
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time
        self.is_trained = True
        return self.training_time
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X_test: Test features
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        return self.model.predict(X_test)
    
    def get_params(self) -> Dict:
        """Get model parameters."""
        return {
            'kernel': self.kernel,
            'C': self.C,
            'max_iter': self.max_iter,
            'gamma': self.gamma,
            'degree': self.degree
        }


class MLPClassifierWrapper:
    """
    MLP (Multi-Layer Perceptron) classifier for comparison.
    """
    
    def __init__(self, hidden_layer_sizes: Tuple[int, ...] = (100,), 
                 max_iter: int = 1000, learning_rate: str = 'constant', scale_data: bool = True):
        """
        Initialize MLP classifier.
        
        Args:
            hidden_layer_sizes: Tuple of hidden layer sizes
            max_iter: Maximum number of iterations
            learning_rate: Learning rate schedule
            scale_data: Whether to scale/normalize input data
        """
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.learning_rate = learning_rate
        self.scale_data = scale_data
        self.scaler = StandardScaler() if scale_data else None
        
        self.model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            max_iter=max_iter,
            learning_rate=learning_rate,
            random_state=42
        )
        self.is_trained = False
        self.training_time = 0.0
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray) -> float:
        """
        Train the MLP classifier.
        
        Args:
            X_train: Training features
            y_train: Training labels
            
        Returns:
            Training time in seconds
        """
        # Scale data if enabled
        if self.scale_data and self.scaler is not None:
            X_train = self.scaler.fit_transform(X_train)
        
        start_time = time.time()
        self.model.fit(X_train, y_train)
        self.training_time = time.time() - start_time
        self.is_trained = True
        return self.training_time
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """
        Make predictions.
        
        Args:
            X_test: Test features
            
        Returns:
            Predicted labels
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Scale data if enabled
        if self.scale_data and self.scaler is not None:
            X_test = self.scaler.transform(X_test)
        
        return self.model.predict(X_test)
    
    def get_params(self) -> Dict:
        """Get model parameters."""
        return {
            'hidden_layer_sizes': self.hidden_layer_sizes,
            'max_iter': self.max_iter,
            'learning_rate': self.learning_rate
        }

