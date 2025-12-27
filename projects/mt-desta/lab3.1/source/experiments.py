"""
Experiment framework for classification experiments.
"""

import time
import numpy as np
from typing import Dict, List, Tuple
from .classifier import SVMClassifier, MLPClassifierWrapper
from .metrics import accuracy, precision, recall, f1_score, classification_report


def run_svm_experiments(X_train: np.ndarray, y_train: np.ndarray,
                       X_test: np.ndarray, y_test: np.ndarray,
                       kernels: List[str] = ['linear', 'rbf', 'poly'],
                       max_iters: List[int] = [100, 500, 1000, 2000],
                       C: float = 1.0) -> List[Dict]:
    """
    Run SVM experiments with different kernels and iteration counts.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        kernels: List of kernel types to test
        max_iters: List of max_iter values to test
        C: Regularization parameter
        
    Returns:
        List of experiment results
    """
    results = []
    
    for kernel in kernels:
        for max_iter in max_iters:
            print(f"\nExperiment: kernel={kernel}, max_iter={max_iter}")
            
            # Create and train classifier
            classifier = SVMClassifier(kernel=kernel, C=C, max_iter=max_iter)
            training_time = classifier.train(X_train, y_train)
            
            # Make predictions
            y_pred = classifier.predict(X_test)
            
            # Calculate metrics
            acc = accuracy(y_test, y_pred)
            prec = precision(y_test, y_pred, average='macro')
            rec = recall(y_test, y_pred, average='macro')
            f1 = f1_score(y_test, y_pred, average='macro')
            
            result = {
                'model_type': 'SVM',
                'kernel': kernel,
                'max_iter': max_iter,
                'C': C,
                'accuracy': acc,
                'precision': prec,
                'recall': rec,
                'f1_score': f1,
                'training_time': training_time
            }
            
            results.append(result)
            
            print(f"  Accuracy: {acc:.4f}, F1: {f1:.4f}, Time: {training_time:.2f}s")
    
    return results


def run_model_comparison(X_train: np.ndarray, y_train: np.ndarray,
                        X_test: np.ndarray, y_test: np.ndarray,
                        max_iters: List[int] = [100, 500, 1000]) -> List[Dict]:
    """
    Compare SVM with other models (e.g., MLP).
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        max_iters: List of max_iter values to test
        
    Returns:
        List of experiment results
    """
    results = []
    
    # Test SVM
    for max_iter in max_iters:
        print(f"\nSVM Experiment: max_iter={max_iter}")
        classifier = SVMClassifier(kernel='linear', max_iter=max_iter)
        training_time = classifier.train(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        result = {
            'model_type': 'SVM',
            'kernel': 'linear',
            'max_iter': max_iter,
            'accuracy': accuracy(y_test, y_pred),
            'precision': precision(y_test, y_pred, average='macro'),
            'recall': recall(y_test, y_pred, average='macro'),
            'f1_score': f1_score(y_test, y_pred, average='macro'),
            'training_time': training_time
        }
        results.append(result)
    
    # Test MLP
    for max_iter in max_iters:
        print(f"\nMLP Experiment: max_iter={max_iter}")
        classifier = MLPClassifierWrapper(hidden_layer_sizes=(100,), max_iter=max_iter)
        training_time = classifier.train(X_train, y_train)
        y_pred = classifier.predict(X_test)
        
        result = {
            'model_type': 'MLP',
            'kernel': None,
            'max_iter': max_iter,
            'accuracy': accuracy(y_test, y_pred),
            'precision': precision(y_test, y_pred, average='macro'),
            'recall': recall(y_test, y_pred, average='macro'),
            'f1_score': f1_score(y_test, y_pred, average='macro'),
            'training_time': training_time
        }
        results.append(result)
    
    return results


def run_vector_transformation_experiments(X_train: np.ndarray, y_train: np.ndarray,
                                         X_test: np.ndarray, y_test: np.ndarray,
                                         model_type: str = 'SVM', kernel: str = 'linear',
                                         optimal_max_iter: int = 1000,
                                         transformation_type: str = 'drop_dimensions',
                                         param_values: List[int] = None) -> List[Dict]:
    """
    Run experiments with vector transformations.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        model_type: 'SVM' or 'MLP'
        kernel: Kernel type for SVM
        optimal_max_iter: Optimal max_iter value from previous experiments
        transformation_type: 'drop_dimensions', 'reduce_dim', or 'add_features'
        param_values: List of parameter values to test
        
    Returns:
        List of experiment results
    """
    from .vector_transformations import drop_random_dimensions, reduce_dimensionality, add_math_features, apply_pca_reduction
    
    results = []
    
    if transformation_type == 'drop_dimensions':
        if param_values is None:
            # Default: drop 0, 10, 20, 30, 40, 50 dimensions
            original_dim = X_train.shape[1]
            param_values = [0, 10, 20, 30, 40, 50] if original_dim > 50 else list(range(0, original_dim, 5))
        
        for num_drop in param_values:
            if num_drop >= X_train.shape[1]:
                continue
            
            print(f"\nTransformation: drop {num_drop} dimensions")
            
            # Apply transformation
            X_train_transformed, _ = drop_random_dimensions(X_train, num_drop)
            X_test_transformed, _ = drop_random_dimensions(X_test, num_drop, random_state=42)
            
            # Train and evaluate
            if model_type == 'SVM':
                classifier = SVMClassifier(kernel=kernel, max_iter=optimal_max_iter)
            else:
                classifier = MLPClassifierWrapper(max_iter=optimal_max_iter)
            
            training_time = classifier.train(X_train_transformed, y_train)
            y_pred = classifier.predict(X_test_transformed)
            
            result = {
                'transformation': 'drop_dimensions',
                'param_value': num_drop,
                'new_dim': X_train_transformed.shape[1],
                'accuracy': accuracy(y_test, y_pred),
                'precision': precision(y_test, y_pred, average='macro'),
                'recall': recall(y_test, y_pred, average='macro'),
                'f1_score': f1_score(y_test, y_pred, average='macro'),
                'training_time': training_time
            }
            results.append(result)
    
    elif transformation_type == 'reduce_dim':
        if param_values is None:
            # Default: reduce to various dimensions
            original_dim = X_train.shape[1]
            param_values = [50, 75, 100, 125, 150] if original_dim > 150 else list(range(50, original_dim + 1, 25))
        
        for target_dim in param_values:
            if target_dim >= X_train.shape[1]:
                continue
            
            print(f"\nTransformation: reduce to {target_dim} dimensions")
            
            # Apply PCA reduction
            X_train_transformed = apply_pca_reduction(X_train, target_dim)
            X_test_transformed = apply_pca_reduction(X_test, target_dim)
            
            # Train and evaluate
            if model_type == 'SVM':
                classifier = SVMClassifier(kernel=kernel, max_iter=optimal_max_iter)
            else:
                classifier = MLPClassifierWrapper(max_iter=optimal_max_iter)
            
            training_time = classifier.train(X_train_transformed, y_train)
            y_pred = classifier.predict(X_test_transformed)
            
            result = {
                'transformation': 'reduce_dim',
                'param_value': target_dim,
                'new_dim': target_dim,
                'accuracy': accuracy(y_test, y_pred),
                'precision': precision(y_test, y_pred, average='macro'),
                'recall': recall(y_test, y_pred, average='macro'),
                'f1_score': f1_score(y_test, y_pred, average='macro'),
                'training_time': training_time
            }
            results.append(result)
    
    elif transformation_type == 'add_features':
        print(f"\nTransformation: add mathematical features")
        
        # Apply transformation
        X_train_transformed = add_math_features(X_train)
        X_test_transformed = add_math_features(X_test)
        
        print(f"Original dimension: {X_train.shape[1]}, New dimension: {X_train_transformed.shape[1]}")
        
        # Train and evaluate
        if model_type == 'SVM':
            classifier = SVMClassifier(kernel=kernel, max_iter=optimal_max_iter)
        else:
            classifier = MLPClassifierWrapper(max_iter=optimal_max_iter)
        
        training_time = classifier.train(X_train_transformed, y_train)
        y_pred = classifier.predict(X_test_transformed)
        
        result = {
            'transformation': 'add_features',
            'param_value': None,
            'new_dim': X_train_transformed.shape[1],
            'accuracy': accuracy(y_test, y_pred),
            'precision': precision(y_test, y_pred, average='macro'),
            'recall': recall(y_test, y_pred, average='macro'),
            'f1_score': f1_score(y_test, y_pred, average='macro'),
            'training_time': training_time
        }
        results.append(result)
    
    return results

