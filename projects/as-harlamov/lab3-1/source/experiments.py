import time
import warnings
import numpy as np
from typing import Dict, List, Tuple
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

from metrics import calculate_all_metrics

def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    kernel: str = 'linear',
    max_iter: int = 1000,
    C: float = 1.0,
    tol: float = 1e-3
) -> Tuple[SVC, float]:
    
    model = SVC(
        kernel=kernel, 
        max_iter=max_iter, 
        C=C, 
        tol=tol,
        random_state=42,
        verbose=False
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    return model, training_time

def train_mlp(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_iter: int = 1000,
    hidden_layer_sizes: Tuple[int, ...] = (100,),
    random_state: int = 42
) -> Tuple[MLPClassifier, float]:
    
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layer_sizes,
        max_iter=max_iter,
        random_state=random_state,
        early_stopping=False
    )
    
    start_time = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start_time
    
    return model, training_time

def run_experiment(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str = 'svm',
    kernel: str = 'linear',
    max_iter: int = 1000,
    num_classes: int = None,
    **kwargs
) -> Dict:
    
    if num_classes is None:
        num_classes = len(np.unique(np.concatenate([y_train, y_test])))
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    if model_type == 'svm':
        if 'C' in kwargs:
            C = kwargs['C']
        else:
            C = 1.0
        tol = kwargs.get('tol', 1e-3)
        model, training_time = train_svm(X_train_scaled, y_train, kernel=kernel, max_iter=max_iter, C=C, tol=tol)
    elif model_type == 'mlp':
        hidden_layer_sizes = kwargs.get('hidden_layer_sizes', (100,))
        model, training_time = train_mlp(X_train_scaled, y_train, max_iter=max_iter, 
                                        hidden_layer_sizes=hidden_layer_sizes)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    y_pred = model.predict(X_test_scaled)
    
    metrics = calculate_all_metrics(y_test, y_pred, num_classes)
    
    result = {
        'model_type': model_type,
        'kernel': kernel if model_type == 'svm' else None,
        'max_iter': max_iter,
        'training_time': training_time,
        'accuracy': metrics['accuracy'],
        'precision': metrics['macro_precision'],
        'recall': metrics['macro_recall'],
        'f1_score': metrics['macro_f1'],
        'per_class_precision': metrics['per_class_precision'],
        'per_class_recall': metrics['per_class_recall'],
        'per_class_f1': metrics['per_class_f1'],
        'confusion_matrix': metrics['confusion_matrix'].tolist()
    }
    
    return result

def run_iteration_experiments(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_type: str = 'svm',
    kernel: str = 'linear',
    max_iters: List[int] = None,
    num_classes: int = None
) -> List[Dict]:
    
    if max_iters is None:
        max_iters = [500, 1000, 2000]
    
    results = []
    for max_iter in max_iters:
        print(f"  Эксперимент: {model_type}, kernel={kernel}, max_iter={max_iter}")
        result = run_experiment(
            X_train, y_train, X_test, y_test,
            model_type=model_type,
            kernel=kernel,
            max_iter=max_iter,
            num_classes=num_classes
        )
        results.append(result)
        print(f"    Accuracy: {result['accuracy']:.4f}, F1: {result['f1_score']:.4f}, Time: {result['training_time']:.2f}s")
    
    return results

def run_kernel_comparison(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    kernels: List[str] = None,
    max_iter: int = 1000,
    num_classes: int = None
) -> List[Dict]:
    
    if kernels is None:
        kernels = ['linear', 'rbf']
    
    results = []
    for kernel in kernels:
        print(f"  Эксперимент: SVM, kernel={kernel}, max_iter={max_iter}")
        result = run_experiment(
            X_train, y_train, X_test, y_test,
            model_type='svm',
            kernel=kernel,
            max_iter=max_iter,
            num_classes=num_classes
        )
        results.append(result)
        print(f"    Accuracy: {result['accuracy']:.4f}, F1: {result['f1_score']:.4f}, Time: {result['training_time']:.2f}s")
    
    return results
