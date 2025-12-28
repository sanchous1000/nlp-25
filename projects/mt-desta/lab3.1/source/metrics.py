"""
Custom implementation of classification metrics.
All metrics are implemented from scratch without using library functions.
"""

import numpy as np
from typing import Dict, List, Tuple


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = None) -> np.ndarray:
    """
    Compute confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        num_classes: Number of classes (if None, inferred from data)
        
    Returns:
        Confusion matrix of shape (num_classes, num_classes)
    """
    if num_classes is None:
        num_classes = max(np.max(y_true), np.max(y_pred)) + 1
    
    cm = np.zeros((num_classes, num_classes), dtype=np.int32)
    
    for i in range(len(y_true)):
        true_label = int(y_true[i])
        pred_label = int(y_pred[i])
        cm[true_label, pred_label] += 1
    
    return cm


def precision(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> float:
    """
    Calculate precision score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: 'macro' for macro-averaged precision, 'micro' for micro-averaged
        
    Returns:
        Precision score
    """
    num_classes = max(np.max(y_true), np.max(y_pred)) + 1
    cm = confusion_matrix(y_true, y_pred, num_classes)
    
    if average == 'macro':
        # Macro-averaged precision: average precision for each class
        precisions = []
        for i in range(num_classes):
            tp = cm[i, i]  # True positives
            fp = np.sum(cm[:, i]) - tp  # False positives
            if tp + fp > 0:
                precisions.append(tp / (tp + fp))
            else:
                precisions.append(0.0)
        return np.mean(precisions) if precisions else 0.0
    
    elif average == 'micro':
        # Micro-averaged precision: overall precision
        tp_total = np.trace(cm)  # Sum of diagonal (all true positives)
        fp_total = np.sum(cm) - tp_total  # All false positives
        if tp_total + fp_total > 0:
            return tp_total / (tp_total + fp_total)
        return 0.0
    
    else:
        raise ValueError(f"Unknown average type: {average}")


def recall(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> float:
    """
    Calculate recall score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: 'macro' for macro-averaged recall, 'micro' for micro-averaged
        
    Returns:
        Recall score
    """
    num_classes = max(np.max(y_true), np.max(y_pred)) + 1
    cm = confusion_matrix(y_true, y_pred, num_classes)
    
    if average == 'macro':
        # Macro-averaged recall: average recall for each class
        recalls = []
        for i in range(num_classes):
            tp = cm[i, i]  # True positives
            fn = np.sum(cm[i, :]) - tp  # False negatives
            if tp + fn > 0:
                recalls.append(tp / (tp + fn))
            else:
                recalls.append(0.0)
        return np.mean(recalls) if recalls else 0.0
    
    elif average == 'micro':
        # Micro-averaged recall: overall recall (same as accuracy for multiclass)
        tp_total = np.trace(cm)  # Sum of diagonal
        fn_total = np.sum(cm) - tp_total  # All false negatives
        if tp_total + fn_total > 0:
            return tp_total / (tp_total + fn_total)
        return 0.0
    
    else:
        raise ValueError(f"Unknown average type: {average}")


def f1_score(y_true: np.ndarray, y_pred: np.ndarray, average: str = 'macro') -> float:
    """
    Calculate F1 score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        average: 'macro' for macro-averaged F1, 'micro' for micro-averaged
        
    Returns:
        F1 score
    """
    prec = precision(y_true, y_pred, average)
    rec = recall(y_true, y_pred, average)
    
    if prec + rec > 0:
        return 2 * (prec * rec) / (prec + rec)
    return 0.0


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate accuracy score.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy score
    """
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total if total > 0 else 0.0


def classification_report(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    """
    Generate classification report with all metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with metrics
    """
    num_classes = max(np.max(y_true), np.max(y_pred)) + 1
    cm = confusion_matrix(y_true, y_pred, num_classes)
    
    report = {
        'accuracy': accuracy(y_true, y_pred),
        'precision_macro': precision(y_true, y_pred, average='macro'),
        'precision_micro': precision(y_true, y_pred, average='micro'),
        'recall_macro': recall(y_true, y_pred, average='macro'),
        'recall_micro': recall(y_true, y_pred, average='micro'),
        'f1_macro': f1_score(y_true, y_pred, average='macro'),
        'f1_micro': f1_score(y_true, y_pred, average='micro'),
        'confusion_matrix': cm,
        'per_class_metrics': {}
    }
    
    # Per-class metrics
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        tn = np.sum(cm) - tp - fp - fn
        
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        
        report['per_class_metrics'][i] = {
            'precision': prec,
            'recall': rec,
            'f1_score': f1,
            'support': int(tp + fn)
        }
    
    return report

