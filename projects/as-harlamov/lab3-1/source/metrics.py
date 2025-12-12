import numpy as np
from typing import Dict

def calculate_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        if 0 <= true_label < num_classes and 0 <= pred_label < num_classes:
            cm[int(true_label)][int(pred_label)] += 1
        else:
            raise ValueError(
                f"Метка выходит за допустимый диапазон: "
                f"true_label={true_label}, pred_label={pred_label}, "
                f"num_classes={num_classes}. "
                f"Убедитесь, что метки нормализованы к диапазону [0, {num_classes-1}]"
            )
    return cm

def calculate_precision(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[int, float]:
    cm = calculate_confusion_matrix(y_true, y_pred, num_classes)
    precisions = {}
    
    for i in range(num_classes):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        
        if tp + fp == 0:
            precisions[i] = 0.0
        else:
            precisions[i] = tp / (tp + fp)
    
    return precisions

def calculate_recall(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[int, float]:
    cm = calculate_confusion_matrix(y_true, y_pred, num_classes)
    recalls = {}
    
    for i in range(num_classes):
        tp = cm[i, i]
        fn = np.sum(cm[i, :]) - tp
        
        if tp + fn == 0:
            recalls[i] = 0.0
        else:
            recalls[i] = tp / (tp + fn)
    
    return recalls

def calculate_f1_score(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict[int, float]:
    precisions = calculate_precision(y_true, y_pred, num_classes)
    recalls = calculate_recall(y_true, y_pred, num_classes)
    f1_scores = {}
    
    for i in range(num_classes):
        p = precisions[i]
        r = recalls[i]
        if p + r == 0:
            f1_scores[i] = 0.0
        else:
            f1_scores[i] = 2 * (p * r) / (p + r)
    
    return f1_scores

def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total if total > 0 else 0.0

def calculate_macro_averages(
    y_true: np.ndarray, 
    y_pred: np.ndarray, 
    num_classes: int
) -> Dict[str, float]:
    precisions = calculate_precision(y_true, y_pred, num_classes)
    recalls = calculate_recall(y_true, y_pred, num_classes)
    f1_scores = calculate_f1_score(y_true, y_pred, num_classes)
    
    macro_precision = np.mean(list(precisions.values()))
    macro_recall = np.mean(list(recalls.values()))
    macro_f1 = np.mean(list(f1_scores.values()))
    
    return {
        'precision': macro_precision,
        'recall': macro_recall,
        'f1': macro_f1
    }

def calculate_all_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Dict:
    accuracy = calculate_accuracy(y_true, y_pred)
    macro_avg = calculate_macro_averages(y_true, y_pred, num_classes)
    
    precisions = calculate_precision(y_true, y_pred, num_classes)
    recalls = calculate_recall(y_true, y_pred, num_classes)
    f1_scores = calculate_f1_score(y_true, y_pred, num_classes)
    
    return {
        'accuracy': accuracy,
        'macro_precision': macro_avg['precision'],
        'macro_recall': macro_avg['recall'],
        'macro_f1': macro_avg['f1'],
        'per_class_precision': precisions,
        'per_class_recall': recalls,
        'per_class_f1': f1_scores,
        'confusion_matrix': calculate_confusion_matrix(y_true, y_pred, num_classes)
    }
