from typing import Dict, List

import numpy as np


def calc_precision(y_true: np.ndarray, y_pred: np.ndarray, classes: List[int]) -> Dict[int, float]:
    result = {}
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fp = np.sum((y_pred == c) & (y_true != c))
        result[c] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    return result


def calc_recall(y_true: np.ndarray, y_pred: np.ndarray, classes: List[int]) -> Dict[int, float]:
    result = {}
    for c in classes:
        tp = np.sum((y_pred == c) & (y_true == c))
        fn = np.sum((y_pred != c) & (y_true == c))
        result[c] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return result


def calc_f1(precision: Dict[int, float], recall: Dict[int, float]) -> Dict[int, float]:
    result = {}
    for c in precision:
        p, r = precision[c], recall[c]
        result[c] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    return result


def calc_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sum(y_true == y_pred) / len(y_true)


def macro_avg(metric_dict: Dict[int, float]) -> float:
    return np.mean(list(metric_dict.values()))