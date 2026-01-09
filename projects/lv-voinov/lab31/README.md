## Отчет по 3 лабораторной

Код подсчета метрик:

```

def calculate_metrics(y_true, y_pred):
    classes = np.unique(y_true)
    n_classes = len(classes)
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    class_to_idx = {cls: i for i, cls in enumerate(classes)}
    
    for true, pred in zip(y_true, y_pred):
        cm[class_to_idx[true]][class_to_idx[pred]] += 1
    
    precision_sum = 0
    recall_sum = 0
    
    for i in range(n_classes):
        tp = cm[i][i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        precision_sum += precision
        recall_sum += recall
    
    precision = precision_sum / n_classes
    recall = recall_sum / n_classes
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = np.trace(cm) / np.sum(cm)
    
    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'accuracy': accuracy,
        'confusion_matrix': cm
    }
```

Была обучена модель SVM для классификации с ядром rbf и различным количеством эпох: 250, 500, 1000, 2000, 4000, 8000. Результаты по метрикам:

```
max_iter= 250: Accuracy=0.5455, F1=0.5638, Precision=0.5834, Recall=0.5455, Time=14.30с
max_iter= 500: Accuracy=0.6270, F1=0.6528, Precision=0.6807, Recall=0.6270, Time=27.14с
max_iter=1000: Accuracy=0.6795, F1=0.6936, Precision=0.7082, Recall=0.6795, Time=59.01с
max_iter=2000: Accuracy=0.7559, F1=0.7599, Precision=0.7638, Recall=0.7559, Time=108.47с
max_iter=4000: Accuracy=0.8471, F1=0.8497, Precision=0.8522, Recall=0.8471, Time=187.43с
max_iter=8000: Accuracy=0.8893, F1=0.8894, Precision=0.8895, Recall=0.8893, Time=291.68с
```

Оптимальное количество эпох - 8000, значение может быть еще выше.

Эксперименты с отбрасыванием элементов векторных представлений были проведены для 1000 эпох из-за слишком большого времени обучения.

Результаты экспериментов:

```
Отброшено   1% размерностей: Accuracy=0.6859, F1=0.6893, Precision=0.6927, Recall=0.6859, Time=54.77с
Отброшено   2% размерностей: Accuracy=0.7093, F1=0.7096, Precision=0.7099, Recall=0.7093, Time=53.30с
Отброшено   4% размерностей: Accuracy=0.7076, F1=0.7113, Precision=0.7150, Recall=0.7076, Time=51.70с
Отброшено   8% размерностей: Accuracy=0.7014, F1=0.7076, Precision=0.7139, Recall=0.7014, Time=59.19с
Отброшено  16% размерностей: Accuracy=0.6672, F1=0.6732, Precision=0.6792, Recall=0.6672, Time=49.43с
Отброшено  32% размерностей: Accuracy=0.6776, F1=0.6829, Precision=0.6883, Recall=0.6776, Time=45.09с
Отброшено  64% размерностей: Accuracy=0.5804, F1=0.5918, Precision=0.6036, Recall=0.5804, Time=33.14с
```

Оптимальный процент отброшенных размерностей - 4%. В векторах, полученных во 2 лабораторной, 100 элементов, поэтому оптимальное количество по результатам экспериментов - 96.
