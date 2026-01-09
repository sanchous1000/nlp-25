import numpy as np


def calculate_metrics_per_class(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes))
    for i in range(len(y_true)):
        cm[y_true[i]-1, y_pred[i]-1] += 1

    TP = np.diag(cm)
    FP = np.sum(cm, axis=0) - TP
    FN = np.sum(cm, axis=1) - TP
    
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1_score = 2 * (precision * recall) / (precision + recall)
    accuracy = np.trace(cm) / np.sum(cm)
    
    return precision, recall, f1_score, accuracy

def classification_report(precision, recall, f1_score, accuracy):
    headers = ["Class", "Precision", "Recall", "F1"]
    rows = []
    for idx in range(len(precision)):
        row = [
            f"{idx+1}",
            round(precision[idx], 4),
            round(recall[idx], 4),
            round(f1_score[idx], 4)
        ]
        rows.append(row)
    
    rows.append(["Accuracy", "", "", round(accuracy, 4)])
    
    col_width = max(len(str(word)) for row in rows for word in row) + 2
    print(" ".join([header.ljust(col_width) for header in headers]))
    for row in rows:
        print(" ".join([str(item).ljust(col_width) for item in row]))

