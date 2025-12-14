import numpy as np

def calculate_metrics_per_class(y_true, y_pred, num_classes=4):
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(y_true)):
        confusion_matrix[y_true[i]-1][y_pred[i]-1] += 1

    TP = np.diag(confusion_matrix)
    FP = np.sum(confusion_matrix, axis=0) - TP
    FN = np.sum(confusion_matrix, axis=1) - TP

    precision = TP / (TP + FP) if not np.any(TP + FP) == 0 else 0
    recall = TP / (TP + FN) if not np.any(TP + FN) == 0 else 0  
    f1 = 2 * (precision * recall) / (precision + recall) if not np.any(precision + recall) == 0 else 0
    accuracy = np.trace(confusion_matrix) / np.sum(confusion_matrix)

    return precision, recall, f1, accuracy

def classification_report(precision, recall, f1, accuracy):
    headers = ["Class", "Precision", "Recall", "F1"]
    rows = []
    for idx in range(len(precision)):
        row = [f"{idx+1}", round(precision[idx], 2), round(recall[idx], 2), round(f1[idx], 2)]
        rows.append(row)
    rows.append(["Accuracy", "", "", round(accuracy, 2)])
    col_width = max(len(str(word)) for row in rows for word in row) + 2
    print(" ".join([header.ljust(col_width) for header in headers]))
    for row in rows:
        print(" ".join([str(item).ljust(col_width) for item in row]))