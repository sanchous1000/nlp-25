import time
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np


def load_embeddings_tsv(path: Path) -> Dict[str, np.ndarray]:
    """
    Загружает эмбеддинги документов из TSV-файла (lab2).
    Формат строки:
    doc_id \t v1 \t v2 \t ... \t vN
    """
    embeddings = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            doc_id = parts[0]
            vector = np.array([float(x) for x in parts[1:]], dtype=np.float32)
            embeddings[doc_id] = vector

    return embeddings


def build_dataset(
    embeddings: Dict[str, np.ndarray],
    corpus_dir: Path
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Формирует X, y на основе структуры:
    train/
      class1/*.tsv
      class2/*.tsv
    """
    X, y = [], []
    label_names = []

    for label_id, class_dir in enumerate(sorted(corpus_dir.iterdir())):
        if not class_dir.is_dir():
            continue

        label_names.append(class_dir.name)

        for doc_file in class_dir.glob("*.tsv"):
            doc_id = doc_file.stem
            if doc_id not in embeddings:
                continue

            X.append(embeddings[doc_id])
            y.append(label_id)

    return np.array(X), np.array(y), label_names

def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int
) -> Dict[str, float]:
    eps = 1e-9

    precisions, recalls, f1s = [], [], []

    for c in range(num_classes):
        tp = ((y_pred == c) & (y_true == c)).sum()
        fp = ((y_pred == c) & (y_true != c)).sum()
        fn = ((y_pred != c) & (y_true == c)).sum()

        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        f1 = 2 * precision * recall / (precision + recall + eps)

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    accuracy = (y_pred == y_true).mean()

    return {
        "precision": float(np.mean(precisions)),
        "recall": float(np.mean(recalls)),
        "f1": float(np.mean(f1s)),
        "accuracy": float(accuracy),
    }


from sklearn.svm import LinearSVC, SVC

def run_linear_svm(
    X_train, y_train,
    X_test, y_test,
    max_iter: int
) -> Dict[str, float]:
    model = LinearSVC(max_iter=max_iter)

    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start

    y_pred = model.predict(X_test)

    metrics = classification_metrics(
        y_test, y_pred, num_classes=len(set(y_train))
    )

    metrics.update({
        "model": "LinearSVM",
        "kernel": "linear",
        "max_iter": max_iter,
        "n_iter": int(model.n_iter_),
        "training_time": training_time,
    })

    return metrics

def run_rbf_svm(
    X_train, y_train,
    X_test, y_test,
    max_iter: int
) -> Dict[str, float]:
    model = SVC(kernel="rbf", max_iter=max_iter)

    start = time.time()
    model.fit(X_train, y_train)
    training_time = time.time() - start

    y_pred = model.predict(X_test)

    metrics = classification_metrics(
        y_test, y_pred, num_classes=len(set(y_train))
    )

    metrics.update({
        "model": "SVM",
        "kernel": "rbf",
        "max_iter": max_iter,
        "n_iter": int(np.max(model.n_iter_)),
        "training_time": training_time,
    })

    return metrics


def drop_dimensions(X: np.ndarray, new_dim: int) -> np.ndarray:
    """
    Отбрасывает последние размерности векторного представления
    """
    return X[:, :new_dim]

