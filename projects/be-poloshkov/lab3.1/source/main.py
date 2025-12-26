import random
import time
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from metrics import calc_precision, calc_recall, calc_f1, calc_accuracy, macro_avg

BASE_DIR = Path(__file__).parent.parent.parent
CORPUS_DIR = BASE_DIR / "assets" / "annotated-corpus"
EMBEDDINGS_DIR = BASE_DIR / "lab2" / "output"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

TRAIN_SIZE = 20000
TEST_SIZE = 5000


def load_embeddings(filepath: Path, sample_size: int = None):
    embeddings = {}
    with open(filepath, 'r') as f:
        lines = f.readlines()
    
    if sample_size and len(lines) > sample_size:
        random.seed(42)
        lines = random.sample(lines, sample_size)
    
    for line in lines:
        parts = line.strip().split('\t')
        doc_id = parts[0]
        vector = np.array([float(x) for x in parts[1:]])
        embeddings[doc_id] = vector
    return embeddings


def get_labels_from_corpus(corpus_dir: Path):
    labels = {}
    for class_dir in corpus_dir.iterdir():
        if not class_dir.is_dir():
            continue
        label = int(class_dir.name)
        for doc_file in class_dir.glob("*.tsv"):
            labels[doc_file.stem] = label
    return labels


def prepare_data(embeddings: dict, labels: dict):
    X, y, ids = [], [], []
    for doc_id, emb in embeddings.items():
        if doc_id in labels:
            X.append(emb)
            y.append(labels[doc_id])
            ids.append(doc_id)
    return np.array(X), np.array(y), ids


def run_experiment(X_train, y_train, X_test, y_test, max_iter, kernel='linear'):
    classes = sorted(list(set(y_train)))
    
    start = time.time()
    clf = SVC(kernel=kernel, max_iter=max_iter)
    clf.fit(X_train, y_train)
    train_time = time.time() - start
    
    y_pred = clf.predict(X_test)
    
    prec = calc_precision(y_test, y_pred, classes)
    rec = calc_recall(y_test, y_pred, classes)
    f1 = calc_f1(prec, rec)
    acc = calc_accuracy(y_test, y_pred)
    
    return {
        'max_iter': max_iter,
        'kernel': kernel,
        'precision': macro_avg(prec),
        'recall': macro_avg(rec),
        'f1': macro_avg(f1),
        'accuracy': acc,
        'train_time': train_time
    }


def drop_dims(X: np.ndarray, n_drop: int):
    if n_drop >= X.shape[1]:
        return X[:, :1]
    random.seed(42)
    drop_idx = set(random.sample(range(X.shape[1]), n_drop))
    keep_idx = [i for i in range(X.shape[1]) if i not in drop_idx]
    return X[:, keep_idx]


def main():
    train_emb = load_embeddings(EMBEDDINGS_DIR / "train_embeddings.tsv", TRAIN_SIZE)
    test_emb = load_embeddings(EMBEDDINGS_DIR / "test_embeddings.tsv", TEST_SIZE)
    
    train_labels = get_labels_from_corpus(CORPUS_DIR / "train")
    test_labels = get_labels_from_corpus(CORPUS_DIR / "test")
    
    X_train, y_train, _ = prepare_data(train_emb, train_labels)
    X_test, y_test, _ = prepare_data(test_emb, test_labels)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    print("\nIterations start")
    iters = [100, 500, 1000, 2000, 5000]
    results_iter = []
    
    for mi in iters:
        r = run_experiment(X_train, y_train, X_test, y_test, mi)
        results_iter.append(r)
        print(f"iter={mi}: acc={r['accuracy']:.4f}, f1={r['f1']:.4f}, t={r['train_time']:.1f}s")
    
    best_iter = max(results_iter, key=lambda x: x['f1'])['max_iter']
    print(f"Best iter: {best_iter}")
    
    print("\nKernels Start---")
    kernels = ['linear', 'rbf', 'poly']
    results_kern = []
    
    for k in kernels:
        r = run_experiment(X_train, y_train, X_test, y_test, best_iter, k)
        results_kern.append(r)
        print(f"kernel={k}: acc={r['accuracy']:.4f}, f1={r['f1']:.4f}, t={r['train_time']:.1f}s")
    
    best_kern = max(results_kern, key=lambda x: x['f1'])['kernel']
    print(f"Best kernel: {best_kern}")
    
    print("\nDimensions Drop Start")
    dims = X_train.shape[1]
    drops = [0, dims//10, dims//5, dims//4, dims//3, dims//2, dims*2//3, dims*3//4]
    results_dim = []
    
    for nd in drops:
        Xtr = drop_dims(X_train, nd)
        Xte = drop_dims(X_test, nd)
        r = run_experiment(Xtr, y_train, Xte, y_test, best_iter, best_kern)
        r['drop'] = nd
        r['left'] = Xtr.shape[1]
        results_dim.append(r)
        print(f"drop={nd}, left={r['left']}: acc={r['accuracy']:.4f}, f1={r['f1']:.4f}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "results.txt", 'w') as f:
        f.write("Iterations\n")
        f.write("iter\tprec\trec\tf1\tacc\ttime\n")
        for r in results_iter:
            f.write(f"{r['max_iter']}\t{r['precision']:.4f}\t{r['recall']:.4f}\t{r['f1']:.4f}\t{r['accuracy']:.4f}\t{r['train_time']:.1f}\n")
        f.write(f"Best: {best_iter}\n\n")
        
        f.write("Kernels\n")
        f.write("kernel\tprec\trec\tf1\tacc\ttime\n")
        for r in results_kern:
            f.write(f"{r['kernel']}\t{r['precision']:.4f}\t{r['recall']:.4f}\t{r['f1']:.4f}\t{r['accuracy']:.4f}\t{r['train_time']:.1f}\n")
        f.write(f"Best: {best_kern}\n\n")
        
        f.write("Dimensions Drop\n")
        f.write("drop\tleft\tprec\trec\tf1\tacc\n")
        for r in results_dim:
            f.write(f"{r['drop']}\t{r['left']}\t{r['precision']:.4f}\t{r['recall']:.4f}\t{r['f1']:.4f}\t{r['accuracy']:.4f}\n")


if __name__ == "__main__":
    main()
