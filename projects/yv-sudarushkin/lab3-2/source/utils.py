from typing import List, Dict, Tuple

import numpy as np
from pathlib import Path
from sklearn.decomposition import LatentDirichletAllocation

def load_tdm(path: Path) -> Tuple[List[str], np.ndarray]:
    doc_ids = []
    rows = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split('\t')
            doc_ids.append(parts[0])
            rows.append([float(x) for x in parts[1:]])

    return doc_ids, np.array(rows)

def load_vocab(path: Path) -> List[str]:
    with open(path, encoding="utf-8") as f:
        return [line.strip() for line in f]


def save_doc_topic(doc_ids, probs, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        for doc_id, row in zip(doc_ids, probs):
            values = "\t".join(f"{x:.6f}" for x in row)
            f.write(f"{doc_id}\t{values}\n")

def run_lda(
    X_train: np.ndarray,
    X_test: np.ndarray,
    n_topics: int,
    max_iter: int = 10,
    random_state: int = 42
):
    """
    Обучает LDA и возвращает модель и perplexity на тесте
    """
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=max_iter,
        random_state=random_state,
        learning_method="batch"
    )

    lda.fit(X_train)
    perplexity = lda.perplexity(X_test)

    return lda, perplexity


def get_top_words(
    lda: LatentDirichletAllocation,
    vocab: List[str],
    top_k: int = 10
) -> List[List[str]]:
    """
    Возвращает top_k слов для каждой темы
    """
    topics = []

    for topic in lda.components_:
        top_indices = topic.argsort()[-top_k:][::-1]
        topics.append([vocab[i] for i in top_indices])

    return topics


def train_lda_experiments(
    X_train: np.ndarray,
    train_ids: List[str],
    X_test: np.ndarray,
    vocab: List[str],
    topic_counts: List[int],
    output_dir: Path,
    max_iter: List[int] = [10],
    need_safe = True
) -> List[Dict]:
    """
    Запускает серию экспериментов LDA с разным числом тем
    """
    results = []
    output_dir.mkdir(parents=True, exist_ok=True)

    for k in topic_counts:
        for iter in max_iter:

            lda, perplexity = run_lda(
                X_train,
                X_test,
                n_topics=k,
                max_iter=iter
            )

            topics = get_top_words(lda, vocab)
            doc_probs = lda.transform(X_train)

            doc_topic_path = output_dir / f"doc_topic_k{k}.tsv" if need_safe else None
            if doc_topic_path is not None:
                save_doc_topic(train_ids, doc_probs, doc_topic_path)

            results.append({
                "k": k,
                "max_iter" : iter,
                "perplexity": perplexity,
                "topics": topics,
                "doc_topic_path": doc_topic_path
            })

            print(f"[LDA] k={k}, perplexity={perplexity:.2f}")

    return results
