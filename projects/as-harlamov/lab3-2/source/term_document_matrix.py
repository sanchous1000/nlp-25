from collections import Counter, defaultdict
from typing import List, Tuple

import numpy as np
from scipy.sparse import csr_matrix


class SimpleVectorizer:
    def __init__(self, min_df=2, max_df=0.95):
        self.min_df = min_df
        self.max_df = max_df
        self.vocabulary_ = {}
        self.feature_names_ = []

    def fit(self, documents: List[List[str]]):
        if not documents:
            raise ValueError("Список документов пуст")

        documents = [doc for doc in documents if doc]
        if not documents:
            raise ValueError("Все документы пусты после фильтрации")

        term_counts = Counter()
        doc_freq = defaultdict(int)

        for doc in documents:
            unique_terms = set(doc)
            for term in unique_terms:
                doc_freq[term] += 1
            for term in doc:
                term_counts[term] += 1

        n_docs = len(documents)
        min_doc_freq = max(1, int(self.min_df)) if isinstance(self.min_df, int) else max(1, int(self.min_df * n_docs))
        max_doc_freq = int(self.max_df * n_docs) if isinstance(self.max_df, float) else n_docs

        valid_terms = []
        for term, count in term_counts.items():
            df = doc_freq[term]
            if df >= min_doc_freq and df <= max_doc_freq:
                valid_terms.append((term, count))

        if not valid_terms:
            raise ValueError(
                f"После фильтрации (min_df={min_doc_freq}, max_df={max_doc_freq}) "
                f"не осталось терминов. Всего уникальных терминов: {len(term_counts)}",
            )

        valid_terms.sort(key=lambda x: x[1], reverse=True)
        self.feature_names_ = [term for term, _ in valid_terms]
        self.vocabulary_ = {term: idx for idx, term in enumerate(self.feature_names_)}

        return self

    def transform(self, documents: List[List[str]]) -> csr_matrix:
        n_docs = len(documents)
        n_features = len(self.vocabulary_)

        rows = []
        cols = []
        data = []

        for doc_idx, doc in enumerate(documents):
            term_counts = Counter(doc)
            for term, count in term_counts.items():
                if term in self.vocabulary_:
                    feature_idx = self.vocabulary_[term]
                    rows.append(doc_idx)
                    cols.append(feature_idx)
                    data.append(count)

        matrix = csr_matrix((data, (rows, cols)), shape=(n_docs, n_features))
        return matrix

    def fit_transform(self, documents: List[List[str]]) -> csr_matrix:
        return self.fit(documents).transform(documents)

    def get_feature_names_out(self) -> np.ndarray:
        return np.array(self.feature_names_)


def build_term_document_matrix(
    documents: List[List[str]],
    min_df: int = 2,
    max_df: float = 0.95,
) -> Tuple[csr_matrix, List[str], SimpleVectorizer]:
    non_empty_docs = [doc for doc in documents if doc]
    if not non_empty_docs:
        raise ValueError("Все документы пусты")

    print(f"  Непустых документов: {len(non_empty_docs)} из {len(documents)}")

    vectorizer = None
    matrix = None
    feature_names = []

    current_min_df = min_df
    max_attempts = 3

    for attempt in range(max_attempts):
        try:
            vectorizer = SimpleVectorizer(min_df=current_min_df, max_df=max_df)
            matrix = vectorizer.fit_transform(non_empty_docs)
            feature_names = vectorizer.get_feature_names_out().tolist()

            if matrix.shape[0] > 0 and matrix.shape[1] > 0 and matrix.nnz > 0:
                if attempt > 0:
                    print(f"  Успешно построена матрица с min_df={current_min_df}")
                break
            else:
                raise ValueError("Матрица пуста")
        except ValueError as e:
            if attempt < max_attempts - 1:
                if isinstance(current_min_df, int):
                    current_min_df = max(1, current_min_df - 1)
                else:
                    current_min_df = max(0.001, current_min_df * 0.5)
                print(f"  Попытка {attempt + 1} не удалась, уменьшаем min_df до {current_min_df}")
            else:
                raise ValueError(f"Не удалось построить матрицу после {max_attempts} попыток: {e}")

    if matrix is None or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError(f"Получена пустая матрица: {matrix.shape if matrix is not None else 'None'}")

    print(
        f"  Размерность матрицы: {matrix.shape}, количество признаков: {len(feature_names)}, ненулевых элементов: {matrix.nnz}",
    )

    return matrix, feature_names, vectorizer


def build_term_document_matrix_from_texts(
    texts: List[str],
    tokenizer_func,
    min_df: int = 2,
    max_df: float = 0.95,
) -> Tuple[csr_matrix, List[str], SimpleVectorizer]:
    tokenized_docs = [tokenizer_func(text) for text in texts]
    return build_term_document_matrix(tokenized_docs, min_df=min_df, max_df=max_df)
