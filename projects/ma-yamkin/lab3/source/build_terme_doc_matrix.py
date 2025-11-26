import os
import re
import numpy as np
from collections import defaultdict, Counter
from tqdm import tqdm
from scipy.sparse import csr_matrix
import pickle
from nltk.corpus import stopwords


english_stopwords = set(stopwords.words('english'))


def read_annotated_corpus():
    train_dir = os.path.join('..', '..', 'assets', 'annotated-corpus', 'train')
    test_dir = os.path.join('..', '..', 'assets', 'annotated-corpus', 'test')

    docs_train = read_mode(train_dir)
    docs_test = read_mode(test_dir)

    for doc in docs_test:
        docs_train.append(doc)

    return docs_train


def read_mode(dir):
    subdirs = ['1', '2', '3', '4']
    documents = []

    for subdir in subdirs:
        subdir_path = os.path.join(dir, subdir)
        if not os.path.exists(subdir_path):
            print(f"Папка не найдена: {subdir_path}")
            continue

        tsv_files = [f for f in os.listdir(subdir_path) if f.endswith('.tsv')]

        for filename in tqdm(tsv_files, desc=f"Чтение файлов из {subdir}", leave=False):
            filepath = os.path.join(subdir_path, filename)

            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            if not content:
                continue

            raw_sentences = [s.strip() for s in content.split('\n\n') if s.strip()]
            sentences = []

            for raw_sentence in raw_sentences:
                tokens = []
                lines = [l.strip() for l in raw_sentence.split('\n') if l.strip()]

                for line in lines:
                    parts = [p.strip() for p in line.split('\t') if p.strip()]
                    if parts:
                        token = parts[0]
                        if token:
                            # Удалим пунктуацию и приведём к нижнему регистру
                            token_clean = re.sub(r'[^\w\s]', '', token.lower())
                            if token_clean:
                                tokens.append(token_clean)

                sentences.append(tokens)

            document = []
            for sent in sentences:
                document.extend(sent)

            documents.append(document)

    return documents


def filter_corpus(sentences, min_df=3):
    print('Начинаем фильтрацию')
    all_tokens = []
    filtered_sentences = []

    for sent in sentences:
        filtered = [
            token for token in sent
            if token not in english_stopwords and len(token) > 1
        ]
        filtered_sentences.append(filtered)
        all_tokens.extend(filtered)

    token_counts = Counter(all_tokens)

    allowed_tokens = {tok for tok, cnt in token_counts.items() if cnt >= min_df}

    print(f"Всего уникальных токенов до фильтрации: {len(token_counts)}")
    print(f"Оставлено после min_df={min_df}: {len(allowed_tokens)}")

    # Финальная фильтрация документов
    final_sentences = []
    for sent in filtered_sentences:
        filtered = [tok for tok in sent if tok in allowed_tokens]
        final_sentences.append(filtered)

    print(f"Финальная длина выборки: {len(final_sentences)}")

    return final_sentences


def build_term_document_matrix(sentences, output_dir, min_df):
    """
    Строит матрицу термин–документ с предварительной фильтрацией.
    """
    # Фильтрация корпуса
    filtered_sentences = filter_corpus(sentences, min_df=min_df)

    os.makedirs(output_dir, exist_ok=True)

    print("Сбор словаря...")
    term_freq = defaultdict(int)
    for sent in filtered_sentences:
        for term in set(sent):  # уникальные термины в документе для df
            term_freq[term] += 1

    terms = sorted(term_freq.keys())
    term_to_index = {term: idx for idx, term in enumerate(terms)}

    n_terms = len(terms)
    n_docs = len(filtered_sentences)

    print(f"Итог: терминов = {n_terms}, документов = {n_docs}")

    rows, cols, data = [], [], []

    for doc_idx, doc in enumerate(tqdm(filtered_sentences, desc="Построение матрицы")):
        term_counts = Counter(doc)
        for term, count in term_counts.items():
            term_idx = term_to_index[term]
            rows.append(term_idx)
            cols.append(doc_idx)
            data.append(count)

    td_matrix = csr_matrix((data, (rows, cols)), shape=(n_terms, n_docs), dtype=np.int32)

    # Сохранение
    print("Сохранение...")
    np.savez_compressed(
        os.path.join(output_dir, 'term_document_matrix.npz'),
        data=td_matrix.data,
        indices=td_matrix.indices,
        indptr=td_matrix.indptr,
        shape=td_matrix.shape
    )

    with open(os.path.join(output_dir, 'term_to_index.pkl'), 'wb') as f:
        pickle.dump(term_to_index, f)

    with open(os.path.join(output_dir, 'documents_filtered.pkl'), 'wb') as f:
        pickle.dump(filtered_sentences, f)

    with open(os.path.join(output_dir, 'terms.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(terms))

    print(f"Матрица сохранена в: {os.path.abspath(output_dir)}")


if __name__ == "__main__":
    mode = 'train'
    sentences = read_annotated_corpus()
    build_term_document_matrix(sentences, f'../../assets/annotated-corpus/term_doc_matrix_{mode}', min_df=10)
