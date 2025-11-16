import re
import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from pathlib import Path


def advanced_tokenize(text):
    if not isinstance(text, str) or not text.strip():
        return []

    # Шаблоны для сложных случаев
    patterns = [
        (r'\b[a-zA-Z]+@[a-zA-Z]+\.[a-zA-Z]+\b', 'EMAIL'),
        (r'(?:\+7-\d{3}-\d{3}-\d{2}-\d{2}|\b8\(\d{3}\)\d{7}\b)', 'PHONE'),
        (r'\b(?:Dr|Mr|Mrs|Ms|Prof)\.\s+[A-Z][a-zA-Z]*', 'APPEAL'),
        (r'^[a-zA-Z_]\w*\s*=\s*[\w\s()+\-*/^]+(?:\s*=\s*[\w\s()+\-*/^]+)*$', 'MATH')
    ]

    placeholders = {}
    counter = 0

    for pattern, tag in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        for match in set(matches):
            placeholder = f"__{tag}_{counter}__"
            text = text.replace(match, placeholder)
            placeholders[placeholder] = match
            counter += 1

    # Разбиение на предложения
    sentences = re.split(r'[.!?]+', text)

    # Удаляем пустые предложения
    sentences = [sen.strip() for sen in sentences if sen.strip()]

    tokens_all = []
    for sent in sentences:
        # Токенизация простого текста
        tokens = re.findall(r"\w+(?:'\w+)?|[^\w\s]", sent)
        # Восстановление сложных токенов
        restored = []
        for token in tokens:
            if token in placeholders:
                restored.append(placeholders[token])
            else:
                restored.append(token)
        if restored:  # Добавляем только непустые предложения
            tokens_all.append(restored)

    return tokens_all


def vectorize_token(token, model):
    if token in model.wv:
        return model.wv[token]
    return None


def vectorize_sentence(sentence_tokens, model):
    if not sentence_tokens:
        return np.zeros(model.vector_size)

    vectors = []
    for token in sentence_tokens:
        vec = vectorize_token(token, model)
        if vec is not None:
            vectors.append(vec)

    if not vectors:
        return np.zeros(model.vector_size)

    # Среднее значение векторов токенов предложения
    sentence_vector = np.mean(vectors, axis=0)
    return sentence_vector


def vectorize_document(doc_sentences, model):
    if not doc_sentences:
        return np.zeros(model.vector_size)

    sentence_vectors = []
    for sentence_tokens in doc_sentences:
        if sentence_tokens:
            sent_vec = vectorize_sentence(sentence_tokens, model)
            sentence_vectors.append(sent_vec)

    if not sentence_vectors:
        return np.zeros(model.vector_size)

    # Среднее значение векторов предложений документа
    doc_vector = np.mean(sentence_vectors, axis=0)
    return doc_vector


def vectorize_text(text, model):
    doc_sentences = advanced_tokenize(str(text) if pd.notna(text) else "")

    doc_vector = vectorize_document(doc_sentences, model)
    return doc_vector


def vectorize_dataframe(df, model: Word2Vec):
    # Векторизация каждого документа
    doc_vectors = []

    for i, row in df.iterrows():
        text = row['text']
        label = row['class']
        emb = vectorize_text(str(text) if pd.notna(text) else "", model)

        doc_vectors.append({
            'doc_id': i,
            'label': label,
            'embedding': emb
        })

    save_annotated_corpus(doc_vectors, 'train')


def sanitize_label(label):
    sanitized = re.sub(r'[<>:"/\\|?*$]', '_', label)
    sanitized = sanitized.strip().strip('.')
    sanitized = re.sub(r'[_\s]+', '_', sanitized)
    return sanitized


def save_annotated_corpus(documents, mode):
    base_dir = Path(f"../../assets/annotated-corpus")

    output_path = base_dir / f"{mode}.tsv"

    with open(output_path, "w", encoding="utf-8") as f:
        for doc in documents:
            doc_id = sanitize_label(str(doc["doc_id"]))
            embedding = doc["embedding"]
            embedding_str = '\t'.join([f"{x:.6f}" for x in embedding])

            f.write(f"{doc_id}\t{embedding_str}\n")


if __name__ == '__main__':
    MODEL = Word2Vec.load('word2vec_model.model')

    column_names = ['class', 'title', 'text']
    DATA = pd.read_csv("../../train.csv", header=None, names=column_names)

    vectorize_dataframe(df=DATA, model=MODEL)
