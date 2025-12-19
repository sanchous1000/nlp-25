import numpy as np
from pathlib import Path

def get_unknown_vector(vector_size=30):
    return np.random.rand(vector_size) * 0.01

def read_and_group(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        sentence = []
        corpus = []

        for line in file:
            stripped_line = line.strip()
            if stripped_line != '':
                words = stripped_line.split("\t")
                sentence.append(words[0])
            else:
                if sentence:
                    corpus.append(sentence)
                sentence = []
        if sentence:
            corpus.append(sentence)
    return corpus


def read_all_files_from_directory(directory_path):
    corpus = []
    directory = Path(directory_path)
    
    if not directory.exists():
        raise FileNotFoundError(f"Директория не найдена: {directory_path}")

    tsv_files = list(directory.rglob("*.tsv"))
    if not tsv_files:
        raise FileNotFoundError(f"Не найдено TSV файлов в директории: {directory_path}")
    for tsv_file in tsv_files:
        file_corpus = read_and_group(str(tsv_file))
        corpus.extend(file_corpus)
    
    return corpus


def read_all_train_files(base_path="assets/annotated-corpus/train"):
    corpus = []
    base_dir = Path(base_path)

    for subdir in ['0', '1', '2', '3']:
        subdir_path = base_dir / subdir
        if subdir_path.exists():
            subdir_corpus = read_all_files_from_directory(str(subdir_path))
            corpus.extend(subdir_corpus)
    
    return corpus


def read_document_from_file(filepath):
    return read_and_group(filepath)


def weighted_average_vector(tokens, model, term_weights):
    vector_size = model.wv.vector_size
    unknown_vector = get_unknown_vector(vector_size)
    vectors = []
    total_weight = 0

    for token in tokens:
        try:
            token_embedding = model.wv[token]
        except KeyError:
            token_embedding = unknown_vector
        weight = term_weights.get(token.lower(), 0)
        vectors.append(token_embedding * weight)
        total_weight += weight

    if total_weight != 0:
        return np.array(vectors).sum(axis=0) / total_weight
    else:
        vectors = []
        for token in tokens:
            try:
                token_embedding = model.wv[token]
            except KeyError:
                token_embedding = unknown_vector
            vectors.append(token_embedding)
        if vectors:
            return np.array(vectors).mean(axis=0)
        else:
            return unknown_vector


def vectorize_document(filepath, model, term_weights):
    vector_size = model.wv.vector_size
    unknown_vector = get_unknown_vector(vector_size)

    sentences = read_document_from_file(filepath)
    
    if not sentences:
        return unknown_vector

    sentence_vectors = []
    for sentence in sentences:
        if sentence:
            sentence_vector = weighted_average_vector(sentence, model, term_weights)
            if sentence_vector is not None:
                sentence_vectors.append(sentence_vector)

    if sentence_vectors:
        document_vector = np.array(sentence_vectors).mean(axis=0)
        return document_vector
    else:
        return unknown_vector