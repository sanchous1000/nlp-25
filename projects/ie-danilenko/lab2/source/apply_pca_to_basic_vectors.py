"""
Скрипт для применения метода сокращения размерности (PCA) к векторам,
полученным базовыми способами векторизации.

Реализует следующие методы:
1. Кодирование текста в виде последовательности частот токенов
2. Кодирование текста в виде единичной матрицы (one-hot)
3. Кодирование текста в виде матрицы частот токенов
4. Кодирование текста в виде наборов значений метрики tf-idf
5. Кодирование текста в виде наборов значений частот токенов и tf-idf по предложениям
"""

import numpy as np
import argparse
from pathlib import Path
from collections import Counter
from sklearn.decomposition import PCA
import pickle
from text_to_tfidf import (
    tokenize_text,
    load_vocabulary,
    load_term_document_matrix,
    compute_tf,
    compute_idf
)
from build_vocabulary_and_matrix import read_annotation_file


def text_to_frequency_vector(
    text,
    vocabulary
):
    """
    Преобразует текст в вектор частот токенов.
    
    Args:
        text: Входной текст
        vocabulary: Словарь токен -> индекс
        
    Returns:
        Вектор частот размером равным размеру словаря
    """
    tokens = tokenize_text(text)
    if not tokens:
        return np.zeros(len(vocabulary))
    
    # Подсчитываем частоты
    token_counter = Counter(tokens)
    vector = np.zeros(len(vocabulary))
    
    for token, count in token_counter.items():
        if token in vocabulary:
            idx = vocabulary[token]
            vector[idx] = count
    
    # Нормализуем по длине текста
    total_tokens = len(tokens)
    if total_tokens > 0:
        vector = vector / total_tokens
    
    return vector


def text_to_onehot_matrix(
    text,
    vocabulary
):
    """
    Преобразует текст в единичную матрицу (one-hot encoding).
    Каждая строка соответствует одному токену, каждый столбец - токену словаря.
    
    Args:
        text: Входной текст
        vocabulary: Словарь токен -> индекс
        
    Returns:
        Матрица размером (количество_токенов, размер_словаря)
    """
    tokens = tokenize_text(text)
    if not tokens:
        return np.zeros((1, len(vocabulary)))
    
    matrix = np.zeros((len(tokens), len(vocabulary)))
    
    for i, token in enumerate(tokens):
        if token in vocabulary:
            idx = vocabulary[token]
            matrix[i, idx] = 1.0
    
    return matrix


def onehot_matrix_to_vector(matrix, method='mean'):
    """
    Преобразует единичную матрицу в вектор.
    
    Args:
        matrix: Единичная матрица (количество_токенов, размер_словаря)
        method: Способ преобразования ('mean', 'sum', 'max')
        
    Returns:
        Вектор размером равным размеру словаря
    """
    if method == 'mean':
        return np.mean(matrix, axis=0)
    elif method == 'sum':
        return np.sum(matrix, axis=0)
    elif method == 'max':
        return np.max(matrix, axis=0)
    else:
        raise ValueError(f"Неизвестный метод: {method}")


def text_to_frequency_matrix(
    text,
    vocabulary
):
    """
    Преобразует текст в матрицу частот токенов.
    Каждая строка соответствует одному токену входного текста.
    
    Args:
        text: Входной текст
        vocabulary: Словарь токен -> индекс
        
    Returns:
        Матрица размером (количество_токенов, размер_словаря)
    """
    tokens = tokenize_text(text)
    if not tokens:
        return np.zeros((1, len(vocabulary)))
    
    matrix = np.zeros((len(tokens), len(vocabulary)))
    
    for i, token in enumerate(tokens):
        if token in vocabulary:
            idx = vocabulary[token]
            matrix[i, idx] = 1.0  # Каждый токен встречается один раз в своей позиции
    
    return matrix


def frequency_matrix_to_vector(matrix, method='mean'):
    """
    Преобразует матрицу частот в вектор.
    
    Args:
        matrix: Матрица частот (количество_токенов, размер_словаря)
        method: Способ преобразования ('mean', 'sum', 'max')
        
    Returns:
        Вектор размером равным размеру словаря
    """
    return onehot_matrix_to_vector(matrix, method)


def text_to_tfidf_vector(
    text,
    vocabulary,
    num_docs,
    term_doc_counts
):
    """
    Преобразует текст в вектор значений tf-idf.
    
    Args:
        text: Входной текст
        vocabulary: Словарь токен -> индекс
        num_docs: Количество документов в корпусе
        term_doc_counts: Словарь term_index -> количество документов, содержащих термин
        
    Returns:
        Вектор tf-idf значений
    """
    tokens = tokenize_text(text)
    if not tokens:
        return np.zeros(len(vocabulary))
    
    token_counter = Counter(tokens)
    total_tokens = len(tokens)
    vector = np.zeros(len(vocabulary))
    
    for token, count in token_counter.items():
        if token in vocabulary:
            term_index = vocabulary[token]
            tf = compute_tf(count, total_tokens)
            idf = compute_idf(num_docs, term_doc_counts, term_index)
            tfidf = tf * idf
            vector[term_index] = tfidf
    
    return vector


def segment_text_to_sentences(text):
    """
    Сегментирует текст на предложения.
    Простая реализация: разбиение по знакам препинания.
    
    Args:
        text: Входной текст
        
    Returns:
        Список предложений
    """
    import re
    # Разбиваем по точкам, восклицательным и вопросительным знакам
    sentences = re.split(r'[.!?]+\s+', text)
    # Убираем пустые предложения
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences if sentences else [text]


def text_to_sentence_vectors(
    text,
    vocabulary,
    num_docs,
    term_doc_counts
):
    """
    Преобразует текст в векторы частот и tf-idf по предложениям.
    
    Args:
        text: Входной текст
        vocabulary: Словарь токен -> индекс
        num_docs: Количество документов в корпусе
        term_doc_counts: Словарь term_index -> количество документов, содержащих термин
        
    Returns:
        tuple: (матрица частот по предложениям, матрица tf-idf по предложениям)
    """
    sentences = segment_text_to_sentences(text)
    
    if not sentences:
        empty_vector = np.zeros(len(vocabulary))
        return np.array([empty_vector]), np.array([empty_vector])
    
    freq_matrix = []
    tfidf_matrix = []
    
    for sentence in sentences:
        # Вектор частот для предложения
        freq_vector = text_to_frequency_vector(sentence, vocabulary)
        freq_matrix.append(freq_vector)
        
        # Вектор tf-idf для предложения
        tfidf_vector = text_to_tfidf_vector(sentence, vocabulary, num_docs, term_doc_counts)
        tfidf_matrix.append(tfidf_vector)
    
    return np.array(freq_matrix), np.array(tfidf_matrix)


def sentence_vectors_to_document_vector(
    sentence_vectors,
    method='mean'
):
    """
    Преобразует векторы предложений в вектор документа.
    
    Args:
        sentence_vectors: Матрица векторов предложений (количество_предложений, размер_вектора)
        method: Способ преобразования ('mean', 'sum', 'max')
        
    Returns:
        Вектор документа
    """
    if method == 'mean':
        return np.mean(sentence_vectors, axis=0)
    elif method == 'sum':
        return np.sum(sentence_vectors, axis=0)
    elif method == 'max':
        return np.max(sentence_vectors, axis=0)
    else:
        raise ValueError(f"Неизвестный метод: {method}")


def collect_training_texts(train_dir, max_docs=None):
    """
    Собирает тексты из обучающей выборки для обучения PCA.
    
    Args:
        train_dir: Путь к директории с обучающей выборкой
        max_docs: Максимальное количество документов для выборки (None - все)
        
    Returns:
        Список текстов (каждый текст - объединение всех токенов документа)
    """
    texts = []
    
    if not train_dir.exists():
        raise FileNotFoundError(f"Директория {train_dir} не найдена")
    
    doc_count = 0
    for class_dir in sorted(train_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        
        for tsv_file in sorted(class_dir.glob("*.tsv")):
            if max_docs and doc_count >= max_docs:
                break
            
            tokens = read_annotation_file(tsv_file)
            if tokens:
                # Объединяем токены в текст
                text = ' '.join(tokens)
                texts.append(text)
                doc_count += 1
        
        if max_docs and doc_count >= max_docs:
            break
    
    return texts


def train_pca_model(
    vectors,
    target_dim,
    method_name
):
    """
    Обучает модель PCA на векторах.
    
    Args:
        vectors: Матрица векторов (количество_документов, размерность)
        target_dim: Целевая размерность
        method_name: Название метода (для вывода информации)
        
    Returns:
        Обученная модель PCA
    """
    print(f"  Обучение PCA для {method_name}...")
    print(f"    Исходная размерность: {vectors.shape[1]}")
    print(f"    Целевая размерность: {target_dim}")
    print(f"    Количество векторов: {vectors.shape[0]}")
    
    pca = PCA(n_components=target_dim)
    pca.fit(vectors)
    
    explained_variance = np.sum(pca.explained_variance_ratio_) * 100
    print(f"    Объясненная дисперсия: {explained_variance:.2f}%")
    
    return pca


def apply_pca_to_basic_vectors(
    train_dir,
    vocab_path,
    matrix_path,
    target_dim=100,
    max_docs=1000,
    output_dir=None
):
    """
    Применяет PCA к различным методам базовой векторизации.
    
    Args:
        train_dir: Путь к директории с обучающей выборкой
        vocab_path: Путь к файлу словаря
        matrix_path: Путь к файлу матрицы "термин-документ"
        target_dim: Целевая размерность после PCA (по умолчанию 100, как у GloVe)
        max_docs: Максимальное количество документов для обучения PCA
        output_dir: Директория для сохранения моделей PCA
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "output"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("Применение PCA к базовым методам векторизации")
    print("=" * 80)
    
    # Загружаем словарь и матрицу
    print("\n1. Загрузка словаря и матрицы...")
    vocabulary = load_vocabulary(vocab_path)
    num_docs, term_doc_counts = load_term_document_matrix(matrix_path)
    print(f"   Размер словаря: {len(vocabulary)}")
    print(f"   Количество документов в корпусе: {num_docs}")
    
    # Собираем тексты для обучения
    print(f"\n2. Сбор текстов для обучения PCA (максимум {max_docs} документов)...")
    texts = collect_training_texts(train_dir, max_docs=max_docs)
    print(f"   Собрано {len(texts)} текстов")
    
    # Создаем векторы для каждого метода
    print(f"\n3. Создание векторов для различных методов...")
    
    # 1. Частоты токенов
    print("\n3.1. Векторы частот токенов...")
    freq_vectors = []
    for text in texts:
        vec = text_to_frequency_vector(text, vocabulary)
        freq_vectors.append(vec)
    freq_vectors = np.array(freq_vectors)
    print(f"   Создано {len(freq_vectors)} векторов размерности {freq_vectors.shape[1]}")
    
    # 2. Единичная матрица (one-hot)
    print("\n3.2. Векторы из единичных матриц...")
    onehot_vectors = []
    for text in texts:
        matrix = text_to_onehot_matrix(text, vocabulary)
        vec = onehot_matrix_to_vector(matrix, method='mean')
        onehot_vectors.append(vec)
    onehot_vectors = np.array(onehot_vectors)
    print(f"   Создано {len(onehot_vectors)} векторов размерности {onehot_vectors.shape[1]}")
    
    # 3. Матрица частот токенов
    print("\n3.3. Векторы из матриц частот...")
    freq_matrix_vectors = []
    for text in texts:
        matrix = text_to_frequency_matrix(text, vocabulary)
        vec = frequency_matrix_to_vector(matrix, method='mean')
        freq_matrix_vectors.append(vec)
    freq_matrix_vectors = np.array(freq_matrix_vectors)
    print(f"   Создано {len(freq_matrix_vectors)} векторов размерности {freq_matrix_vectors.shape[1]}")
    
    # 4. TF-IDF
    print("\n3.4. Векторы TF-IDF...")
    tfidf_vectors = []
    for text in texts:
        vec = text_to_tfidf_vector(text, vocabulary, num_docs, term_doc_counts)
        tfidf_vectors.append(vec)
    tfidf_vectors = np.array(tfidf_vectors)
    print(f"   Создано {len(tfidf_vectors)} векторов размерности {tfidf_vectors.shape[1]}")
    
    # 5. Частоты и TF-IDF по предложениям
    print("\n3.5. Векторы из частот и TF-IDF по предложениям...")
    sentence_vectors = []
    for text in texts:
        freq_matrix, tfidf_matrix = text_to_sentence_vectors(
            text, vocabulary, num_docs, term_doc_counts
        )
        # Объединяем частоты и TF-IDF
        combined = np.concatenate([
            sentence_vectors_to_document_vector(freq_matrix, method='mean'),
            sentence_vectors_to_document_vector(tfidf_matrix, method='mean')
        ])
        sentence_vectors.append(combined)
    sentence_vectors = np.array(sentence_vectors)
    print(f"   Создано {len(sentence_vectors)} векторов размерности {sentence_vectors.shape[1]}")
    
    # Обучаем PCA для каждого метода
    print(f"\n4. Обучение моделей PCA (целевая размерность: {target_dim})...")
    
    pca_models = {}
    
    # PCA для частот
    pca_freq = train_pca_model(freq_vectors, target_dim, "частоты токенов")
    pca_models['frequency'] = pca_freq
    
    # PCA для one-hot
    pca_onehot = train_pca_model(onehot_vectors, target_dim, "единичная матрица")
    pca_models['onehot'] = pca_onehot
    
    # PCA для матрицы частот
    pca_freq_matrix = train_pca_model(freq_matrix_vectors, target_dim, "матрица частот")
    pca_models['frequency_matrix'] = pca_freq_matrix
    
    # PCA для TF-IDF
    pca_tfidf = train_pca_model(tfidf_vectors, target_dim, "TF-IDF")
    pca_models['tfidf'] = pca_tfidf
    
    # PCA для предложений
    pca_sentences = train_pca_model(sentence_vectors, target_dim, "частоты и TF-IDF по предложениям")
    pca_models['sentences'] = pca_sentences
    
    # Сохраняем модели PCA
    print(f"\n5. Сохранение моделей PCA...")
    for method_name, pca_model in pca_models.items():
        pca_path = output_dir / f"pca_{method_name}.pkl"
        with open(pca_path, 'wb') as f:
            pickle.dump(pca_model, f)
        print(f"   Сохранено: {pca_path}")
    
    print(f"\n{'='*80}")
    print("Готово! Модели PCA обучены и сохранены.")
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Применение PCA к базовым методам векторизации'
    )
    parser.add_argument(
        '--train-dir',
        type=str,
        default=None,
        help='Путь к директории с обучающей выборкой'
    )
    parser.add_argument(
        '--vocab',
        type=str,
        default=None,
        help='Путь к файлу словаря (vocabulary.json)'
    )
    parser.add_argument(
        '--matrix',
        type=str,
        default=None,
        help='Путь к файлу матрицы "термин-документ" (term_document_matrix.json)'
    )
    parser.add_argument(
        '--target-dim',
        type=int,
        default=100,
        help='Целевая размерность после PCA (по умолчанию: 100)'
    )
    parser.add_argument(
        '--max-docs',
        type=int,
        default=1000,
        help='Максимальное количество документов для обучения PCA (по умолчанию: 1000)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Директория для сохранения моделей PCA'
    )
    
    args = parser.parse_args()
    
    # Пути
    base_dir = Path(__file__).parent
    
    if args.train_dir:
        train_dir = Path(args.train_dir)
        if not train_dir.is_absolute():
            train_dir = base_dir / train_dir
    else:
        train_dir = base_dir / "assets" / "annotated-corpus" / "train"
    
    if args.vocab:
        vocab_path = Path(args.vocab)
        if not vocab_path.is_absolute():
            vocab_path = base_dir / vocab_path
    else:
        vocab_path = base_dir / "output" / "vocabulary.json"
    
    if args.matrix:
        matrix_path = Path(args.matrix)
        if not matrix_path.is_absolute():
            matrix_path = base_dir / matrix_path
    else:
        matrix_path = base_dir / "output" / "term_document_matrix.json"
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
        if not output_dir.is_absolute():
            output_dir = base_dir / output_dir
    else:
        output_dir = base_dir / "output"
    
    apply_pca_to_basic_vectors(
        train_dir=train_dir,
        vocab_path=vocab_path,
        matrix_path=matrix_path,
        target_dim=args.target_dim,
        max_docs=args.max_docs,
        output_dir=output_dir
    )

