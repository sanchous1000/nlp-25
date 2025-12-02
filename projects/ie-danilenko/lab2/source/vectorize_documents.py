"""
Скрипт для векторизации документов с использованием модели GloVe.
Реализует алгоритм:
1. Сегментация текста на предложения и токены
2. Формирование векторных представлений каждого токена с помощью GloVe
3. Подсчет среднего/взвешенного среднего векторных представлений токенов каждого предложения
4. Подсчет векторного представления документа из векторов предложений
"""

import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm

from text_to_glove import (
    initialize_glove_model,
    get_device,
    read_annotation_file
)
from text_to_tfidf import (
    load_vocabulary,
    load_term_document_matrix,
    compute_tf,
    compute_idf
)
from demonstrate_glove_similarity import get_word_vector


def token_to_glove_vector(
    token,
    glove_model
):
    """
    Получает векторное представление токена из модели GloVe.
    
    Args:
        token: Токен
        glove_model: Модель GloVe
        
    Returns:
        Вектор токена или None, если токен не найден
    """
    return get_word_vector(token, glove_model)


def sentence_to_vector(
    tokens,
    glove_model,
    use_tfidf_weights=False,
    vocabulary=None,
    num_docs=None,
    term_doc_counts=None
):
    """
    Преобразует предложение (список токенов) в векторное представление.
    
    Args:
        tokens: Список токенов предложения
        glove_model: Модель GloVe
        use_tfidf_weights: Использовать ли взвешенное среднее с TF-IDF весами
        vocabulary: Словарь для TF-IDF (необходим если use_tfidf_weights=True)
        num_docs: Количество документов для TF-IDF
        term_doc_counts: Словарь term_index -> количество документов для TF-IDF
        
    Returns:
        Векторное представление предложения
    """
    if not tokens:
        # Возвращаем нулевой вектор, если предложение пустое
        embeddings = glove_model.get_embeddings().cpu().numpy()
        return np.zeros(glove_model.embedding_dim)
    
    # Получаем векторы для каждого токена
    token_vectors = []
    token_weights = []
    
    for token in tokens:
        token_vector = token_to_glove_vector(token, glove_model)
        if token_vector is not None:
            token_vectors.append(token_vector)
            
            # Вычисляем вес токена
            if use_tfidf_weights and vocabulary is not None and num_docs is not None and term_doc_counts is not None:
                # Используем TF-IDF как вес
                if token in vocabulary:
                    term_index = vocabulary[token]
                    # TF для токена в предложении
                    token_count = tokens.count(token)
                    tf = compute_tf(token_count, len(tokens))
                    # IDF для токена
                    idf = compute_idf(num_docs, term_doc_counts, term_index)
                    weight = tf * idf
                else:
                    weight = 1.0
            else:
                weight = 1.0
            
            token_weights.append(weight)
    
    if not token_vectors:
        # Если ни один токен не найден, возвращаем нулевой вектор
        embeddings = glove_model.get_embeddings().cpu().numpy()
        return np.zeros(glove_model.embedding_dim)
    
    # Преобразуем в numpy массивы
    token_vectors = np.array(token_vectors)
    token_weights = np.array(token_weights)
    
    # Нормализуем веса
    if use_tfidf_weights:
        if np.sum(token_weights) > 0:
            token_weights = token_weights / np.sum(token_weights)
        else:
            token_weights = np.ones(len(token_weights)) / len(token_weights)
    
    # Вычисляем взвешенное среднее
    sentence_vector = np.average(token_vectors, axis=0, weights=token_weights)
    
    return sentence_vector


def document_to_vector(
    sentences,
    glove_model,
    sentence_aggregation='mean',
    use_tfidf_weights=False,
    vocabulary=None,
    num_docs=None,
    term_doc_counts=None
):
    """
    Преобразует документ (список предложений) в векторное представление.
    
    Args:
        sentences: Список предложений, где каждое предложение - список токенов
        glove_model: Модель GloVe
        sentence_aggregation: Способ агрегации векторов предложений ('mean', 'sum', 'max')
        use_tfidf_weights: Использовать ли взвешенное среднее для токенов с TF-IDF весами
        vocabulary: Словарь для TF-IDF (необходим если use_tfidf_weights=True)
        num_docs: Количество документов для TF-IDF
        term_doc_counts: Словарь term_index -> количество документов для TF-IDF
        
    Returns:
        Векторное представление документа
    """
    if not sentences:
        embeddings = glove_model.get_embeddings().cpu().numpy()
        return np.zeros(glove_model.embedding_dim)
    
    # Преобразуем каждое предложение в вектор
    sentence_vectors = []
    for sentence_tokens in sentences:
        sentence_vec = sentence_to_vector(
            sentence_tokens,
            glove_model,
            use_tfidf_weights=use_tfidf_weights,
            vocabulary=vocabulary,
            num_docs=num_docs,
            term_doc_counts=term_doc_counts
        )
        sentence_vectors.append(sentence_vec)
    
    sentence_vectors = np.array(sentence_vectors)
    
    # Агрегируем векторы предложений
    if sentence_aggregation == 'mean':
        return np.mean(sentence_vectors, axis=0)
    elif sentence_aggregation == 'sum':
        return np.sum(sentence_vectors, axis=0)
    elif sentence_aggregation == 'max':
        return np.max(sentence_vectors, axis=0)
    else:
        raise ValueError(f"Неизвестный метод агрегации: {sentence_aggregation}")


def vectorize_test_set(
    test_dir,
    glove_model,
    output_path,
    use_tfidf_weights=False,
    vocabulary=None,
    num_docs=None,
    term_doc_counts=None,
    sentence_aggregation='mean',
    verbose=True
):
    """
    Векторизует тестовую выборку и сохраняет результаты в TSV файл.
    
    Args:
        test_dir: Директория с тестовой выборкой (TSV файлы аннотаций)
        glove_model: Модель GloVe
        output_path: Путь для сохранения результатов (TSV файл)
        use_tfidf_weights: Использовать ли взвешенное среднее с TF-IDF весами
        vocabulary: Словарь для TF-IDF
        num_docs: Количество документов для TF-IDF
        term_doc_counts: Словарь term_index -> количество документов для TF-IDF
        sentence_aggregation: Способ агрегации векторов предложений
        verbose: Выводить ли информацию о прогрессе
    """
    if verbose:
        print("=" * 80)
        print("Векторизация тестовой выборки")
        print("=" * 80)
    
    # Собираем все TSV файлы из тестовой выборки
    tsv_files = []
    for class_dir in sorted(test_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for tsv_file in sorted(class_dir.glob("*.tsv")):
            tsv_files.append(tsv_file)
    
    if verbose:
        print(f"\nНайдено {len(tsv_files)} документов в тестовой выборке")
    
    # Векторизуем документы
    document_vectors = {}
    
    if verbose:
        pbar = tqdm(tsv_files, desc="Векторизация документов", unit="документ")
    else:
        pbar = tsv_files
    
    for tsv_file in pbar:
        doc_id = tsv_file.stem  # Имя файла без расширения
        
        # Читаем предложения из TSV файла
        sentences = read_annotation_file(tsv_file)
        
        # Преобразуем документ в вектор
        doc_vector = document_to_vector(
            sentences,
            glove_model,
            sentence_aggregation=sentence_aggregation,
            use_tfidf_weights=use_tfidf_weights,
            vocabulary=vocabulary,
            num_docs=num_docs,
            term_doc_counts=term_doc_counts
        )
        
        document_vectors[doc_id] = doc_vector
    
    # Сохраняем результаты в TSV файл
    if verbose:
        print(f"\nСохранение результатов в {output_path}...")
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for doc_id in sorted(document_vectors.keys()):
            vector = document_vectors[doc_id]
            # Формируем строку: doc_id \t компонент1 \t компонент2 \t ...
            vector_str = '\t'.join([f"{comp:.10f}" for comp in vector])
            f.write(f"{doc_id}\t{vector_str}\n")
    
    if verbose:
        print(f"✅ Сохранено {len(document_vectors)} векторов")
        print(f"   Размерность векторов: {len(document_vectors[list(document_vectors.keys())[0]])}")
        print(f"   Файл: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Векторизация документов с использованием модели GloVe'
    )
    parser.add_argument(
        '--glove-model',
        type=str,
        default=None,
        help='Путь к модели GloVe (по умолчанию: output/glove_model.pkl)'
    )
    parser.add_argument(
        '--test-dir',
        type=str,
        default=None,
        help='Директория с тестовой выборкой (по умолчанию: assets/annotated-corpus/test)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Путь к выходному TSV файлу (по умолчанию: output/test_vectors.tsv)'
    )
    parser.add_argument(
        '--vocab',
        type=str,
        default=None,
        help='Путь к словарю для TF-IDF (по умолчанию: output/vocabulary.json)'
    )
    parser.add_argument(
        '--matrix',
        type=str,
        default=None,
        help='Путь к матрице "термин-документ" для TF-IDF (по умолчанию: output/term_document_matrix.json)'
    )
    parser.add_argument(
        '--use-tfidf-weights',
        action='store_true',
        help='Использовать взвешенное среднее с TF-IDF весами для токенов'
    )
    parser.add_argument(
        '--sentence-aggregation',
        type=str,
        default='mean',
        choices=['mean', 'sum', 'max'],
        help='Способ агрегации векторов предложений (по умолчанию: mean)'
    )
    
    args = parser.parse_args()
    
    # Пути
    base_dir = Path(__file__).parent
    
    if args.glove_model:
        glove_model_path = Path(args.glove_model)
        if not glove_model_path.is_absolute():
            glove_model_path = base_dir / glove_model_path
    else:
        glove_model_path = base_dir / "output" / "glove_model.pkl"
    
    if args.test_dir:
        test_dir = Path(args.test_dir)
        if not test_dir.is_absolute():
            test_dir = base_dir / test_dir
    else:
        test_dir = base_dir / "assets" / "annotated-corpus" / "test"
    
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = base_dir / output_path
    else:
        output_path = base_dir / "output" / "test_vectors.tsv"
    
    # Загружаем модель GloVe
    print("1. Загрузка модели GloVe...")
    device = get_device()
    glove_model, word_to_id = initialize_glove_model(
        model_path=glove_model_path,
        device=device,
        retrain=False
    )
    print(f"   Размер словаря: {len(word_to_id)}")
    print(f"   Размерность векторов: {glove_model.embedding_dim}")
    
    # Загружаем словарь и матрицу для TF-IDF (если нужно)
    vocabulary = None
    num_docs = None
    term_doc_counts = None
    
    if args.use_tfidf_weights:
        print("\n2. Загрузка словаря и матрицы для TF-IDF весов...")
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
        
        vocabulary = load_vocabulary(vocab_path)
        num_docs, term_doc_counts = load_term_document_matrix(matrix_path)
        print(f"   Размер словаря: {len(vocabulary)}")
        print(f"   Количество документов: {num_docs}")
    
    # Векторизуем тестовую выборку
    print(f"\n3. Векторизация тестовой выборки...")
    print(f"   Директория: {test_dir}")
    print(f"   Использование TF-IDF весов: {'Да' if args.use_tfidf_weights else 'Нет'}")
    print(f"   Агрегация предложений: {args.sentence_aggregation}")
    
    vectorize_test_set(
        test_dir=test_dir,
        glove_model=glove_model,
        output_path=output_path,
        use_tfidf_weights=args.use_tfidf_weights,
        vocabulary=vocabulary,
        num_docs=num_docs,
        term_doc_counts=term_doc_counts,
        sentence_aggregation=args.sentence_aggregation,
        verbose=True
    )
    
    print("\n" + "=" * 80)
    print("Готово!")
    print("=" * 80)

