"""
Скрипт для проведения экспериментов с LDA тематическим моделированием
"""

import json
import numpy as np
import pandas as pd
import time
from collections import Counter
from sklearn.decomposition import LatentDirichletAllocation
from scipy.sparse import coo_matrix, csr_matrix
from pathlib import Path
from tabulate import tabulate


def load_term_document_matrix(filepath):
    """Загружает матрицу термин-документ из JSON формата (sparse COO)"""
    print(f"Загрузка матрицы термин-документ из {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    num_docs = data['num_docs']
    num_terms = data['num_terms']
    # В формате из lab2: rows = term_indices, cols = doc_indices
    term_indices = data['rows']  # индексы терминов
    doc_indices = data['cols']   # индексы документов
    
    # Получаем значения (может быть 'values' или 'data')
    if 'values' in data:
        values = data['values']
    elif 'data' in data:
        values = data['data']
    else:
        # Если значений нет, предполагаем, что все значения равны 1
        values = [1] * len(term_indices)
    
    # Создаем sparse matrix (документы x термины)
    # Оставляем в sparse формате для экономии памяти
    sparse_matrix = coo_matrix((values, (doc_indices, term_indices)), shape=(num_docs, num_terms))
    # Конвертируем в CSR формат для более эффективной работы с LDA
    matrix = sparse_matrix.tocsr().astype(np.int32)
    
    print(f"Загружена sparse матрица размером {matrix.shape}")
    print(f"  Ненулевых элементов: {matrix.nnz:,}")
    return matrix


def load_vocabulary(filepath):
    """Загружает словарь из JSON файла"""
    print(f"Загрузка словаря из {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    # Возвращаем tokens и token_to_index для удобства
    tokens = vocab_data['tokens']
    token_to_index = vocab_data.get('token_to_index', {})
    
    # Если token_to_index не в словаре, создаем его
    if not token_to_index:
        token_to_index = {token.lower(): idx for idx, token in enumerate(tokens)}
    
    return tokens, token_to_index


def read_annotation_file(file_path, vocabulary_dict):
    """
    Читает файл аннотации и извлекает токены.
    
    Args:
        file_path: Путь к файлу аннотации в формате TSV
        vocabulary_dict: Словарь токен -> индекс для проверки наличия токена
        
    Returns:
        Список токенов из документа (только те, что есть в словаре)
    """
    tokens = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # Пустая строка между предложениями
                    continue
                parts = line.split('\t')
                if len(parts) >= 1:
                    token = parts[0].strip().lower()
                    # Проверяем, есть ли токен в словаре (словарь уже отфильтрован)
                    if token and token in vocabulary_dict:
                        tokens.append(token)
    except Exception as e:
        print(f"Ошибка при чтении файла {file_path}: {e}")
    return tokens


def build_test_matrix(test_annotated_dir, vocabulary, token_to_index):
    """
    Строит матрицу термин-документ для тестовой выборки.
    """
    print("Построение матрицы для тестовой выборки...")
    
    vocab_dict = {}
    for token, idx in token_to_index.items():
        vocab_dict[token.lower()] = idx
    
    # Собираем токены из всех тестовых документов
    test_documents = []
    doc_ids = []
    
    test_path = Path(test_annotated_dir)
    if not test_path.exists():
        raise FileNotFoundError(f"Директория {test_annotated_dir} не найдена")
    
    # Проходим по всем поддиректориям (1, 2, 3, 4) - как в lab2
    class_dirs = sorted([d for d in test_path.iterdir() if d.is_dir()])
    
    for class_dir in class_dirs:
        print(f"Обработка директории {class_dir.name}...")
        tsv_files = sorted(class_dir.glob("*.tsv"))
        
        for tsv_file in tsv_files:
            doc_id = tsv_file.stem  # Имя файла без расширения
            
            tokens = read_annotation_file(tsv_file, vocab_dict)
            if tokens:
                test_documents.append(tokens)
                doc_ids.append(doc_id)
    
    if not test_documents:
        raise ValueError("Не найдено тестовых документов с токенами!")
    
    print(f"Обработано {len(test_documents)} тестовых документов")
    
    vocab_size = len(vocabulary)
    
    rows = []
    cols = []
    data = []
    
    for doc_idx, tokens in enumerate(test_documents):
        token_counter = Counter(tokens)
        for token, count in token_counter.items():
            if token in vocab_dict:
                rows.append(doc_idx)
                cols.append(vocab_dict[token])
                data.append(count)
    
    test_matrix = csr_matrix((data, (rows, cols)), shape=(len(test_documents), vocab_size), dtype=np.int32)
    
    print(f"Построена sparse матрица размером {test_matrix.shape}")
    print(f"  Ненулевых элементов: {test_matrix.nnz:,}")
    return test_matrix


def get_top_words(model, vocabulary, n_words=10):
    """Получает топ-N ключевых слов для каждой темы"""
    feature_names = vocabulary
    topics = []
    
    for _, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-n_words:][::-1]
        top_words = [(feature_names[i], topic[i]) for i in top_indices]
        topics.append(top_words)
    
    return topics


def run_lda_experiment(n_topics, train_matrix, test_matrix, vocabulary, output_dir, max_iter=20):
    """
    Проводит эксперимент с LDA для заданного количества тем и итераций
    
    Args:
        n_topics: Количество тем
        train_matrix: Матрица термин-документ для обучающей выборки
        test_matrix: Матрица термин-документ для тестовой выборки
        vocabulary: Список токенов словаря
        output_dir: Директория для сохранения результатов
        max_iter: Количество итераций обучения (по умолчанию 20)
    """
    print(f"\n{'='*60}")
    print(f"Эксперимент с {n_topics} темами, {max_iter} итераций")
    print(f"{'='*60}")
    
    print("Обучение модели LDA...")
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=max_iter,
        learning_method='batch',
        n_jobs=1
    )
    
    # Измеряем время обучения
    start_time = time.time()
    lda.fit(train_matrix)
    training_time = time.time() - start_time
    
    # Получаем топ-10 ключевых слов для каждой темы
    print("Извлечение топ-10 ключевых слов...")
    top_words = get_top_words(lda, vocabulary, n_words=10)
    
    # Вычисляем perplexity на тестовой выборке
    print("Вычисление perplexity на тестовой выборке...")
    perplexity = lda.perplexity(test_matrix)
    
    # Получаем вероятности принадлежности документов обучающей выборки к темам
    print("Вычисление вероятностей принадлежности документов...")
    doc_topic_probs = lda.transform(train_matrix)
    
    # Находим документы с наибольшей вероятностью для каждой темы
    top_docs_per_topic = []
    for topic_idx in range(n_topics):
        topic_probs = doc_topic_probs[:, topic_idx]
        top_indices = topic_probs.argsort()[-10:][::-1]
        top_docs_per_topic.append([(int(idx), float(topic_probs[idx])) for idx in top_indices])
    
    # Сохраняем результаты
    results = {
        'n_topics': n_topics,
        'max_iter': max_iter,
        'perplexity': float(perplexity),
        'training_time_seconds': float(training_time),
        'top_words': {f'topic_{i}': [(word, float(score)) for word, score in words] 
                      for i, words in enumerate(top_words)},
        'top_documents_per_topic': {f'topic_{i}': docs 
                                    for i, docs in enumerate(top_docs_per_topic)}
    }
    
    # Выводим результаты
    print(f"\nPerplexity: {perplexity:.4f}")
    print(f"Время обучения: {training_time:.2f} секунд ({training_time/60:.2f} минут)")
    print("\nТоп-10 ключевых слов для каждой темы:")
    for topic_idx, words in enumerate(top_words):
        print(f"\nТема {topic_idx}:")
        for word, score in words:
            print(f"  {word}: {score:.4f}")
    
    return results


if __name__ == '__main__':
    base_dir = Path(__file__).parent.parent
    source_dir = base_dir / 'source'
    output_dir = source_dir / 'output'
    
    term_doc_matrix_file = output_dir / 'term_document_matrix.json'
    vocabulary_file = output_dir / 'vocabulary.json'
    train_data_file = source_dir / 'dataset' / 'train.csv'
    
    train_matrix = load_term_document_matrix(term_doc_matrix_file)
    vocabulary, token_to_index = load_vocabulary(vocabulary_file)
    
    print(f"Размер обучающей матрицы: {train_matrix.shape}")
    print(f"Размер словаря: {len(vocabulary)}")
    
    test_annotated_dir = source_dir / 'assets' / 'annotated-corpus' / 'test'
    test_matrix = build_test_matrix(test_annotated_dir, vocabulary, token_to_index)
    
    train_df = pd.read_csv(train_data_file, header=None)
    n_classes = train_df[0].nunique()
    print(f"Количество классов в датасете: {n_classes}")
    
    n_topics_list = [2, 5, 10, 20, 40, n_classes]
    max_iter_list = [10, 20, 40]

    all_results = {}
    perplexities_by_iter = {max_iter: [] for max_iter in max_iter_list}
    
    print(f"\n{'='*60}")
    print("Проведение экспериментов с разным количеством тем и итераций")
    print(f"{'='*60}")
    
    for max_iter in max_iter_list:
        print(f"\n{'#'*60}")
        print(f"Эксперименты с {max_iter} итерациями")
        print(f"{'#'*60}")
        
        for n_topics in n_topics_list:
            result_key = f"{n_topics}topics_{max_iter}iter"
            
            results = run_lda_experiment(
                n_topics, 
                train_matrix, 
                test_matrix, 
                vocabulary,
                output_dir,
                max_iter=max_iter
            )
            all_results[result_key] = results
            perplexities_by_iter[max_iter].append((n_topics, results['perplexity']))
    
    results_file = output_dir / 'lda_experiments_results.json'
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n{'='*60}")
    print("Все результаты сохранены в", results_file)
    print(f"{'='*60}")
    
    print("\n" + "="*60)
    print("Сводка по perplexity для разных количеств итераций:")
    print("="*60)
    
    for max_iter in max_iter_list:
        print(f"\n{max_iter} итераций:")
        print("-" * 40)
        for n_topics, perp in perplexities_by_iter[max_iter]:
            print(f"  {n_topics} тем: {perp:.4f}")
    
    print("\n" + "="*60)
    print("Сравнительная таблица: Perplexity для разных комбинаций тем и итераций")
    print("="*60)
    
    table_data_perp = []
    headers_perp = ["Количество тем"] + [f"{max_iter} итераций" for max_iter in max_iter_list]
    
    for n_topics in n_topics_list:
        row = [f"{n_topics}"]
        for max_iter in max_iter_list:
            result_key = f"{n_topics}topics_{max_iter}iter"
            if result_key in all_results:
                perp = all_results[result_key]['perplexity']
                row.append(f"{perp:.4f}")
            else:
                row.append("N/A")
        table_data_perp.append(row)
    
    print("\n" + tabulate(table_data_perp, headers=headers_perp, tablefmt="grid", stralign="center"))
    
    print("\n" + "="*60)
    print("Сравнительная таблица: Время обучения для разных комбинаций тем и итераций")
    print("="*60)
    
    table_data_time = []
    headers_time = ["Количество тем"] + [f"{max_iter} итераций (сек)" for max_iter in max_iter_list]
    
    for n_topics in n_topics_list:
        row = [f"{n_topics}"]
        for max_iter in max_iter_list:
            result_key = f"{n_topics}topics_{max_iter}iter"
            if result_key in all_results:
                train_time = all_results[result_key]['training_time_seconds']
                row.append(f"{train_time:.2f}")
            else:
                row.append("N/A")
        table_data_time.append(row)
    
    print("\n" + tabulate(table_data_time, headers=headers_time, tablefmt="grid", stralign="center"))
    
    print("\n" + "="*60)
    print("Анализ оптимального количества итераций:")
    print("="*60)
    
    best_iter_by_topics = {}
    for n_topics in n_topics_list:
        best_perp = float('inf')
        best_iter = None
        for max_iter in max_iter_list:
            result_key = f"{n_topics}topics_{max_iter}iter"
            if result_key in all_results:
                perp = all_results[result_key]['perplexity']
                if perp < best_perp:
                    best_perp = perp
                    best_iter = max_iter
        if best_iter is not None:
            best_iter_by_topics[n_topics] = (best_iter, best_perp)
            print(f"  {n_topics} тем: оптимально {best_iter} итераций (perplexity = {best_perp:.4f})")
    
    iter_counts = {}
    for n_topics, (best_iter, _) in best_iter_by_topics.items():
        iter_counts[best_iter] = iter_counts.get(best_iter, 0) + 1
    
    most_common_iter = max(iter_counts.items(), key=lambda x: x[1])
    print(f"\nВывод: оптимальное количество итераций - {most_common_iter[0]} "
          f"(оптимально для {most_common_iter[1]} из {len(n_topics_list)} экспериментов с разным количеством тем)")