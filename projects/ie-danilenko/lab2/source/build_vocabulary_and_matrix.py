"""
Скрипт для построения словаря токенов и матрицы "термин-документ"
на основе аннотации в формате TSV из первой лабораторной работы.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from text_processing import STOP_WORDS, is_punctuation


def read_annotation_file(file_path):
    """
    Читает файл аннотации и извлекает токены.
    
    Args:
        file_path: Путь к файлу аннотации в формате TSV
        
    Returns:
        Список токенов из документа
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
                    # Пропускаем пустые токены, пунктуацию и стоп-слова
                    if token and not is_punctuation(token) and token not in STOP_WORDS:
                        tokens.append(token)
    except Exception as e:
        print(f"Ошибка при чтении файла {file_path}: {e}")
    return tokens


def collect_all_tokens(train_dir):
    """
    Собирает все токены из всех файлов аннотации в обучающей выборке.
    
    Args:
        train_dir: Путь к директории с обучающей выборкой
        
    Returns:
        tuple: (словарь токенов с частотами, словарь токенов по документам)
    """
    all_tokens_counter = Counter()
    doc_tokens = {}  # doc_id -> список токенов
    
    train_path = Path(train_dir)
    if not train_path.exists():
        raise FileNotFoundError(f"Директория {train_dir} не найдена")
    
    # Проходим по всем поддиректориям (1, 2, 3, 4)
    for class_dir in sorted(train_path.iterdir()):
        if not class_dir.is_dir():
            continue
            
        print(f"Обработка директории {class_dir.name}...")
        
        # Проходим по всем TSV файлам в директории класса
        for tsv_file in sorted(class_dir.glob("*.tsv")):
            doc_id = tsv_file.stem  # Имя файла без расширения
            
            tokens = read_annotation_file(tsv_file)
            if tokens:
                all_tokens_counter.update(tokens)
                doc_tokens[doc_id] = tokens
    
    print(f"Обработано {len(doc_tokens)} документов")
    print(f"Найдено {len(all_tokens_counter)} уникальных токенов")
    
    return all_tokens_counter, doc_tokens


def filter_low_frequency_tokens(token_freq, min_frequency=2):
    """
    Фильтрует низкочастотные токены.
    
    Args:
        token_freq: Словарь токен -> частота
        min_frequency: Минимальная частота для сохранения токена
        
    Returns:
        Отфильтрованный словарь токенов
    """
    filtered = {token: freq for token, freq in token_freq.items() 
                if freq >= min_frequency}
    print(f"После фильтрации (min_freq={min_frequency}): {len(filtered)} токенов")
    return filtered


def build_term_document_matrix(doc_tokens, vocabulary):
    """
    Строит матрицу "термин-документ" (term-document matrix).
    
    В матрице:
    - Строки соответствуют токенам (терминам)
    - Столбцы соответствуют документам
    
    Args:
        doc_tokens: Словарь doc_id -> список токенов
        vocabulary: Словарь токен -> индекс в словаре
        
    Returns:
        tuple: (список doc_ids, словарь term_index -> doc_index -> frequency)
    """
    # Создаем mapping doc_id -> индекс
    doc_ids = sorted(doc_tokens.keys())
    doc_to_idx = {doc_id: idx for idx, doc_id in enumerate(doc_ids)}
    
    # Строим разреженную матрицу в формате COO (Coordinate)
    # Храним как: term_index -> {doc_index: frequency}
    matrix = defaultdict(lambda: defaultdict(int))
    
    for doc_id, tokens in doc_tokens.items():
        doc_idx = doc_to_idx[doc_id]
        token_counter = Counter(tokens)
        
        for token, count in token_counter.items():
            if token in vocabulary:
                term_idx = vocabulary[token]
                matrix[term_idx][doc_idx] = count
    
    return doc_ids, dict(matrix)


def save_vocabulary(vocabulary, output_path):
    """
    Сохраняет словарь токенов в JSON файл.
    
    Args:
        vocabulary: Словарь токен -> частота
        output_path: Путь для сохранения
    """
    # Сортируем по частоте (по убыванию)
    sorted_vocab = sorted(vocabulary.items(), key=lambda x: x[1], reverse=True)
    
    # Создаем словарь с индексами для эффективного доступа
    vocab_with_indices = {
        'tokens': [token for token, _ in sorted_vocab],
        'frequencies': [freq for _, freq in sorted_vocab],
        'token_to_index': {token: idx for idx, (token, _) in enumerate(sorted_vocab)},
        'total_tokens': sum(vocabulary.values()),
        'unique_tokens': len(vocabulary)
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(vocab_with_indices, f, ensure_ascii=False, indent=2)
    
    print(f"Словарь сохранен в {output_path}")
    print(f"  Всего уникальных токенов: {vocab_with_indices['unique_tokens']}")
    print(f"  Всего токенов в корпусе: {vocab_with_indices['total_tokens']}")


def save_matrix_sparse(doc_ids, matrix, vocabulary, output_path):
    """
    Сохраняет разреженную матрицу в компактном формате COO (Coordinate).
    
    Использует эффективный формат для разреженной матрицы:
    - Сохраняются только ненулевые значения
    - Формат: три массива (rows, cols, values) для координат и значений
    
    Args:
        doc_ids: Список идентификаторов документов
        matrix: Словарь term_index -> {doc_index: frequency}
        vocabulary: Словарь токен -> индекс (для обратного поиска)
        output_path: Путь для сохранения
    """
    # Создаем компактную версию для эффективного хранения
    compact_data = {
        'doc_ids': doc_ids,
        'num_docs': len(doc_ids),
        'num_terms': len(vocabulary),
        'rows': [],  # term_indices
        'cols': [],  # doc_indices
        'values': []  # frequencies
    }
    
    for term_idx in sorted(matrix.keys()):
        for doc_idx, frequency in sorted(matrix[term_idx].items()):
            compact_data['rows'].append(term_idx)
            compact_data['cols'].append(doc_idx)
            compact_data['values'].append(frequency)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(compact_data, f, ensure_ascii=False)
    
    num_nonzero = len(compact_data['values'])
    total_elements = len(doc_ids) * len(vocabulary)
    sparsity = (1 - num_nonzero / total_elements) * 100 if total_elements > 0 else 0
    
    print(f"Матрица сохранена в {output_path}")
    print(f"  Размер матрицы: {len(vocabulary)} x {len(doc_ids)}")
    print(f"  Ненулевых элементов: {num_nonzero}")
    print(f"  Разреженность: {sparsity:.2f}%")

def parse_arguments():
    """Парсит аргументы командной строки."""
    parser = argparse.ArgumentParser(
        description="Построение словаря токенов и матрицы 'термин-документ'"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Путь к директории с обучающей выборкой (TSV файлы)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Путь к директории для сохранения результатов"
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Минимальная частота токена для включения в словарь (по умолчанию: 2)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    
    # Пути
    train_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    # Проверяем существование входной директории
    if not train_dir.exists():
        raise FileNotFoundError(f"Входная директория {train_dir} не найдена")
    
    # Создаем выходную директорию, если её нет
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("Построение словаря токенов и матрицы 'термин-документ'")
    print("=" * 60)
    print(f"Входная директория: {train_dir}")
    print(f"Выходная директория: {output_dir}")
    print(f"Минимальная частота: {args.min_frequency}")
    
    # 1. Собираем все токены из обучающей выборки
    print("\n1. Сбор токенов из обучающей выборки...")
    token_freq, doc_tokens = collect_all_tokens(train_dir)
    
    # 2. Фильтруем низкочастотные токены
    print("\n2. Фильтрация низкочастотных токенов...")
    filtered_vocab = filter_low_frequency_tokens(dict(token_freq), args.min_frequency)
    
    # 3. Создаем словарь с индексами
    sorted_tokens = sorted(filtered_vocab.keys())
    vocabulary = {token: idx for idx, token in enumerate(sorted_tokens)}
    vocabulary_freq = filtered_vocab
    
    # 4. Строим матрицу "термин-документ"
    print("\n3. Построение матрицы 'термин-документ'...")
    doc_ids, matrix = build_term_document_matrix(doc_tokens, vocabulary)
    
    # 5. Сохраняем результаты
    print("\n4. Сохранение результатов...")
    vocab_path = output_dir / f"vocabulary.json"
    save_vocabulary(vocabulary_freq, vocab_path)
    
    matrix_path = output_dir / f"term_document_matrix.json"
    save_matrix_sparse(doc_ids, matrix, vocabulary, matrix_path)
    
    print("\n" + "=" * 60)
    print("Готово!")
    print("=" * 60)

