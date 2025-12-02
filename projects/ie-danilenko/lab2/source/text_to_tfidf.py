"""
Скрипт для преобразования произвольного текста в вектор значений tf-idf.
Использует словарь токенов и матрицу "термин-документ", полученные ранее.
"""

import json
import math
from collections import Counter
from pathlib import Path

from text_processing import STOP_WORDS, is_punctuation


def tokenize_text(text):
    """
    Токенизирует текст с использованием тех же правил, что и при построении словаря.
    
    Args:
        text: Входной текст для токенизации
        
    Returns:
        Список токенов
    """
    # Преобразуем текст в нижний регистр и разбиваем на слова
    # Простая токенизация по пробелам и пунктуации
    tokens = []
    
    # Убираем пунктуацию и разбиваем на слова
    text = text.lower()
    words = text.split()
    
    for word in words:
        # Убираем пунктуацию в начале и конце слова
        word = word.strip('.,!?;:"()[]{}\'\"-/—–')
        if word and not is_punctuation(word) and word not in STOP_WORDS:
            tokens.append(word)
    
    return tokens


def load_vocabulary(vocab_path):
    """
    Загружает словарь токенов из JSON файла.
    
    Args:
        vocab_path: Путь к файлу vocabulary.json
        
    Returns:
        Словарь токен -> индекс
    """
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_data = json.load(f)
    
    return vocab_data['token_to_index']


def load_term_document_matrix(matrix_path):
    """
    Загружает матрицу "термин-документ" из JSON файла.
    
    Args:
        matrix_path: Путь к файлу term_document_matrix.json
        
    Returns:
        tuple: (num_docs, term_doc_counts)
            num_docs: количество документов в корпусе
            term_doc_counts: словарь term_index -> количество документов, содержащих этот термин
    """
    with open(matrix_path, 'r', encoding='utf-8') as f:
        matrix_data = json.load(f)
    
    num_docs = matrix_data['num_docs']
    
    # Подсчитываем, сколько документов содержит каждый термин
    # Используем множества для учета уникальных документов для каждого термина
    term_doc_sets = {}
    rows = matrix_data['rows']
    cols = matrix_data['cols']
    
    for i in range(len(rows)):
        term_idx = rows[i]
        doc_idx = cols[i]
        
        if term_idx not in term_doc_sets:
            term_doc_sets[term_idx] = set()
        term_doc_sets[term_idx].add(doc_idx)
    
    # Преобразуем множества в количества
    term_doc_counts = {term_idx: len(doc_set) for term_idx, doc_set in term_doc_sets.items()}
    
    return num_docs, term_doc_counts


def compute_idf(num_docs, term_doc_counts, term_index):
    """
    Вычисляет IDF (Inverse Document Frequency) для термина.
    
    Формула: idf(t, D) = log(N / df(t))
    где N - количество документов в корпусе,
    df(t) - количество документов, содержащих термин t.
    
    Args:
        num_docs: Общее количество документов в корпусе
        term_doc_counts: Словарь term_index -> количество документов, содержащих термин
        term_index: Индекс термина в словаре
        
    Returns:
        Значение IDF для термина
    """
    # Если термин не встречается ни в одном документе, df = 1 (для избежания деления на 0)
    doc_freq = term_doc_counts.get(term_index, 0)
    if doc_freq == 0:
        doc_freq = 1  # Защита от деления на ноль
    
    # Стандартная формула IDF с натуральным логарифмом
    idf = math.log(num_docs / doc_freq)
    return idf


def compute_tf(token_count, total_tokens):
    """
    Вычисляет TF (Term Frequency) для термина в документе.
    
    Формула: tf(t, d) = count(t, d) / |d|
    где count(t, d) - количество вхождений термина t в документ d,
    |d| - общее количество токенов в документе d.
    
    Args:
        token_count: Количество вхождений термина в документ
        total_tokens: Общее количество токенов в документе
        
    Returns:
        Значение TF для термина
    """
    if total_tokens == 0:
        return 0.0
    return token_count / total_tokens


def initialize_tfidf_model(
    vocab_path=None,
    matrix_path=None
):
    """
    Инициализирует модель TF-IDF, загружая словарь и матрицу.
    Удобная функция для загрузки данных один раз при многократном использовании.
    
    Args:
        vocab_path: Путь к файлу vocabulary.json (если None, используется путь по умолчанию)
        matrix_path: Путь к файлу term_document_matrix.json (если None, используется путь по умолчанию)
        
    Returns:
        tuple: (vocabulary, num_docs, term_doc_counts)
            vocabulary: Словарь токен -> индекс
            num_docs: Количество документов в корпусе
            term_doc_counts: Словарь term_index -> количество документов, содержащих термин
    """
    if vocab_path is None:
        base_dir = Path(__file__).parent
        vocab_path = base_dir / "output" / "vocabulary.json"
    
    if matrix_path is None:
        base_dir = Path(__file__).parent
        matrix_path = base_dir / "output" / "term_document_matrix.json"
    
    vocabulary = load_vocabulary(vocab_path)
    num_docs, term_doc_counts = load_term_document_matrix(matrix_path)
    
    return vocabulary, num_docs, term_doc_counts


def text_to_tfidf_vector(
    text,
    vocabulary,
    num_docs,
    term_doc_counts
):
    """
    Преобразует произвольный текст в вектор значений tf-idf.
    
    Args:
        text: Входной текст для векторизации
        vocabulary: Словарь токен -> индекс
        num_docs: Количество документов в обучающем корпусе
        term_doc_counts: Словарь term_index -> количество документов, содержащих термин
        
    Returns:
        Вектор tf-idf значений размером равным размеру словаря
    """
    # Токенизируем текст
    tokens = tokenize_text(text)
    
    if not tokens:
        # Если текст пустой, возвращаем нулевой вектор
        return [0.0] * len(vocabulary)
    
    # Подсчитываем частоты токенов в тексте
    token_counter = Counter(tokens)
    total_tokens = len(tokens)
    
    # Инициализируем вектор нулями
    tfidf_vector = [0.0] * len(vocabulary)
    
    # Для каждого уникального токена в тексте вычисляем tf-idf
    for token, count in token_counter.items():
        if token in vocabulary:
            term_index = vocabulary[token]
            
            # Вычисляем TF для термина в данном документе
            tf = compute_tf(count, total_tokens)
            
            # Вычисляем IDF для термина на основе обучающего корпуса
            idf = compute_idf(num_docs, term_doc_counts, term_index)
            
            # Вычисляем TF-IDF
            tfidf = tf * idf
            tfidf_vector[term_index] = tfidf
    
    return tfidf_vector


if __name__ == "__main__":
    # Пути
    base_dir = Path(__file__).parent
    vocab_path = base_dir / "output" / "vocabulary.json"
    matrix_path = base_dir / "output" / "term_document_matrix.json"
    
    print("=" * 60)
    print("Преобразование текста в вектор tf-idf")
    print("=" * 60)
    
    # Загружаем словарь и матрицу
    print("\n1. Загрузка словаря и матрицы...")
    vocabulary, num_docs, term_doc_counts = initialize_tfidf_model(vocab_path, matrix_path)
    
    print(f"   Размер словаря: {len(vocabulary)} токенов")
    print(f"   Количество документов в корпусе: {num_docs}")
    
    # Примеры текстов для тестирования
    test_texts = [
        "The president announced a new policy on foreign affairs.",
        "Software development requires programming skills and knowledge.",
        "Company shares increased by ten percent yesterday.",
    ]
    
    print("\n2. Преобразование текстов в векторы tf-idf...")
    for i, text in enumerate(test_texts, 1):
        print(f"\n   Пример {i}:")
        print(f"   Текст: {text}")
        
        tfidf_vector = text_to_tfidf_vector(text, vocabulary, num_docs, term_doc_counts)
        
        # Подсчитываем ненулевые значения
        non_zero_count = sum(1 for x in tfidf_vector if x > 0)
        max_tfidf = max(tfidf_vector) if tfidf_vector else 0.0
        max_index = tfidf_vector.index(max_tfidf) if max_tfidf > 0 else -1
        
        print(f"   Размер вектора: {len(tfidf_vector)}")
        print(f"   Ненулевых значений: {non_zero_count}")
        print(f"   Максимальное tf-idf: {max_tfidf:.6f}")
        
        # Находим токен с максимальным tf-idf
        if max_index >= 0:
            # Находим токен по индексу
            token_at_max = None
            for token, idx in vocabulary.items():
                if idx == max_index:
                    token_at_max = token
                    break
            print(f"   Токен с максимальным tf-idf: '{token_at_max}' (индекс {max_index})")
        
        # Показываем топ-5 токенов по tf-idf
        indexed_values = [(idx, val) for idx, val in enumerate(tfidf_vector) if val > 0]
        indexed_values.sort(key=lambda x: x[1], reverse=True)
        
        print(f"   Топ-5 токенов по tf-idf:")
        for rank, (idx, val) in enumerate(indexed_values[:5], 1):
            token = None
            for tok, tok_idx in vocabulary.items():
                if tok_idx == idx:
                    token = tok
                    break
            print(f"      {rank}. '{token}': {val:.6f}")
    
    print("\n" + "=" * 60)
    print("Готово!")
    print("=" * 60)

