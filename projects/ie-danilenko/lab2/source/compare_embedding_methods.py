"""
Скрипт для сравнения эффективности метода векторизации с использованием нейронных сетей (GloVe)
и базовых методов векторизации с последующим сокращением размерности (PCA).

Использует разработанный метод подсчета косинусного расстояния для сравнения.
"""

import numpy as np
import argparse
import pickle
from pathlib import Path
from text_to_glove import (
    initialize_glove_model,
    get_device
)
from text_to_tfidf import load_vocabulary, load_term_document_matrix
from demonstrate_glove_similarity import cosine_distance, get_word_vector
from apply_pca_to_basic_vectors import (
    text_to_frequency_vector,
    text_to_onehot_matrix,
    onehot_matrix_to_vector,
    text_to_frequency_matrix,
    frequency_matrix_to_vector,
    text_to_tfidf_vector
)


def get_word_vector_basic(
    word,
    vocabulary,
    method,
    pca_model=None,
    num_docs=None,
    term_doc_counts=None
):
    """
    Получает векторное представление слова базовым методом с применением PCA.
    
    Args:
        word: Слово
        vocabulary: Словарь токен -> индекс
        method: Метод векторизации ('frequency', 'onehot', 'frequency_matrix', 'tfidf')
        pca_model: Обученная модель PCA (если None, возвращается исходный вектор)
        num_docs: Количество документов (для TF-IDF)
        term_doc_counts: Словарь term_index -> количество документов (для TF-IDF)
        
    Returns:
        Вектор слова или None, если слово не найдено
    """
    word_lower = word.lower()
    
    # Создаем текст из одного слова для векторизации
    text = word_lower
    
    # Векторизация в зависимости от метода
    if method == 'frequency':
        vector = text_to_frequency_vector(text, vocabulary)
    elif method == 'onehot':
        matrix = text_to_onehot_matrix(text, vocabulary)
        vector = onehot_matrix_to_vector(matrix, method='mean')
    elif method == 'frequency_matrix':
        matrix = text_to_frequency_matrix(text, vocabulary)
        vector = frequency_matrix_to_vector(matrix, method='mean')
    elif method == 'tfidf':
        if num_docs is None or term_doc_counts is None:
            raise ValueError("Для TF-IDF необходимы num_docs и term_doc_counts")
        vector = text_to_tfidf_vector(text, vocabulary, num_docs, term_doc_counts)
    else:
        raise ValueError(f"Неизвестный метод: {method}")
    
    # Применяем PCA, если модель предоставлена
    if pca_model is not None:
        vector = pca_model.transform(vector.reshape(1, -1))[0]
    
    return vector


def load_pca_model(pca_path):
    """
    Загружает модель PCA из файла.
    
    Args:
        pca_path: Путь к файлу модели PCA
        
    Returns:
        Загруженная модель PCA
    """
    with open(pca_path, 'rb') as f:
        pca_model = pickle.load(f)
    return pca_model


def evaluate_word_pairs(
    word_pairs,
    glove_model,
    basic_methods,
    verbose=True
):
    """
    Оценивает пары слов с использованием различных методов векторизации.
    
    Args:
        word_pairs: Список кортежей (слово1, слово2, категория)
        glove_model: Модель GloVe
        basic_methods: Словарь {метод: (pca_model, vocabulary, num_docs, term_doc_counts)}
        verbose: Выводить ли информацию
        
    Returns:
        Словарь результатов {метод: {категория: [расстояния]}}
    """
    results = {}
    
    # Добавляем GloVe в результаты
    results['glove'] = {'close': [], 'distant': []}
    
    # Добавляем базовые методы
    for method_name in basic_methods.keys():
        results[method_name] = {'close': [], 'distant': []}
    
    if verbose:
        print(f"\nОценка {len(word_pairs)} пар слов...")
    
    for word1, word2, category in word_pairs:
        # GloVe
        vec1_glove = get_word_vector(word1, glove_model)
        vec2_glove = get_word_vector(word2, glove_model)
        
        if vec1_glove is not None and vec2_glove is not None:
            dist_glove = cosine_distance(vec1_glove, vec2_glove)
            results['glove'][category].append(dist_glove)
        
        # Базовые методы
        for method_name, (pca_model, vocabulary, num_docs, term_doc_counts) in basic_methods.items():
            vec1_basic = get_word_vector_basic(
                word1, vocabulary, method_name, pca_model, num_docs, term_doc_counts
            )
            vec2_basic = get_word_vector_basic(
                word2, vocabulary, method_name, pca_model, num_docs, term_doc_counts
            )
            
            if vec1_basic is not None and vec2_basic is not None:
                dist_basic = cosine_distance(vec1_basic, vec2_basic)
                results[method_name][category].append(dist_basic)
    
    return results


def compare_methods(
    glove_model_path,
    pca_models_dir,
    vocab_path,
    matrix_path,
    output_dir=None
):
    """
    Сравнивает эффективность различных методов векторизации.
    
    Args:
        glove_model_path: Путь к модели GloVe
        pca_models_dir: Директория с моделями PCA
        vocab_path: Путь к словарю
        matrix_path: Путь к матрице "термин-документ"
        output_dir: Директория для сохранения результатов
    """
    print("=" * 80)
    print("Сравнение эффективности методов векторизации")
    print("=" * 80)
    
    # Загружаем модель GloVe
    print("\n1. Загрузка модели GloVe...")
    device = get_device()
    glove_model, word_to_id = initialize_glove_model(
        model_path=glove_model_path,
        device=device,
        retrain=False
    )
    print(f"   Размер словаря GloVe: {len(word_to_id)}")
    print(f"   Размерность векторов: {glove_model.embedding_dim}")
    
    # Загружаем словарь и матрицу для базовых методов
    print("\n2. Загрузка словаря и матрицы для базовых методов...")
    vocabulary = load_vocabulary(vocab_path)
    num_docs, term_doc_counts = load_term_document_matrix(matrix_path)
    print(f"   Размер словаря: {len(vocabulary)}")
    print(f"   Количество документов: {num_docs}")
    
    # Загружаем модели PCA
    print("\n3. Загрузка моделей PCA...")
    basic_methods = {}
    
    method_configs = {
        'frequency': 'pca_frequency.pkl',
        'onehot': 'pca_onehot.pkl',
        'frequency_matrix': 'pca_frequency_matrix.pkl',
        'tfidf': 'pca_tfidf.pkl',
    }
    
    for method_name, pca_filename in method_configs.items():
        pca_path = pca_models_dir / pca_filename
        if pca_path.exists():
            pca_model = load_pca_model(pca_path)
            basic_methods[method_name] = (pca_model, vocabulary, num_docs, term_doc_counts)
            print(f"   Загружена модель PCA для метода '{method_name}'")
        else:
            print(f"   ⚠️  Модель PCA не найдена: {pca_path}")
    
    if not basic_methods:
        print("\n❌ Не найдено ни одной модели PCA! Запустите сначала apply_pca_to_basic_vectors.py")
        return
    
    # Определяем тестовые пары слов
    print("\n4. Создание тестового набора пар слов...")
    
    # Семантически близкие пары
    close_pairs = [
        ('cat', 'tiger', 'close'),
        ('cat', 'feline', 'close'),
        ('president', 'leader', 'close'),
        ('president', 'chief', 'close'),
        ('company', 'corporation', 'close'),
        ('company', 'business', 'close'),
        ('war', 'battle', 'close'),
        ('war', 'conflict', 'close'),
        ('software', 'program', 'close'),
        ('software', 'application', 'close'),
    ]
    
    # Семантически далекие пары
    distant_pairs = [
        ('cat', 'sentence', 'distant'),
        ('cat', 'computer', 'distant'),
        ('president', 'animal', 'distant'),
        ('president', 'food', 'distant'),
        ('company', 'nature', 'distant'),
        ('company', 'music', 'distant'),
        ('war', 'peace', 'distant'),
        ('war', 'love', 'distant'),
        ('software', 'animal', 'distant'),
        ('software', 'food', 'distant'),
    ]
    
    all_pairs = close_pairs + distant_pairs
    print(f"   Семантически близких пар: {len(close_pairs)}")
    print(f"   Семантически далеких пар: {len(distant_pairs)}")
    print(f"   Всего пар: {len(all_pairs)}")
    
    # Оцениваем пары
    print("\n5. Вычисление косинусных расстояний...")
    results = evaluate_word_pairs(all_pairs, glove_model, basic_methods, verbose=True)
    
    # Анализируем результаты
    print("\n6. Анализ результатов...")
    print("\n" + "=" * 80)
    print("РЕЗУЛЬТАТЫ СРАВНЕНИЯ МЕТОДОВ")
    print("=" * 80)
    
    method_stats = {}
    
    for method_name in ['glove'] + list(basic_methods.keys()):
        if method_name not in results:
            continue
        
        close_distances = results[method_name]['close']
        distant_distances = results[method_name]['distant']
        
        if not close_distances or not distant_distances:
            continue
        
        avg_close = np.mean(close_distances)
        avg_distant = np.mean(distant_distances)
        std_close = np.std(close_distances)
        std_distant = np.std(distant_distances)
        
        # Разница между средними расстояниями (чем больше, тем лучше)
        separation = avg_distant - avg_close
        
        method_stats[method_name] = {
            'avg_close': avg_close,
            'avg_distant': avg_distant,
            'std_close': std_close,
            'std_distant': std_distant,
            'separation': separation
        }
        
        print(f"\n{method_name.upper()}:")
        print(f"  Семантически близкие слова:")
        print(f"    Среднее расстояние: {avg_close:.6f} ± {std_close:.6f}")
        print(f"    Минимум: {min(close_distances):.6f}, Максимум: {max(close_distances):.6f}")
        print(f"  Семантически далекие слова:")
        print(f"    Среднее расстояние: {avg_distant:.6f} ± {std_distant:.6f}")
        print(f"    Минимум: {min(distant_distances):.6f}, Максимум: {max(distant_distances):.6f}")
        print(f"  Разделение (далекие - близкие): {separation:.6f}")
    
    # Сравнение методов
    print("\n" + "=" * 80)
    print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
    print("=" * 80)
    print(f"{'Метод':<20} {'Среднее (близкие)':<20} {'Среднее (далекие)':<20} {'Разделение':<15}")
    print("-" * 80)
    
    for method_name, stats in sorted(method_stats.items(), key=lambda x: x[1]['separation'], reverse=True):
        print(f"{method_name:<20} {stats['avg_close']:<20.6f} {stats['avg_distant']:<20.6f} {stats['separation']:<15.6f}")
    
    # Выводы
    print("\n" + "=" * 80)
    print("ВЫВОДЫ")
    print("=" * 80)
    
    # Находим метод с наибольшим разделением
    best_method = max(method_stats.items(), key=lambda x: x[1]['separation'])
    
    print(f"\n✅ Лучший метод по разделению близких и далеких слов: {best_method[0].upper()}")
    print(f"   Разделение: {best_method[1]['separation']:.6f}")
    
    # Сравнение с GloVe
    if 'glove' in method_stats:
        glove_separation = method_stats['glove']['separation']
        print(f"\n📊 Сравнение с GloVe (разделение: {glove_separation:.6f}):")
        
        for method_name, stats in method_stats.items():
            if method_name == 'glove':
                continue
            
            diff = stats['separation'] - glove_separation
            percent_diff = (diff / glove_separation) * 100 if glove_separation > 0 else 0
            
            if diff > 0:
                print(f"  {method_name}: лучше на {diff:.6f} ({percent_diff:+.2f}%)")
            elif diff < 0:
                print(f"  {method_name}: хуже на {abs(diff):.6f} ({abs(percent_diff):+.2f}%)")
            else:
                print(f"  {method_name}: одинаково")
    
    # Общий вывод
    print(f"\n📝 ЗАКЛЮЧЕНИЕ:")
    print(f"   На основе анализа косинусных расстояний для семантически близких и далеких пар слов:")
    
    if best_method[0] == 'glove':
        print(f"   - Метод GloVe (нейронные сети) показывает наилучшие результаты")
        print(f"   - GloVe лучше разделяет семантически близкие и далекие слова")
        print(f"   - Базовые методы с PCA показывают хорошие результаты, но уступают GloVe")
    else:
        print(f"   - Метод {best_method[0]} показывает наилучшие результаты")
        print(f"   - Базовые методы с PCA могут быть эффективнее GloVe на данном датасете")
        if 'glove' in method_stats:
            print(f"   - GloVe показывает разделение: {glove_separation:.6f}")
    
    print("\n" + "=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Сравнение эффективности методов векторизации'
    )
    parser.add_argument(
        '--glove-model',
        type=str,
        default=None,
        help='Путь к модели GloVe (по умолчанию: output/glove_model.pkl)'
    )
    parser.add_argument(
        '--pca-dir',
        type=str,
        default=None,
        help='Директория с моделями PCA (по умолчанию: output/)'
    )
    parser.add_argument(
        '--vocab',
        type=str,
        default=None,
        help='Путь к файлу словаря (по умолчанию: output/vocabulary.json)'
    )
    parser.add_argument(
        '--matrix',
        type=str,
        default=None,
        help='Путь к файлу матрицы "термин-документ" (по умолчанию: output/term_document_matrix.json)'
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
    
    if args.pca_dir:
        pca_models_dir = Path(args.pca_dir)
        if not pca_models_dir.is_absolute():
            pca_models_dir = base_dir / pca_models_dir
    else:
        pca_models_dir = base_dir / "output"
    
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
    
    compare_methods(
        glove_model_path=glove_model_path,
        pca_models_dir=pca_models_dir,
        vocab_path=vocab_path,
        matrix_path=matrix_path
    )

