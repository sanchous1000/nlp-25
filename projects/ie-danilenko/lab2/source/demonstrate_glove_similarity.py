"""
Скрипт для демонстрации работы модели GloVe с использованием косинусного расстояния.
Показывает, что для семантически близких слов модель генерирует векторы,
для которых косинусное расстояние меньше, чем для семантически далеких токенов.
"""

import numpy as np
import argparse
from pathlib import Path
from text_to_glove import (
    initialize_glove_model,
    get_device,
    GloVeModel
)


def cosine_distance(vec1, vec2):
    """
    Вычисляет косинусное расстояние между двумя векторами.
    
    Косинусное расстояние = 1 - косинусное сходство
    Косинусное сходство = (A · B) / (||A|| * ||B||)
    
    Args:
        vec1: Первый вектор
        vec2: Второй вектор
        
    Returns:
        Косинусное расстояние (от 0 до 2, где 0 - одинаковые векторы)
    """
    # Проверяем, что векторы не нулевые
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 2.0  # Максимальное расстояние для нулевых векторов
    
    # Вычисляем косинусное сходство
    dot_product = np.dot(vec1, vec2)
    cosine_similarity = dot_product / (norm1 * norm2)
    
    # Косинусное расстояние = 1 - косинусное сходство
    # Результат в диапазоне [0, 2], где 0 - одинаковые векторы
    cosine_distance = 1.0 - cosine_similarity
    
    return cosine_distance


def get_word_vector(word, glove_model):
    """
    Получает векторное представление слова из модели GloVe.
    
    Args:
        word: Слово (в нижнем регистре)
        glove_model: Обученная модель GloVe
        
    Returns:
        Вектор слова или None, если слово не найдено в словаре
    """
    word_lower = word.lower()
    if word_lower not in glove_model.word_to_id:
        return None
    
    embeddings = glove_model.get_embeddings().cpu().numpy()
    word_id = glove_model.word_to_id[word_lower]
    return embeddings[word_id]


def find_similar_words(
    target_word,
    candidate_words,
    glove_model,
    top_k=None
):
    """
    Находит наиболее похожие слова из списка кандидатов по косинусному расстоянию.
    
    Args:
        target_word: Исходное слово
        candidate_words: Список слов-кандидатов
        glove_model: Обученная модель GloVe
        top_k: Количество лучших результатов (None - вернуть все)
        
    Returns:
        Список кортежей (слово, косинусное_расстояние), отсортированный по возрастанию расстояния
    """
    target_vector = get_word_vector(target_word, glove_model)
    if target_vector is None:
        return []
    
    distances = []
    for word in candidate_words:
        word_vector = get_word_vector(word, glove_model)
        if word_vector is not None:
            dist = cosine_distance(target_vector, word_vector)
            distances.append((word, dist))
    
    # Сортируем по возрастанию расстояния (меньше = ближе)
    distances.sort(key=lambda x: x[1])
    
    if top_k is not None:
        return distances[:top_k]
    return distances


def demonstrate_word_similarity(
    target_word,
    similar_words,
    same_domain_words,
    different_words,
    glove_model
):
    """
    Демонстрирует косинусное расстояние для разных групп слов.
    
    Args:
        target_word: Исходное слово
        similar_words: Слова с похожим значением
        same_domain_words: Слова из той же предметной области
        different_words: Слова с совершенно другими свойствами
        glove_model: Обученная модель GloVe
    """
    print(f"\n{'='*80}")
    print(f"Анализ слова: '{target_word}'")
    print(f"{'='*80}")
    
    # Проверяем наличие исходного слова
    target_vector = get_word_vector(target_word, glove_model)
    if target_vector is None:
        print(f"⚠️  Слово '{target_word}' не найдено в словаре модели!")
        return
    
    # Собираем все слова и их категории
    all_words = []
    all_words.extend([(word, "Похожее значение") for word in similar_words])
    all_words.extend([(word, "Та же область") for word in same_domain_words])
    all_words.extend([(word, "Другое значение") for word in different_words])
    
    # Вычисляем расстояния
    results = []
    for word, category in all_words:
        word_vector = get_word_vector(word, glove_model)
        if word_vector is None:
            results.append((word, category, None, "Не найдено в словаре"))
        else:
            dist = cosine_distance(target_vector, word_vector)
            results.append((word, category, dist, None))
    
    # Фильтруем слова, найденные в словаре
    valid_results = [(w, c, d, _) for w, c, d, _ in results if d is not None]
    not_found = [(w, c) for w, c, _, msg in results if msg is not None]
    
    if not_found:
        print(f"\n⚠️  Слова не найдены в словаре: {', '.join([w for w, _ in not_found])}")
    
    if not valid_results:
        print("\n❌ Нет доступных слов для сравнения!")
        return
    
    # Сортируем по расстоянию
    valid_results.sort(key=lambda x: x[2])
    
    # Выводим результаты
    print(f"\nРанжированный список слов по косинусному расстоянию:")
    print(f"{'Ранг':<6} {'Слово':<20} {'Категория':<25} {'Косинусное расстояние':<25}")
    print("-" * 80)
    
    for rank, (word, category, distance, _) in enumerate(valid_results, 1):
        print(f"{rank:<6} {word:<20} {category:<25} {distance:<25.6f}")
    
    # Статистика по категориям
    print(f"\n📊 Статистика по категориям:")
    
    similar_distances = [d for _, c, d, _ in valid_results if c == "Похожее значение"]
    domain_distances = [d for _, c, d, _ in valid_results if c == "Та же область"]
    different_distances = [d for _, c, d, _ in valid_results if c == "Другое значение"]
    
    if similar_distances:
        avg_similar = np.mean(similar_distances)
        print(f"  Похожее значение:    среднее = {avg_similar:.6f}, "
              f"мин = {min(similar_distances):.6f}, макс = {max(similar_distances):.6f}")
    
    if domain_distances:
        avg_domain = np.mean(domain_distances)
        print(f"  Та же область:       среднее = {avg_domain:.6f}, "
              f"мин = {min(domain_distances):.6f}, макс = {max(domain_distances):.6f}")
    
    if different_distances:
        avg_different = np.mean(different_distances)
        print(f"  Другое значение:     среднее = {avg_different:.6f}, "
              f"мин = {min(different_distances):.6f}, макс = {max(different_distances):.6f}")
    
    # Проверка гипотезы
    print(f"\n✅ Проверка гипотезы:")
    checks = []
    
    if similar_distances and domain_distances:
        max_similar = max(similar_distances)
        min_domain = min(domain_distances)
        check1 = max_similar < min_domain
        checks.append(check1)
        status1 = "✓" if check1 else "✗"
        print(f"  {status1} Похожие < Та же область: "
              f"max(похожие)={max_similar:.6f} < min(область)={min_domain:.6f}")
    
    if domain_distances and different_distances:
        max_domain = max(domain_distances)
        min_different = min(different_distances)
        check2 = max_domain < min_different
        checks.append(check2)
        status2 = "✓" if check2 else "✗"
        print(f"  {status2} Та же область < Другое: "
              f"max(область)={max_domain:.6f} < min(другое)={min_different:.6f}")
    
    if all(checks):
        print(f"\n🎉 Гипотеза подтверждена! Семантически близкие слова имеют меньшее косинусное расстояние.")
    else:
        print(f"\n⚠️  Гипотеза частично подтверждена. Некоторые расстояния не соответствуют ожиданиям.")


if __name__ == "__main__":
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(
        description='Демонстрация работы модели GloVe с косинусным расстоянием'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Путь к файлу модели GloVe (по умолчанию: output/glove_model.pkl)'
    )
    
    args = parser.parse_args()
    
    # Пути
    base_dir = Path(__file__).parent
    
    if args.model:
        model_path = Path(args.model)
        if not model_path.is_absolute():
            model_path = base_dir / model_path
    else:
        model_path = base_dir / "output" / "glove_model.pkl"
    
    # Определяем устройство
    device = get_device()
    
    print("=" * 80)
    print("Демонстрация работы модели GloVe с косинусным расстоянием")
    print("=" * 80)
    
    # Загружаем модель
    print("\n1. Загрузка модели GloVe...")
    print(f"   Путь к модели: {model_path}")
    glove_model, word_to_id = initialize_glove_model(
        model_path=model_path,
        device=device,
        retrain=False
    )
    
    print(f"   Размер словаря: {len(word_to_id)} токенов")
    print(f"   Размерность векторов: {glove_model.embedding_dim}")
    
    # Определяем тестовые наборы слов
    # Каждый набор содержит: исходное слово, похожие, из той же области, разные
    
    test_cases = [
        {
            'target': 'cat',
            'similar': ['tiger', 'feline', 'kitten'],
            'same_domain': ['animal', 'rabbit', 'dog'],
            'different': ['sentence', 'creation', 'computer']
        },
        {
            'target': 'president',
            'similar': ['leader', 'chief', 'executive'],
            'same_domain': ['government', 'politics', 'election'],
            'different': ['animal', 'food', 'music']
        },
        {
            'target': 'company',
            'similar': ['corporation', 'business', 'firm'],
            'same_domain': ['market', 'economy', 'trade'],
            'different': ['animal', 'nature', 'science']
        },
        {
            'target': 'software',
            'similar': ['program', 'application', 'system'],
            'same_domain': ['computer', 'technology', 'development'],
            'different': ['animal', 'food', 'music']
        },
        {
            'target': 'war',
            'similar': ['battle', 'conflict', 'fighting'],
            'same_domain': ['military', 'soldier', 'weapon'],
            'different': ['peace', 'love', 'happiness']
        }
    ]
    
    print(f"\n2. Анализ {len(test_cases)} тестовых случаев...")
    
    # Демонстрируем для каждого тестового случая
    for i, test_case in enumerate(test_cases, 1):
        demonstrate_word_similarity(
            test_case['target'],
            test_case['similar'],
            test_case['same_domain'],
            test_case['different'],
            glove_model
        )
    
    print(f"\n{'='*80}")
    print("Демонстрация завершена!")
    print(f"{'='*80}")

