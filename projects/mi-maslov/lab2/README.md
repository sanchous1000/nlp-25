# Лабораторная работа №2: Векторизация текста

## Описание
Реализация методов векторизации текста с использованием TF-IDF и Word2Vec на датасете [AG News](https://huggingface.co/datasets/ag_news).

## Выполненные задачи

### 1. Построение словаря и матрицы термин-документ
- Предобработка текста (lowercase, удаление спецсимволов)
- Фильтрация токенов по минимальной частоте (min_freq=2)
- Построение словаря и term-document матрицы

### 2. TF-IDF векторизация
- Вычисление Term Frequency (TF)
- Вычисление Inverse Document Frequency (IDF)
- Формирование TF-IDF матрицы

### 3. Word2Vec модель
- Обучение модели Word2Vec на корпусе (vector_size=100, window=5, min_count=2)
- Векторизация документов усреднением эмбеддингов токенов
- Сохранение эмбеддингов в формате TSV

### 4. Сокращение размерности
- Применение PCA для снижения размерности до 50 компонент
- Сохранение редуцированных эмбеддингов

### 5. Анализ семантической близости
Проверка ближайших соседей для тестовых слов:

### Результаты

```
Nearest neighbors for 'news':
  profile: 0.9814
  research: 0.9776
  quote: 0.9720
  toy: 0.9667
  firm: 0.9601

Nearest neighbors for 'market':
  price: 0.9756
  stock: 0.9747
  slashed: 0.9456
  slashing: 0.9365
  range: 0.9351

Nearest neighbors for 'technology':
  business: 0.9849
  database: 0.9804
  management: 0.9782
  chip: 0.9780
  manage: 0.9769

Nearest neighbors for 'government':
  witnesses: 0.9210
  military: 0.9173
  warned: 0.9170
  aides: 0.9123
  official: 0.9080
```