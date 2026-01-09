"""
Скрипт для векторизации произвольного текста с использованием модели GloVe.
Модель обучается на обучающей выборке.
Реализация GloVe на PyTorch для эффективного обучения с поддержкой GPU.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
from collections import defaultdict, Counter
import pickle
import argparse
from tqdm import tqdm

from text_processing import STOP_WORDS, is_punctuation
from text_to_tfidf import tokenize_text


def get_device():
    """
    Определяет доступное устройство для вычислений.
    Приоритет: CUDA > MPS > CPU
    
    Returns:
        Строка с названием устройства ('cuda', 'mps' или 'cpu')
    """
    if torch.cuda.is_available():
        return 'cuda'
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


class GloVeDataset(Dataset):
    """
    Dataset для обучения GloVe.
    Хранит пары слов и их совстречаемость.
    """
    
    def __init__(self, pairs):
        """
        Args:
            pairs: Список кортежей (word_id, context_id, cooccurrence_count)
        """
        self.pairs = pairs
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        word_id, context_id, count = self.pairs[idx]
        return torch.LongTensor([word_id]), torch.LongTensor([context_id]), torch.FloatTensor([count])


class GloVeModel(nn.Module):
    """
    Модель GloVe на PyTorch.
    """
    
    def __init__(self, vocab_size, embedding_dim=100, alpha=0.75, 
                 x_max=100.0, device='cpu'):
        """
        Инициализация модели GloVe.
        
        Args:
            vocab_size: Размер словаря
            embedding_dim: Размерность векторного представления
            alpha: Параметр функции взвешивания
            x_max: Максимальное значение для функции взвешивания
            device: Устройство для вычислений ('cpu' или 'cuda')
        """
        super(GloVeModel, self).__init__()
        
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.alpha = alpha
        self.x_max = x_max
        self.device = device
        
        # Embedding слои для слов и контекстов
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.context_embeddings = nn.Embedding(vocab_size, embedding_dim)
        
        # Смещения (biases)
        self.word_biases = nn.Embedding(vocab_size, 1)
        self.context_biases = nn.Embedding(vocab_size, 1)
        
        # Инициализация весов
        self._init_weights()
        
    def _init_weights(self):
        """Инициализация весов случайными значениями."""
        init_range = 0.01
        self.word_embeddings.weight.data.uniform_(-init_range, init_range)
        self.context_embeddings.weight.data.uniform_(-init_range, init_range)
        self.word_biases.weight.data.uniform_(-init_range, init_range)
        self.context_biases.weight.data.uniform_(-init_range, init_range)
    
    def forward(self, word_ids, context_ids):
        """
        Прямой проход модели.
        
        Args:
            word_ids: Тензор с индексами слов
            context_ids: Тензор с индексами контекстов
            
        Returns:
            Предсказанные значения для пар слово-контекст
        """
        # Получаем векторы и смещения
        word_vectors = self.word_embeddings(word_ids)  # [batch_size, embedding_dim]
        context_vectors = self.context_embeddings(context_ids)  # [batch_size, embedding_dim]
        
        word_bias = self.word_biases(word_ids).squeeze()  # [batch_size]
        context_bias = self.context_biases(context_ids).squeeze()  # [batch_size]
        
        # Вычисляем скалярное произведение + смещения
        # [batch_size, embedding_dim] * [batch_size, embedding_dim] -> [batch_size]
        dot_product = (word_vectors * context_vectors).sum(dim=1)
        
        # Предсказание: w_i^T * w_j_tilde + b_i + b_j_tilde
        prediction = dot_product + word_bias + context_bias
        
        return prediction
    
    def _weight_function(self, x):
        """
        Функция взвешивания для GloVe.
        
        Args:
            x: Тензор со значениями совстречаемости
            
        Returns:
            Тензор с весами
        """
        # weight(x) = (x / x_max)^alpha if x < x_max else 1
        weights = torch.where(
            x < self.x_max,
            (x / self.x_max) ** self.alpha,
            torch.ones_like(x)
        )
        return weights
    
    def compute_loss(self, word_ids, context_ids, cooccurrence_counts):
        """
        Вычисляет функцию потерь GloVe.
        
        Args:
            word_ids: Индексы слов
            context_ids: Индексы контекстов
            cooccurrence_counts: Значения совстречаемости
            
        Returns:
            Значение функции потерь
        """
        # Предсказание модели
        predictions = self.forward(word_ids.squeeze(), context_ids.squeeze())
        
        # Целевое значение: log(1 + X_ij)
        targets = torch.log(1.0 + cooccurrence_counts.squeeze())
        
        # Ошибка
        error = predictions - targets
        
        # Веса
        weights = self._weight_function(cooccurrence_counts.squeeze())
        
        # Взвешенная квадратичная ошибка
        loss = weights * (error ** 2)
        
        return loss.mean()
    
    def get_embeddings(self):
        """
        Возвращает финальные векторы (усредненные word и context векторы).
        
        Returns:
            Тензор с векторными представлениями слов [vocab_size, embedding_dim]
        """
        # Усредняем word и context векторы (стандартная практика для GloVe)
        word_embs = self.word_embeddings.weight.data
        context_embs = self.context_embeddings.weight.data
        return (word_embs + context_embs) / 2.0
    
    def get_word_vector(self, word_id):
        """
        Получает вектор для слова по его ID.
        
        Args:
            word_id: ID слова в словаре
            
        Returns:
            Вектор слова
        """
        embeddings = self.get_embeddings()
        return embeddings[word_id]


def read_annotation_file(file_path):
    """
    Читает файл аннотации и извлекает предложения как списки токенов.
    Переиспользует логику из build_vocabulary_and_matrix для обработки токенов.
    
    Args:
        file_path: Путь к файлу аннотации в формате TSV
        
    Returns:
        Список предложений, где каждое предложение - список токенов
    """
    sentences = []
    current_sentence = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:  # Пустая строка между предложениями
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
                    continue
                
                parts = line.split('\t')
                if len(parts) >= 1:
                    token = parts[0].strip().lower()
                    # Используем ту же логику фильтрации, что и в build_vocabulary_and_matrix
                    if token and not is_punctuation(token) and token not in STOP_WORDS:
                        current_sentence.append(token)
            
            # Добавляем последнее предложение, если файл не заканчивается пустой строкой
            if current_sentence:
                sentences.append(current_sentence)
    except Exception as e:
        print(f"Ошибка при чтении файла {file_path}: {e}")
    
    return sentences


def collect_training_corpus(train_dir):
    """
    Собирает корпус из всех файлов аннотации в обучающей выборке.
    
    Args:
        train_dir: Путь к директории с обучающей выборкой
        
    Returns:
        Список предложений, где каждое предложение - список токенов
    """
    all_sentences = []
    
    if not train_dir.exists():
        raise FileNotFoundError(f"Директория {train_dir} не найдена")
    
    print(f"Чтение данных из {train_dir}...")
    
    # Проходим по всем поддиректориям (1, 2, 3, 4)
    for class_dir in sorted(train_dir.iterdir()):
        if not class_dir.is_dir():
            continue
            
        print(f"  Обработка директории {class_dir.name}...")
        doc_count = 0
        
        # Проходим по всем TSV файлам в директории класса
        for tsv_file in sorted(class_dir.glob("*.tsv")):
            sentences = read_annotation_file(tsv_file)
            all_sentences.extend(sentences)
            doc_count += 1
            
            if doc_count % 1000 == 0:
                print(f"    Обработано {doc_count} документов...")
        
        print(f"    Всего обработано {doc_count} документов из директории {class_dir.name}")
    
    print(f"Всего собрано {len(all_sentences)} предложений")
    return all_sentences


def build_cooccurrence_matrix(
    corpus,
    window_size=10,
    min_count=2,
    verbose=True
):
    """
    Строит матрицу совстречаемости слов.
    
    Args:
        corpus: Список предложений (каждое предложение - список токенов)
        window_size: Размер окна для построения матрицы совстречаемости
        min_count: Минимальная частота слова для включения в словарь
        verbose: Выводить ли информацию о прогрессе
        
    Returns:
        tuple: (word_to_id словарь, список пар (word_id, context_id, cooccurrence_count))
    """
    if verbose:
        print("Построение матрицы совстречаемости...")
    
    # Строим словарь
    word_counts = Counter()
    for sentence in corpus:
        word_counts.update(sentence)
    
    # Фильтруем редкие слова
    filtered_words = {word: count for word, count in word_counts.items() if count >= min_count}
    
    # Создаем словарь индексов
    word_to_id = {word: idx for idx, word in enumerate(sorted(filtered_words.keys()))}
    vocab_size = len(word_to_id)
    
    if verbose:
        print(f"Размер словаря: {vocab_size} токенов")
    
    # Строим матрицу совстречаемости
    cooccurrence = defaultdict(float)
    
    for sentence in corpus:
        sentence_words = [w for w in sentence if w in word_to_id]
        for i, center_word in enumerate(sentence_words):
            center_id = word_to_id[center_word]
            
            # Окно вокруг центрального слова
            start = max(0, i - window_size)
            end = min(len(sentence_words), i + window_size + 1)
            
            for j in range(start, end):
                if j == i:
                    continue
                context_word = sentence_words[j]
                context_id = word_to_id[context_word]
                
                # Расстояние между словами (для взвешивания)
                distance = abs(i - j)
                weight = 1.0 / distance
                
                # Обновляем счетчик совстречаемости
                pair = (center_id, context_id)
                cooccurrence[pair] += weight
    
    # Преобразуем в список пар для Dataset
    pairs = [(i, j, count) for (i, j), count in cooccurrence.items()]
    
    if verbose:
        print(f"Собрано {len(pairs)} пар слов")
        print("Матрица совстречаемости построена")
    
    return word_to_id, pairs


def train_glove_model(
    corpus,
    embedding_dim=100,
    window_size=10,
    epochs=30,
    learning_rate=0.05,
    batch_size=1000,
    device='cpu',
    verbose=True
):
    """
    Обучает модель GloVe на предоставленном корпусе.
    
    Args:
        corpus: Список предложений (каждое предложение - список токенов)
        embedding_dim: Размерность векторного представления
        window_size: Размер окна для обучения
        epochs: Количество эпох обучения
        learning_rate: Скорость обучения
        batch_size: Размер батча для обучения
        device: Устройство для обучения ('cpu' или 'cuda')
        verbose: Выводить ли информацию о прогрессе
        
    Returns:
        tuple: (обученная модель GloVe, словарь word_to_id)
    """
    if verbose:
        print("\n" + "=" * 60)
        print("Обучение модели GloVe (PyTorch)")
        print("=" * 60)
    
    # Строим матрицу совстречаемости
    word_to_id, pairs = build_cooccurrence_matrix(corpus, window_size=window_size, verbose=verbose)
    vocab_size = len(word_to_id)
    
    # Создаем модель
    model = GloVeModel(
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        device=device
    ).to(device)
    
    # Создаем Dataset и DataLoader
    dataset = GloVeDataset(pairs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Оптимизатор
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    if verbose:
        print(f"\nОбучение модели GloVe...")
        print(f"Параметры:")
        print(f"  - Размер словаря: {vocab_size}")
        print(f"  - Размерность векторов: {embedding_dim}")
        print(f"  - Размер окна: {window_size}")
        print(f"  - Эпох: {epochs}")
        print(f"  - Скорость обучения: {learning_rate}")
        print(f"  - Размер батча: {batch_size}")
        print(f"  - Устройство: {device}")
        print(f"  - Количество батчей: {len(dataloader)}")
    
    # Обучение
    model.train()
    
    # Создаем прогресс-бар для эпох
    epoch_pbar = tqdm(range(epochs), desc="Обучение", unit="эпоха", disable=not verbose)
    
    # История потерь для визуализации
    loss_history = []
    
    for epoch in epoch_pbar:
        total_loss = 0.0
        num_batches = 0
        
        # Прогресс-бар для батчей внутри эпохи
        batch_pbar = tqdm(
            dataloader, 
            desc=f"Эпоха {epoch + 1}/{epochs}",
            leave=False,
            unit="батч",
            disable=not verbose
        )
        
        for word_ids, context_ids, cooccurrence_counts in batch_pbar:
            word_ids = word_ids.to(device)
            context_ids = context_ids.to(device)
            cooccurrence_counts = cooccurrence_counts.to(device)
            
            # Обнуляем градиенты
            optimizer.zero_grad()
            
            # Вычисляем потери
            loss = model.compute_loss(word_ids, context_ids, cooccurrence_counts)
            
            # Обратное распространение
            loss.backward()
            
            # Обновление весов
            optimizer.step()
            
            batch_loss = loss.item()
            total_loss += batch_loss
            num_batches += 1
            
            # Обновляем прогресс-бар батчей с текущей ошибкой
            current_avg_loss = total_loss / num_batches
            batch_pbar.set_postfix({
                'loss': f'{batch_loss:.6f}',
                'avg_loss': f'{current_avg_loss:.6f}'
            })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        loss_history.append(avg_loss)
        
        # Обновляем прогресс-бар эпох с информацией о средней ошибке
        epoch_pbar.set_postfix({
            'avg_loss': f'{avg_loss:.6f}',
            'min_loss': f'{min(loss_history):.6f}' if loss_history else 'N/A'
        })
    
    # Закрываем прогресс-бары
    epoch_pbar.close()
    
    if verbose:
        print(f"\nФинальная средняя ошибка: {loss_history[-1]:.6f}")
        print(f"Минимальная ошибка: {min(loss_history):.6f}")
    
    # Сохраняем словарь в модели для удобства
    model.word_to_id = word_to_id
    model.id_to_word = {idx: word for word, idx in word_to_id.items()}
    
    if verbose:
        print("\nОбучение завершено!")
    
    return model, word_to_id


def save_glove_model(model, word_to_id, model_path):
    """
    Сохраняет обученную модель GloVe в файл.
    
    Args:
        model: Обученная модель GloVe
        word_to_id: Словарь слово -> ID
        model_path: Путь для сохранения модели
    """
    model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Сохраняем модель, словарь и параметры
    save_data = {
        'model_state_dict': model.state_dict(),
        'model_params': {
            'vocab_size': model.vocab_size,
            'embedding_dim': model.embedding_dim,
            'alpha': model.alpha,
            'x_max': model.x_max,
        },
        'word_to_id': word_to_id,
    }
    
    with open(model_path, 'wb') as f:
        pickle.dump(save_data, f)
    print(f"Модель сохранена в {model_path}")


def load_glove_model(model_path, device='cpu'):
    """
    Загружает обученную модель GloVe из файла.
    
    Args:
        model_path: Путь к файлу модели
        device: Устройство для загрузки модели
        
    Returns:
        tuple: (загруженная модель GloVe, словарь word_to_id)
    """
    with open(model_path, 'rb') as f:
        save_data = pickle.load(f)
    
    # Восстанавливаем параметры модели
    model_params = save_data['model_params']
    model = GloVeModel(
        vocab_size=model_params['vocab_size'],
        embedding_dim=model_params['embedding_dim'],
        alpha=model_params['alpha'],
        x_max=model_params['x_max'],
        device=device
    ).to(device)
    
    # Загружаем веса
    model.load_state_dict(save_data['model_state_dict'])
    
    # Восстанавливаем словарь
    word_to_id = save_data['word_to_id']
    model.word_to_id = word_to_id
    model.id_to_word = {idx: word for word, idx in word_to_id.items()}
    
    print(f"Модель загружена из {model_path}")
    return model, word_to_id


def text_to_glove_vector(
    text,
    glove_model,
    aggregation='mean',
    device='cpu'
):
    """
    Преобразует произвольный текст в векторное представление с использованием модели GloVe.
    
    Для текста, состоящего из нескольких токенов, векторное представление вычисляется
    как среднее (или другое агрегированное значение) векторных представлений токенов.
    
    Args:
        text: Входной текст для векторизации
        glove_model: Обученная модель GloVe
        aggregation: Способ агрегации векторов токенов ('mean', 'sum', 'max')
        device: Устройство для вычислений
        
    Returns:
        Векторное представление текста или None, если не найдено ни одного токена
    """
    # Токенизируем текст
    tokens = tokenize_text(text)
    
    if not tokens:
        return None
    
    # Получаем векторы для каждого токена
    vectors = []
    embeddings = glove_model.get_embeddings().cpu().numpy()
    
    for token in tokens:
        if token in glove_model.word_to_id:
            word_id = glove_model.word_to_id[token]
            vector = embeddings[word_id]
            vectors.append(vector)
    
    if not vectors:
        # Если ни один токен не найден в словаре, возвращаем нулевой вектор
        return np.zeros(glove_model.embedding_dim)
    
    # Агрегируем векторы
    vectors = np.array(vectors)
    
    if aggregation == 'mean':
        return np.mean(vectors, axis=0)
    elif aggregation == 'sum':
        return np.sum(vectors, axis=0)
    elif aggregation == 'max':
        return np.max(vectors, axis=0)
    else:
        raise ValueError(f"Неизвестный метод агрегации: {aggregation}")


def initialize_glove_model(
    train_dir=None,
    model_path=None,
    embedding_dim=100,
    window_size=10,
    epochs=30,
    batch_size=1000,
    device=None,
    retrain=False
):
    """
    Инициализирует модель GloVe, загружая существующую или обучая новую.
    Удобная функция для загрузки/обучения модели один раз при многократном использовании.
    
    Args:
        train_dir: Путь к директории с обучающей выборкой (используется для обучения)
        model_path: Путь к файлу модели (если None, используется путь по умолчанию)
        embedding_dim: Размерность векторного представления
        window_size: Размер окна для обучения
        epochs: Количество эпох обучения
        batch_size: Размер батча для обучения
        device: Устройство для обучения (если None, определяется автоматически)
        retrain: Если True, переобучить модель даже если файл существует
        
    Returns:
        tuple: (модель GloVe, словарь word_to_id)
    """
    # Определяем устройство
    if device is None:
        device = get_device()
    
    if model_path is None:
        base_dir = Path(__file__).parent
        model_path = base_dir / "output" / "glove_model.pkl"
    
    # Загружаем существующую модель, если она есть и не требуется переобучение
    if model_path.exists() and not retrain:
        print("Загрузка существующей модели GloVe...")
        return load_glove_model(model_path, device=device)
    
    # Обучаем новую модель
    if train_dir is None:
        base_dir = Path(__file__).parent
        train_dir = base_dir / "assets" / "annotated-corpus" / "train"
    
    print("Обучение новой модели GloVe...")
    corpus = collect_training_corpus(train_dir)
    model, word_to_id = train_glove_model(
        corpus,
        embedding_dim=embedding_dim,
        window_size=window_size,
        epochs=epochs,
        batch_size=batch_size,
        device=device
    )
    
    # Сохраняем модель
    save_glove_model(model, word_to_id, model_path)
    
    return model, word_to_id


if __name__ == "__main__":
    # Парсинг аргументов командной строки
    parser = argparse.ArgumentParser(
        description='Векторизация текста с использованием модели GloVe'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=None,
        help='Путь к файлу модели GloVe (по умолчанию: output/glove_model.pkl)'
    )
    parser.add_argument(
        '--train-dir',
        type=str,
        default=None,
        help='Путь к директории с обучающей выборкой (по умолчанию: assets/annotated-corpus/train)'
    )
    parser.add_argument(
        '--retrain',
        action='store_true',
        help='Переобучить модель даже если файл существует'
    )
    parser.add_argument(
        '--embedding-dim',
        type=int,
        default=100,
        help='Размерность векторного представления (по умолчанию: 100)'
    )
    parser.add_argument(
        '--window-size',
        type=int,
        default=10,
        help='Размер окна для обучения (по умолчанию: 10)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=30,
        help='Количество эпох обучения (по умолчанию: 30)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1000,
        help='Размер батча для обучения (по умолчанию: 1000)'
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
    
    if args.train_dir:
        train_dir = Path(args.train_dir)
        if not train_dir.is_absolute():
            train_dir = base_dir / train_dir
    else:
        train_dir = base_dir / "assets" / "annotated-corpus" / "train"
    
    # Определяем устройство
    device = get_device()
    print(f"Используемое устройство: {device}")
    
    print("=" * 60)
    print("Векторизация текста с использованием GloVe (PyTorch)")
    print("=" * 60)
    
    # Инициализируем или загружаем модель
    print("\n1. Инициализация модели GloVe...")
    print(f"   Путь к модели: {model_path}")
    glove_model, word_to_id = initialize_glove_model(
        train_dir=train_dir,
        model_path=model_path,
        embedding_dim=args.embedding_dim,
        window_size=args.window_size,
        epochs=args.epochs,
        batch_size=args.batch_size,
        device=device,
        retrain=args.retrain
    )
    
    print(f"   Размер словаря: {len(word_to_id)} токенов")
    print(f"   Размерность векторов: {glove_model.embedding_dim}")
    
    # Примеры текстов для тестирования
    test_texts = [
        "The president announced a new policy on foreign affairs.",
        "Software development requires programming skills and knowledge.",
        "Company shares increased by ten percent yesterday.",
    ]
    
    print("\n2. Векторизация текстов...")
    for i, text in enumerate(test_texts, 1):
        print(f"\n   Пример {i}:")
        print(f"   Текст: {text}")
        
        vector = text_to_glove_vector(text, glove_model, aggregation='mean', device=device)
        
        if vector is not None:
            print(f"   Размерность вектора: {len(vector)}")
            print(f"   Первые 10 компонент: {vector[:10]}")
            print(f"   Минимум: {np.min(vector):.6f}, Максимум: {np.max(vector):.6f}")
            print(f"   Среднее: {np.mean(vector):.6f}")
        else:
            print("   Не удалось векторизовать текст")
    
    print("\n" + "=" * 60)
    print("Готово!")
    print("=" * 60)
