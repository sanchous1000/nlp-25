"""
Скрипт для экспериментов с SVM для многоклассовой классификации.
Использует векторные представления документов из второй лабораторной работы.
"""

import numpy as np
import pandas as pd
import time
import json
import warnings
from pathlib import Path
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate
import argparse


def load_vectors_from_tsv(tsv_path, verbose=True):
    """
    Загружает векторы из TSV файла.
    
    Args:
        tsv_path: Путь к TSV файлу с векторами (формат: doc_id \t vector_components)
        verbose: Выводить ли информацию о прогрессе
        
    Returns:
        tuple: (векторы документов, идентификаторы документов)
    """
    if verbose:
        print(f"Загрузка векторов из {tsv_path}...")
    
    vectors = []
    doc_ids = []
    
    with open(tsv_path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) < 2:
                continue
            
            doc_id = parts[0]
            vector_components = [float(x) for x in parts[1:]]
            
            doc_ids.append(doc_id)
            vectors.append(vector_components)
    
    vectors = np.array(vectors)
    
    if verbose:
        print(f"Загружено {len(vectors)} векторов")
        print(f"Размерность векторов: {vectors.shape[1]}")
    
    return vectors, doc_ids


def load_labels_from_csv(csv_path, verbose=True):
    """
    Загружает метки классов из CSV файла.
    
    Args:
        csv_path: Путь к CSV файлу (формат: class, title, text)
        verbose: Выводить ли информацию о прогрессе
        
    Returns:
        Массив меток классов
    """
    if verbose:
        print(f"Загрузка меток из {csv_path}...")
    
    df = pd.read_csv(csv_path, header=None, names=['class', 'title', 'text'])
    labels = df['class'].astype(int).values
    
    if verbose:
        print(f"Загружено {len(labels)} меток")
        print(f"Количество классов: {len(np.unique(labels))}")
    
    return labels


def load_data_from_vectors_and_labels(vectors_tsv_path, labels_csv_path, verbose=True):
    """
    Загружает векторы из TSV и метки из CSV, сопоставляя их по порядку.
    
    Args:
        vectors_tsv_path: Путь к TSV файлу с векторами
        labels_csv_path: Путь к CSV файлу с метками
        verbose: Выводить ли информацию о прогрессе
        
    Returns:
        tuple: (векторы документов, метки классов)
    """
    vectors, doc_ids = load_vectors_from_tsv(vectors_tsv_path, verbose=verbose)
    labels = load_labels_from_csv(labels_csv_path, verbose=verbose)
    
    # Проверяем, что количество совпадает
    if len(vectors) != len(labels):
        if verbose:
            print(f"⚠️  Внимание: количество векторов ({len(vectors)}) не совпадает с количеством меток ({len(labels)})")
            print(f"   Используется минимальное значение: {min(len(vectors), len(labels))}")
        min_len = min(len(vectors), len(labels))
        vectors = vectors[:min_len]
        labels = labels[:min_len]
    
    if verbose:
        print(f"✅ Итоговое количество образцов: {len(vectors)}")
    
    return vectors, labels


def compute_confusion_matrix(y_true, y_pred, classes):
    """
    Вычисляет матрицу ошибок.
    
    Args:
        y_true: Истинные метки
        y_pred: Предсказанные метки
        classes: Список всех классов
        
    Returns:
        Матрица ошибок (n_classes x n_classes)
    """
    n_classes = len(classes)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    for true_label, pred_label in zip(y_true, y_pred):
        true_idx = class_to_idx[true_label]
        pred_idx = class_to_idx[pred_label]
        cm[true_idx, pred_idx] += 1
    
    return cm


def compute_precision(y_true, y_pred, classes):
    """
    Вычисляет precision для каждого класса.
    
    Args:
        y_true: Истинные метки
        y_pred: Предсказанные метки
        classes: Список всех классов
        
    Returns:
        Словарь {класс: precision}
    """
    cm = compute_confusion_matrix(y_true, y_pred, classes)
    precision_dict = {}
    
    for idx, cls in enumerate(classes):
        tp = cm[idx, idx]
        fp = np.sum(cm[:, idx]) - tp
        
        if tp + fp == 0:
            precision_dict[cls] = 0.0
        else:
            precision_dict[cls] = tp / (tp + fp)
    
    return precision_dict


def compute_recall(y_true, y_pred, classes):
    """
    Вычисляет recall для каждого класса.
    
    Args:
        y_true: Истинные метки
        y_pred: Предсказанные метки
        classes: Список всех классов
        
    Returns:
        Словарь {класс: recall}
    """
    cm = compute_confusion_matrix(y_true, y_pred, classes)
    recall_dict = {}
    
    for idx, cls in enumerate(classes):
        tp = cm[idx, idx]
        fn = np.sum(cm[idx, :]) - tp
        
        if tp + fn == 0:
            recall_dict[cls] = 0.0
        else:
            recall_dict[cls] = tp / (tp + fn)
    
    return recall_dict


def compute_f1_score(y_true, y_pred, classes):
    """
    Вычисляет F1-score для каждого класса.
    
    Args:
        y_true: Истинные метки
        y_pred: Предсказанные метки
        classes: Список всех классов
        
    Returns:
        Словарь {класс: f1_score}
    """
    precision_dict = compute_precision(y_true, y_pred, classes)
    recall_dict = compute_recall(y_true, y_pred, classes)
    f1_dict = {}
    
    for cls in classes:
        precision = precision_dict[cls]
        recall = recall_dict[cls]
        
        if precision + recall == 0:
            f1_dict[cls] = 0.0
        else:
            f1_dict[cls] = 2 * (precision * recall) / (precision + recall)
    
    return f1_dict


def compute_accuracy(y_true, y_pred):
    """
    Вычисляет accuracy.
    
    Args:
        y_true: Истинные метки
        y_pred: Предсказанные метки
        
    Returns:
        Accuracy
    """
    correct = np.sum(y_true == y_pred)
    total = len(y_true)
    return correct / total if total > 0 else 0.0


def compute_metrics(y_true, y_pred, classes):
    """
    Вычисляет все метрики.
    
    Args:
        y_true: Истинные метки
        y_pred: Предсказанные метки
        classes: Список всех классов
        
    Returns:
        Словарь с метриками
    """
    precision_dict = compute_precision(y_true, y_pred, classes)
    recall_dict = compute_recall(y_true, y_pred, classes)
    f1_dict = compute_f1_score(y_true, y_pred, classes)
    accuracy = compute_accuracy(y_true, y_pred)
    
    # Вычисляем средние значения
    precision = np.mean(list(precision_dict.values()))
    recall = np.mean(list(recall_dict.values()))
    f1 = np.mean(list(f1_dict.values()))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'per_class_precision': precision_dict,
        'per_class_recall': recall_dict,
        'per_class_f1': f1_dict
    }


def run_svm_experiment(X_train, y_train, X_test, y_test, kernel='linear', max_iter=1000, verbose=True):
    """
    Запускает эксперимент с SVM.
    
    Args:
        X_train: Обучающие векторы
        y_train: Обучающие метки
        X_test: Тестовые векторы
        y_test: Тестовые метки
        kernel: Тип kernel function
        max_iter: Максимальное количество итераций
        verbose: Выводить ли информацию
        
    Returns:
        Словарь с результатами эксперимента
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Эксперимент: SVM с kernel={kernel}, max_iter={max_iter}")
        print(f"{'='*80}")
    
    # Нормализация данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Обучение модели
    model = SVC(kernel=kernel, max_iter=max_iter, random_state=42)
    
    start_time = time.time()
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
        model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    # Предсказания
    y_pred = model.predict(X_test_scaled)
    
    # Вычисление метрик
    classes = sorted(np.unique(np.concatenate([y_train, y_test])))
    metrics = compute_metrics(y_test, y_pred, classes)
    
    results = {
        'model': 'SVM',
        'kernel': kernel,
        'max_iter': max_iter,
        'training_time': training_time,
        **metrics
    }
    
    if verbose:
        print(f"Training time: {training_time:.4f} секунд")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-score: {metrics['f1']:.4f}")
    
    return results


def run_mlp_experiment(X_train, y_train, X_test, y_test, max_iter=1000, hidden_layers=(100,), verbose=True):
    """
    Запускает эксперимент с MLP.
    
    Args:
        X_train: Обучающие векторы
        y_train: Обучающие метки
        X_test: Тестовые векторы
        y_test: Тестовые метки
        max_iter: Максимальное количество итераций
        hidden_layers: Размеры скрытых слоев
        verbose: Выводить ли информацию
        
    Returns:
        Словарь с результатами эксперимента
    """
    if verbose:
        print(f"\n{'='*80}")
        print(f"Эксперимент: MLP с hidden_layers={hidden_layers}, max_iter={max_iter}")
        print(f"{'='*80}")
    
    # Нормализация данных
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Обучение модели
    model = MLPClassifier(
        hidden_layer_sizes=hidden_layers,
        max_iter=max_iter,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1
    )
    
    start_time = time.time()
    model.fit(X_train_scaled, y_train)
    training_time = time.time() - start_time
    
    # Предсказания
    y_pred = model.predict(X_test_scaled)
    
    # Вычисление метрик
    classes = sorted(np.unique(np.concatenate([y_train, y_test])))
    metrics = compute_metrics(y_test, y_pred, classes)
    
    results = {
        'model': 'MLP',
        'hidden_layers': hidden_layers,
        'max_iter': max_iter,
        'training_time': training_time,
        **metrics
    }
    
    if verbose:
        print(f"Training time: {training_time:.4f} секунд")
        print(f"Accuracy: {metrics['accuracy']:.4f}")
        print(f"Precision: {metrics['precision']:.4f}")
        print(f"Recall: {metrics['recall']:.4f}")
        print(f"F1-score: {metrics['f1']:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Эксперименты с SVM для многоклассовой классификации'
    )
    parser.add_argument(
        '--train-vectors',
        type=str,
        default=None,
        help='Путь к TSV файлу с векторами обучающей выборки (по умолчанию: output/train_vectors.tsv)'
    )
    parser.add_argument(
        '--test-vectors',
        type=str,
        default=None,
        help='Путь к TSV файлу с векторами тестовой выборки (по умолчанию: output/test_vectors.tsv)'
    )
    parser.add_argument(
        '--train-csv',
        type=str,
        default=None,
        help='Путь к CSV файлу с метками обучающей выборки (по умолчанию: dataset/train.csv)'
    )
    parser.add_argument(
        '--test-csv',
        type=str,
        default=None,
        help='Путь к CSV файлу с метками тестовой выборки (по умолчанию: dataset/test.csv)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Путь к выходному JSON файлу с результатами (по умолчанию: output/svm_experiments.json)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Максимальное количество образцов для загрузки (для тестирования)'
    )
    parser.add_argument(
        '--kernels',
        nargs='+',
        default=['linear', 'rbf', 'poly'],
        help='Список kernel functions для экспериментов'
    )
    parser.add_argument(
        '--max-iters',
        nargs='+',
        type=int,
        default=[100, 500, 1000],
        help='Список значений max_iter для экспериментов'
    )
    
    args = parser.parse_args()
    
    # Пути
    base_dir = Path(__file__).parent
    
    if args.train_vectors:
        train_vectors_path = Path(args.train_vectors)
        if not train_vectors_path.is_absolute():
            train_vectors_path = base_dir / train_vectors_path
    else:
        train_vectors_path = base_dir / "output" / "train_vectors.tsv"
    
    if args.test_vectors:
        test_vectors_path = Path(args.test_vectors)
        if not test_vectors_path.is_absolute():
            test_vectors_path = base_dir / test_vectors_path
    else:
        test_vectors_path = base_dir / "output" / "test_vectors.tsv"
    
    if args.train_csv:
        train_csv_path = Path(args.train_csv)
        if not train_csv_path.is_absolute():
            train_csv_path = base_dir / train_csv_path
    else:
        train_csv_path = base_dir / "dataset" / "train.csv"
    
    if args.test_csv:
        test_csv_path = Path(args.test_csv)
        if not test_csv_path.is_absolute():
            test_csv_path = base_dir / test_csv_path
    else:
        test_csv_path = base_dir / "dataset" / "test.csv"
    
    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = base_dir / output_path
    else:
        output_path = base_dir / "output" / "svm_experiments.json"
    
    # Проверка наличия файлов
    if not train_vectors_path.exists():
        print(f"❌ Ошибка: файл с векторами обучающей выборки не найден: {train_vectors_path}")
        print(f"   Создайте его, используя vectorize_documents.py для обучающей выборки")
        return
    
    if not test_vectors_path.exists():
        print(f"❌ Ошибка: файл с векторами тестовой выборки не найден: {test_vectors_path}")
        print(f"   Создайте его, используя vectorize_documents.py для тестовой выборки")
        return
    
    if not train_csv_path.exists():
        print(f"❌ Ошибка: файл с метками обучающей выборки не найден: {train_csv_path}")
        return
    
    if not test_csv_path.exists():
        print(f"❌ Ошибка: файл с метками тестовой выборки не найден: {test_csv_path}")
        return
    
    # Загрузка данных из уже векторизованных файлов
    print("=" * 80)
    print("Загрузка обучающей выборки...")
    print("=" * 80)
    X_train, y_train = load_data_from_vectors_and_labels(
        train_vectors_path,
        train_csv_path,
        verbose=True
    )
    
    if args.max_samples:
        X_train = X_train[:args.max_samples]
        y_train = y_train[:args.max_samples]
        print(f"Ограничено до {args.max_samples} образцов")
    
    print("\n" + "=" * 80)
    print("Загрузка тестовой выборки...")
    print("=" * 80)
    X_test, y_test = load_data_from_vectors_and_labels(
        test_vectors_path,
        test_csv_path,
        verbose=True
    )
    
    if args.max_samples:
        X_test = X_test[:args.max_samples // 5]  # Меньше для теста
        y_test = y_test[:args.max_samples // 5]
    
    # Проведение экспериментов
    print("\n" + "=" * 80)
    print("Начало экспериментов")
    print("=" * 80)
    
    all_results = []
    
    # Эксперименты с SVM
    for kernel in args.kernels:
        for max_iter in args.max_iters:
            results = run_svm_experiment(
                X_train, y_train, X_test, y_test,
                kernel=kernel,
                max_iter=max_iter,
                verbose=True
            )
            all_results.append(results)
    
    for max_iter in args.max_iters:
        results = run_mlp_experiment(
            X_train, y_train, X_test, y_test,
            max_iter=max_iter,
            hidden_layers=(100,),
            verbose=True
        )
        all_results.append(results)
    
    # Сохранение результатов
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Преобразуем numpy типы в Python типы для JSON
    json_results = []
    for result in all_results:
        json_result = {}
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                json_result[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                json_result[key] = float(value)
            elif isinstance(value, dict):
                json_result[key] = {str(k): float(v) if isinstance(v, (np.integer, np.floating)) else v 
                                   for k, v in value.items()}
            elif isinstance(value, tuple):
                json_result[key] = list(value)
            else:
                json_result[key] = value
        json_results.append(json_result)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2, ensure_ascii=False)
    
    print("\n" + "=" * 80)
    print("Результаты сохранены в", output_path)
    print("=" * 80)
    
    # Вывод сводки в виде таблицы
    print("\n" + "=" * 80)
    print("Сводка результатов")
    print("=" * 80)
    
    # Подготовка данных для таблицы
    table_data = []
    headers = ["Модель", "Kernel/Layers", "Max Iter", "Accuracy", "Precision", "Recall", "F1-score", "Time (s)"]
    
    for result in all_results:
        model_name = result['model']
        if model_name == 'SVM':
            kernel = result['kernel']
            max_iter = result['max_iter']
            table_data.append([
                model_name,
                kernel,
                max_iter,
                f"{result['accuracy']:.4f}",
                f"{result['precision']:.4f}",
                f"{result['recall']:.4f}",
                f"{result['f1']:.4f}",
                f"{result['training_time']:.4f}"
            ])
        else:
            layers = str(result['hidden_layers'])
            max_iter = result['max_iter']
            table_data.append([
                model_name,
                layers,
                max_iter,
                f"{result['accuracy']:.4f}",
                f"{result['precision']:.4f}",
                f"{result['recall']:.4f}",
                f"{result['f1']:.4f}",
                f"{result['training_time']:.4f}"
            ])
    
    print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".4f"))


if __name__ == "__main__":
    main()

