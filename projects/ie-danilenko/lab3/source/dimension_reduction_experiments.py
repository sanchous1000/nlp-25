"""
Скрипт для экспериментов с преобразованиями векторных представлений документов.
Выполняет эксперименты с сокращением размерности векторов и анализирует влияние на метрики.
"""

import numpy as np
import json
from pathlib import Path
from sklearn.decomposition import PCA
from tabulate import tabulate
import argparse

from svm_classification_experiments import (
    load_data_from_vectors_and_labels,
    run_svm_experiment
)


def reduce_dimension_pca(X_train, X_test, target_dim, verbose=True):
    """
    Сокращает размерность векторов с использованием PCA.
    
    Args:
        X_train: Обучающие векторы
        X_test: Тестовые векторы
        target_dim: Целевая размерность
        verbose: Выводить ли информацию
        
    Returns:
        tuple: (X_train_reduced, X_test_reduced, pca_model)
    """
    if verbose:
        print(f"Применение PCA: {X_train.shape[1]} -> {target_dim} размерностей")
    
    pca = PCA(n_components=target_dim, random_state=42)
    X_train_reduced = pca.fit_transform(X_train)
    X_test_reduced = pca.transform(X_test)
    
    explained_variance = np.sum(pca.explained_variance_ratio_)
    
    if verbose:
        print(f"Объясненная дисперсия: {explained_variance:.4f} ({explained_variance*100:.2f}%)")
    
    return X_train_reduced, X_test_reduced, pca, explained_variance


def run_dimension_reduction_experiments(
    X_train, y_train, X_test, y_test,
    dimensions,
    kernel='linear',
    max_iter=100,
    verbose=True
):
    """
    Проводит серию экспериментов с разными размерностями.
    
    Args:
        X_train: Обучающие векторы
        y_train: Обучающие метки
        X_test: Тестовые векторы
        y_test: Тестовые метки
        dimensions: Список целевых размерностей для экспериментов
        kernel: Тип kernel function для SVM
        max_iter: Максимальное количество итераций
        verbose: Выводить ли информацию
        
    Returns:
        Список результатов экспериментов
    """
    original_dim = X_train.shape[1]
    all_results = []
    
    if verbose:
        print("\n" + "=" * 80)
        print(f"Эксперименты с сокращением размерности")
        print(f"Исходная размерность: {original_dim}")
        print(f"Kernel: {kernel}, Max Iter: {max_iter}")
        print("=" * 80)
    
    # Эксперимент с исходной размерностью (базовая линия)
    if verbose:
        print(f"\n{'='*80}")
        print(f"Базовый эксперимент (размерность: {original_dim})")
        print(f"{'='*80}")
    
    results = run_svm_experiment(
        X_train, y_train, X_test, y_test,
        kernel=kernel,
        max_iter=max_iter,
        verbose=verbose
    )
    results['dimension'] = original_dim
    results['explained_variance'] = 1.0
    all_results.append(results)
    
    # Эксперименты с разными размерностями
    for dim in dimensions:
        if dim >= original_dim:
            if verbose:
                print(f"\n⚠️  Пропуск: целевая размерность {dim} >= исходной {original_dim}")
            continue
        
        if verbose:
            print(f"\n{'='*80}")
            print(f"Эксперимент с размерностью: {dim}")
            print(f"{'='*80}")
        
        # Сокращаем размерность
        X_train_reduced, X_test_reduced, pca, explained_variance = reduce_dimension_pca(
            X_train, X_test, dim, verbose=verbose
        )
        
        # Запускаем эксперимент
        results = run_svm_experiment(
            X_train_reduced, y_train, X_test_reduced, y_test,
            kernel=kernel,
            max_iter=max_iter,
            verbose=verbose
        )
        results['dimension'] = dim
        results['explained_variance'] = explained_variance
        all_results.append(results)
    
    return all_results


def main():
    parser = argparse.ArgumentParser(
        description='Эксперименты с сокращением размерности векторных представлений'
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
        help='Путь к выходному JSON файлу с результатами (по умолчанию: output/dimension_reduction_experiments.json)'
    )
    parser.add_argument(
        '--dimensions',
        nargs='+',
        type=int,
        default=[10, 20, 30, 40, 50, 75, 99],
        help='Список целевых размерностей для экспериментов'
    )
    parser.add_argument(
        '--kernel',
        type=str,
        default='linear',
        help='Тип kernel function для SVM (по умолчанию: linear)'
    )
    parser.add_argument(
        '--max-iter',
        type=int,
        default=100,
        help='Максимальное количество итераций (по умолчанию: 500)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Максимальное количество образцов для загрузки (для тестирования)'
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
        output_path = base_dir / "output" / "dimension_reduction_experiments.json"
    
    # Проверка наличия файлов
    if not train_vectors_path.exists():
        print(f"❌ Ошибка: файл с векторами обучающей выборки не найден: {train_vectors_path}")
        return
    
    if not test_vectors_path.exists():
        print(f"❌ Ошибка: файл с векторами тестовой выборки не найден: {test_vectors_path}")
        return
    
    if not train_csv_path.exists():
        print(f"❌ Ошибка: файл с метками обучающей выборки не найден: {train_csv_path}")
        return
    
    if not test_csv_path.exists():
        print(f"❌ Ошибка: файл с метками тестовой выборки не найден: {test_csv_path}")
        return
    
    # Загрузка данных
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
        X_test = X_test[:args.max_samples // 5]
        y_test = y_test[:args.max_samples // 5]
    
    # Проведение экспериментов
    all_results = run_dimension_reduction_experiments(
        X_train, y_train, X_test, y_test,
        dimensions=args.dimensions,
        kernel=args.kernel,
        max_iter=args.max_iter,
        verbose=True
    )
    
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
    print("Сводка результатов экспериментов с сокращением размерности")
    print("=" * 80)
    
    table_data = []
    headers = ["Размерность", "Explained Variance", "Accuracy", "Precision", "Recall", "F1-score", "Time (s)"]
    
    for result in all_results:
        table_data.append([
            result['dimension'],
            f"{result['explained_variance']:.4f}",
            f"{result['accuracy']:.4f}",
            f"{result['precision']:.4f}",
            f"{result['recall']:.4f}",
            f"{result['f1']:.4f}",
            f"{result['training_time']:.4f}"
        ])
    
    print(tabulate(table_data, headers=headers, tablefmt="grid", floatfmt=".4f"))
    
    # Анализ зависимости метрик от размерности
    print("\n" + "=" * 80)
    print("Анализ зависимости метрик от размерности")
    print("=" * 80)
    
    dimensions = [r['dimension'] for r in all_results]
    accuracies = [r['accuracy'] for r in all_results]
    f1_scores = [r['f1'] for r in all_results]
    explained_variances = [r['explained_variance'] for r in all_results]
    
    # Находим оптимальную размерность (максимальный F1-score)
    best_idx = np.argmax(f1_scores)
    best_dim = dimensions[best_idx]
    best_f1 = f1_scores[best_idx]
    best_acc = accuracies[best_idx]
    
    print(f"\nОптимальная размерность: {best_dim}")
    print(f"  F1-score: {best_f1:.4f}")
    print(f"  Accuracy: {best_acc:.4f}")
    print(f"  Explained Variance: {explained_variances[best_idx]:.4f}")
    
    # Анализ изменений
    original_f1 = f1_scores[0]
    original_acc = accuracies[0]
    
    print(f"\nСравнение с исходной размерностью ({dimensions[0]}):")
    print(f"  F1-score: {original_f1:.4f} -> {best_f1:.4f} ({'+' if best_f1 > original_f1 else ''}{best_f1 - original_f1:.4f})")
    print(f"  Accuracy: {original_acc:.4f} -> {best_acc:.4f} ({'+' if best_acc > original_acc else ''}{best_acc - original_acc:.4f})")
    
    if best_f1 < original_f1:
        print(f"\n⚠️  Сокращение размерности ухудшило качество модели")
    elif best_f1 > original_f1:
        print(f"\n✅ Сокращение размерности улучшило качество модели")
    else:
        print(f"\n➡️  Сокращение размерности не изменило качество модели")


if __name__ == "__main__":
    main()

