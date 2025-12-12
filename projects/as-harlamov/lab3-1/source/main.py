"""
Главный модуль для выполнения лабораторной работы №3.1.
"""
import argparse
import json
import warnings
from pathlib import Path
import numpy as np

# Подавляем warnings для чистого вывода
warnings.filterwarnings('ignore')

from data_loader import load_train_test_data, load_vectors_from_tsv, load_labels_from_csv
from experiments import run_iteration_experiments, run_kernel_comparison, run_experiment
from dimensionality import (
    experiment_dimension_dropout,
    experiment_pca_reduction,
    experiment_mathematical_features
)
from vector_generator import generate_vectors_for_dataset


def generate_vectors_if_needed(
    train_csv_path: Path,
    test_csv_path: Path,
    train_vectors_path: Path,
    test_vectors_path: Path,
    w2v_model_path: Path
):
    """Генерирует векторы, если они не существуют."""
    if not train_vectors_path.exists():
        print(f"Генерация векторов для обучающей выборки...")
        generate_vectors_for_dataset(train_csv_path, w2v_model_path, train_vectors_path)
    
    if not test_vectors_path.exists():
        print(f"Генерация векторов для тестовой выборки...")
        generate_vectors_for_dataset(test_csv_path, w2v_model_path, test_vectors_path)


def main():
    # Подавляем все warnings для чистого вывода
    warnings.filterwarnings('ignore')
    import os
    os.environ['PYTHONWARNINGS'] = 'ignore'
    
    parser = argparse.ArgumentParser(description='Лабораторная работа 3.1: Классификация текстов')
    parser.add_argument('--lab2_path', type=str, default='../lab2', 
                       help='Путь к директории lab2')
    parser.add_argument('--generate_vectors', action='store_true',
                       help='Сгенерировать векторы для train/test датасетов')
    parser.add_argument('--skip_basic', action='store_true',
                       help='Пропустить базовые эксперименты')
    parser.add_argument('--skip_dimension', action='store_true',
                       help='Пропустить эксперименты с размерностью')
    parser.add_argument('--sample_size', type=int, default=None,
                       help='Использовать подвыборку данных указанного размера (для ускорения на CPU)')
    
    args = parser.parse_args()
    
    # Определяем пути
    lab2_path = Path(args.lab2_path).resolve()
    lab3_path = Path(__file__).parent.parent
    
    train_csv = lab2_path / 'assets' / 'source-corpus' / 'train.csv'
    test_csv = lab2_path / 'assets' / 'source-corpus' / 'test.csv'
    w2v_model = lab2_path / 'assets' / 'models' / 'w2v_trained.model'
    
    assets_dir = lab3_path / 'assets'
    assets_dir.mkdir(exist_ok=True)
    train_vectors = assets_dir / 'train_vectors.tsv'
    test_vectors = lab2_path / 'assets' / 'embeddings' / 'test_vectors.tsv'
    
    # Генерируем векторы, если нужно
    if args.generate_vectors or not train_vectors.exists():
        if not w2v_model.exists():
            print(f"ОШИБКА: Модель Word2Vec не найдена по пути {w2v_model}")
            print("Пожалуйста, сначала обучите модель в lab2")
            return
        generate_vectors_if_needed(train_csv, test_csv, train_vectors, test_vectors, w2v_model)
    
    # Загружаем данные
    print("\n" + "="*60)
    print("Загрузка данных...")
    print("="*60)
    X_train, y_train, X_test, y_test = load_train_test_data(
        train_csv, test_csv, train_vectors, test_vectors
    )
    
    # Опционально используем подвыборку для ускорения
    if args.sample_size is not None:
        print(f"\n⚠ Используется подвыборка размером {args.sample_size} образцов для ускорения")
        if args.sample_size < len(X_train):
            indices = np.random.RandomState(42).choice(len(X_train), size=args.sample_size, replace=False)
            X_train = X_train[indices]
            y_train = y_train[indices]
        # Для тестовой выборки используем меньшую подвыборку (например, 10% от train)
        test_sample_size = min(args.sample_size // 10, len(X_test))
        if test_sample_size < len(X_test) and test_sample_size > 0:
            test_indices = np.random.RandomState(42).choice(len(X_test), size=test_sample_size, replace=False)
            X_test = X_test[test_indices]
            y_test = y_test[test_indices]
    
    num_classes = len(np.unique(np.concatenate([y_train, y_test])))
    print(f"Обучающая выборка: {len(X_train)} образцов, {X_train.shape[1]} признаков")
    print(f"Тестовая выборка: {len(X_test)} образцов, {X_test.shape[1]} признаков")
    print(f"Количество классов: {num_classes}")
    print(f"Классы: {sorted(np.unique(y_train))}")
    
    all_results = {}
    
    if not args.skip_basic:
        # Эксперимент 1: SVM с разными kernel функциями и количеством итераций
        print("\n" + "="*60)
        print("Эксперимент 1: SVM с различными kernel функциями")
        print("="*60)
        
        # Используем только линейный и RBF kernel для ускорения (poly и sigmoid медленнее)
        kernels = ['linear', 'rbf']
        kernel_results = {}
        
        for kernel in kernels:
            print(f"\n--- Kernel: {kernel} ---")
            # Уменьшаем количество экспериментов для ускорения
            max_iters = [500, 1000, 2000]
            results = run_iteration_experiments(
                X_train, y_train, X_test, y_test,
                model_type='svm',
                kernel=kernel,
                max_iters=max_iters,
                num_classes=num_classes
            )
            kernel_results[kernel] = results
        
        all_results['svm_kernels'] = kernel_results
        
        # Эксперимент 2: Сравнение с MLP
        print("\n" + "="*60)
        print("Эксперимент 2: Сравнение SVM и MLP")
        print("="*60)
        
        mlp_results = run_iteration_experiments(
            X_train, y_train, X_test, y_test,
            model_type='mlp',
            kernel=None,
            max_iters=[500, 1000, 2000],  # Уменьшено для ускорения
            num_classes=num_classes
        )
        all_results['mlp'] = mlp_results
        
        # Находим оптимальные параметры
        print("\n" + "="*60)
        print("Анализ результатов")
        print("="*60)
        
        best_svm_result = None
        best_svm_score = 0
        best_mlp_result = None
        best_mlp_score = 0
        
        for kernel, results in kernel_results.items():
            for result in results:
                score = result['f1_score']
                if score > best_svm_score:
                    best_svm_score = score
                    best_svm_result = result
        
        for result in mlp_results:
            score = result['f1_score']
            if score > best_mlp_score:
                best_mlp_score = score
                best_mlp_result = result
        
        print(f"\nЛучший SVM результат:")
        print(f"  Kernel: {best_svm_result['kernel']}")
        print(f"  Max iterations: {best_svm_result['max_iter']}")
        print(f"  Accuracy: {best_svm_result['accuracy']:.4f}")
        print(f"  F1-score: {best_svm_result['f1_score']:.4f}")
        print(f"  Training time: {best_svm_result['training_time']:.2f}s")
        
        print(f"\nЛучший MLP результат:")
        print(f"  Max iterations: {best_mlp_result['max_iter']}")
        print(f"  Accuracy: {best_mlp_result['accuracy']:.4f}")
        print(f"  F1-score: {best_mlp_result['f1_score']:.4f}")
        print(f"  Training time: {best_mlp_result['training_time']:.2f}s")
        
        # Выбираем лучшую модель для дальнейших экспериментов
        if best_svm_score >= best_mlp_score:
            best_model_type = 'svm'
            best_kernel = best_svm_result['kernel']
            best_max_iter = best_svm_result['max_iter']
            print(f"\n✓ Выбрана модель: SVM ({best_kernel}) с {best_max_iter} итерациями")
        else:
            best_model_type = 'mlp'
            best_kernel = None
            best_max_iter = best_mlp_result['max_iter']
            print(f"\n✓ Выбрана модель: MLP с {best_max_iter} итерациями")
        
        all_results['best_model'] = {
            'type': best_model_type,
            'kernel': best_kernel,
            'max_iter': best_max_iter
        }
    
    if not args.skip_dimension:
        # Эксперимент 3: Изменение размерности
        print("\n" + "="*60)
        print("Эксперимент 3: Изменение размерности векторных представлений")
        print("="*60)
        
        if args.skip_basic:
            # Используем значения по умолчанию
            best_model_type = 'svm'
            best_kernel = 'linear'
            best_max_iter = 1000
        else:
            best_model_type = all_results['best_model']['type']
            best_kernel = all_results['best_model']['kernel']
            best_max_iter = all_results['best_model']['max_iter']
        
        dimension_results = {}
        
        # 3.1: Отбрасывание случайных размерностей
        print("\n--- 3.1: Отбрасывание случайных размерностей ---")
        # Уменьшаем количество экспериментов для ускорения
        num_drops = [0, 20, 40]
        dropout_experiments = experiment_dimension_dropout(X_train, X_test, num_drops)
        
        dropout_results = []
        for num_drop, X_train_new, X_test_new in dropout_experiments:
            print(f"  Эксперимент: отброшено {num_drop} размерностей ({X_train_new.shape[1]} осталось)")
            result = run_experiment(
                X_train_new, y_train, X_test_new, y_test,
                model_type=best_model_type,
                kernel=best_kernel,
                max_iter=best_max_iter,
                num_classes=num_classes
            )
            result['num_dropped'] = num_drop
            result['remaining_dimensions'] = X_train_new.shape[1]
            dropout_results.append(result)
            print(f"    Accuracy: {result['accuracy']:.4f}, F1: {result['f1_score']:.4f}")
        
        dimension_results['dropout'] = dropout_results
        
        # 3.2: Сокращение размерности через PCA
        print("\n--- 3.2: Сокращение размерности через PCA ---")
        # Уменьшаем количество экспериментов для ускорения
        n_components_list = [100, 50, 20]
        pca_experiments = experiment_pca_reduction(X_train, X_test, n_components_list)
        
        pca_results = []
        for n_components, X_train_new, X_test_new, pca in pca_experiments:
            explained_variance = np.sum(pca.explained_variance_ratio_)
            print(f"  Эксперимент: PCA до {n_components} размерностей (объяснено {explained_variance:.2%} дисперсии)")
            result = run_experiment(
                X_train_new, y_train, X_test_new, y_test,
                model_type=best_model_type,
                kernel=best_kernel,
                max_iter=best_max_iter,
                num_classes=num_classes
            )
            result['n_components'] = n_components
            result['explained_variance'] = float(explained_variance)
            pca_results.append(result)
            print(f"    Accuracy: {result['accuracy']:.4f}, F1: {result['f1_score']:.4f}")
        
        dimension_results['pca'] = pca_results
        
        # 3.3: Добавление математических признаков
        print("\n--- 3.3: Добавление математических признаков ---")
        X_train_enhanced, X_test_enhanced = experiment_mathematical_features(X_train, X_test)
        print(f"  Исходная размерность: {X_train.shape[1]}")
        print(f"  Новая размерность: {X_train_enhanced.shape[1]}")
        
        enhanced_result = run_experiment(
            X_train_enhanced, y_train, X_test_enhanced, y_test,
            model_type=best_model_type,
            kernel=best_kernel,
            max_iter=best_max_iter,
            num_classes=num_classes
        )
        enhanced_result['original_dimensions'] = X_train.shape[1]
        enhanced_result['new_dimensions'] = X_train_enhanced.shape[1]
        dimension_results['mathematical_features'] = enhanced_result
        print(f"    Accuracy: {enhanced_result['accuracy']:.4f}, F1: {enhanced_result['f1_score']:.4f}")
        
        all_results['dimension_experiments'] = dimension_results
    
    # Сохранение результатов
    results_path = assets_dir / 'results.json'
    # Преобразуем numpy типы в Python типы для JSON
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        return obj
    
    all_results_converted = convert_numpy(all_results)
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(all_results_converted, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Результаты сохранены в {results_path}")
    print("\n" + "="*60)
    print("Эксперименты завершены!")
    print("="*60)


if __name__ == "__main__":
    main()

