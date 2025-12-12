"""
Скрипт для анализа результатов экспериментов классификации.
"""
import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import seaborn as sns

# Настройка для корректного отображения русского текста
matplotlib.rcParams['font.family'] = 'DejaVu Sans'
try:
    plt.style.use('seaborn-v0_8-darkgrid')
except OSError:
    try:
        plt.style.use('seaborn-darkgrid')
    except OSError:
        plt.style.use('default')
sns.set_palette("husl")


def load_results(results_path: Path) -> Dict:
    """Загружает результаты из JSON файла."""
    with open(results_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def plot_svm_kernels_comparison(kernel_results: Dict, output_dir: Path) -> None:
    """Строит графики сравнения kernel функций SVM."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Comparison of SVM Kernel Functions', fontsize=16, fontweight='bold')
    
    kernels = list(kernel_results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(kernels)))
    
    # График F1-score по итерациям
    ax1 = axes[0, 0]
    for kernel, color in zip(kernels, colors):
        results = kernel_results[kernel]
        iters = [r['max_iter'] for r in results]
        f1_scores = [r['f1_score'] for r in results]
        ax1.plot(iters, f1_scores, marker='o', label=kernel, color=color, linewidth=2)
    ax1.set_xlabel('Max Iterations')
    ax1.set_ylabel('F1-score')
    ax1.set_title('F1-score vs Iterations')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # График Accuracy по итерациям
    ax2 = axes[0, 1]
    for kernel, color in zip(kernels, colors):
        results = kernel_results[kernel]
        iters = [r['max_iter'] for r in results]
        accuracies = [r['accuracy'] for r in results]
        ax2.plot(iters, accuracies, marker='s', label=kernel, color=color, linewidth=2)
    ax2.set_xlabel('Max Iterations')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Iterations')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # График времени обучения
    ax3 = axes[1, 0]
    kernel_names = []
    times = []
    for kernel, results in kernel_results.items():
        best = max(results, key=lambda x: x['f1_score'])
        kernel_names.append(kernel)
        times.append(best['training_time'])
    bars = ax3.bar(kernel_names, times, color=colors[:len(kernel_names)])
    ax3.set_xlabel('Kernel')
    ax3.set_ylabel('Training Time (seconds)')
    ax3.set_title('Training Time Comparison')
    ax3.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom')
    
    # Сравнение лучших метрик
    ax4 = axes[1, 1]
    metrics = ['F1-score', 'Accuracy', 'Precision', 'Recall']
    x = np.arange(len(metrics))
    width = 0.8 / len(kernels)
    
    for i, kernel in enumerate(kernels):
        best = max(kernel_results[kernel], key=lambda x: x['f1_score'])
        values = [
            best['f1_score'],
            best['accuracy'],
            best['precision'],
            best['recall']
        ]
        offset = (i - len(kernels)/2 + 0.5) * width
        ax4.bar(x + offset, values, width, label=kernel, color=colors[i])
    
    ax4.set_xlabel('Metrics')
    ax4.set_ylabel('Score')
    ax4.set_title('Best Metrics Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(metrics)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    ax4.set_ylim([0, 1])
    
    plt.tight_layout()
    output_path = output_dir / 'svm_kernels_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ График сохранен: {output_path}")


def analyze_svm_kernels(kernel_results: Dict, output_dir: Path = None) -> None:
    """Анализирует результаты экспериментов с различными kernel функциями SVM."""
    print("\n" + "="*70)
    print("АНАЛИЗ 1: Сравнение различных kernel функций SVM")
    print("="*70)
    
    kernel_summary = {}
    
    for kernel, results in kernel_results.items():
        best_result = max(results, key=lambda x: x['f1_score'])
        kernel_summary[kernel] = {
            'best_iter': best_result['max_iter'],
            'accuracy': best_result['accuracy'],
            'f1_score': best_result['f1_score'],
            'precision': best_result['precision'],
            'recall': best_result['recall'],
            'training_time': best_result['training_time']
        }
    
    # Создаем DataFrame для удобного отображения
    df_data = []
    for kernel, summary in kernel_summary.items():
        df_data.append({
            'Kernel': kernel,
            'Лучшие итерации': summary['best_iter'],
            'Accuracy': f"{summary['accuracy']:.4f}",
            'F1-score': f"{summary['f1_score']:.4f}",
            'Precision': f"{summary['precision']:.4f}",
            'Recall': f"{summary['recall']:.4f}",
            'Время обучения (с)': f"{summary['training_time']:.2f}"
        })
    
    df = pd.DataFrame(df_data)
    print("\nСводная таблица лучших результатов для каждого kernel:")
    print(df.to_string(index=False))
    
    # Определяем лучший kernel
    best_kernel = max(kernel_summary.items(), key=lambda x: x[1]['f1_score'])
    print(f"\n✓ Лучший kernel: {best_kernel[0]}")
    print(f"  F1-score: {best_kernel[1]['f1_score']:.4f}")
    print(f"  Accuracy: {best_kernel[1]['accuracy']:.4f}")
    print(f"  Оптимальное количество итераций: {best_kernel[1]['best_iter']}")
    
    # Строим графики
    if output_dir:
        plot_svm_kernels_comparison(kernel_results, output_dir)


def plot_iterations_analysis(results: List[Dict], model_name: str, output_dir: Path) -> None:
    """Строит графики зависимости метрик от количества итераций."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Metrics vs Iterations: {model_name}', fontsize=16, fontweight='bold')
    
    iters = [r['max_iter'] for r in results]
    
    # F1-score
    ax1 = axes[0, 0]
    f1_scores = [r['f1_score'] for r in results]
    ax1.plot(iters, f1_scores, marker='o', color='#2ecc71', linewidth=2, markersize=8)
    ax1.set_xlabel('Max Iterations')
    ax1.set_ylabel('F1-score')
    ax1.set_title('F1-score vs Iterations')
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2 = axes[0, 1]
    accuracies = [r['accuracy'] for r in results]
    ax2.plot(iters, accuracies, marker='s', color='#3498db', linewidth=2, markersize=8)
    ax2.set_xlabel('Max Iterations')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Accuracy vs Iterations')
    ax2.grid(True, alpha=0.3)
    
    # Training time
    ax3 = axes[1, 0]
    times = [r['training_time'] for r in results]
    ax3.plot(iters, times, marker='^', color='#e74c3c', linewidth=2, markersize=8)
    ax3.set_xlabel('Max Iterations')
    ax3.set_ylabel('Training Time (seconds)')
    ax3.set_title('Training Time vs Iterations')
    ax3.grid(True, alpha=0.3)
    
    # Все метрики вместе (нормализованные)
    ax4 = axes[1, 1]
    ax4.plot(iters, f1_scores, marker='o', label='F1-score', linewidth=2)
    ax4.plot(iters, accuracies, marker='s', label='Accuracy', linewidth=2)
    ax4.plot(iters, [r['precision'] for r in results], marker='^', label='Precision', linewidth=2)
    ax4.plot(iters, [r['recall'] for r in results], marker='d', label='Recall', linewidth=2)
    ax4.set_xlabel('Max Iterations')
    ax4.set_ylabel('Score')
    ax4.set_title('All Metrics Comparison')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim([0, 1])
    
    plt.tight_layout()
    safe_name = model_name.replace(' ', '_').replace('(', '').replace(')', '').replace('/', '_')
    output_path = output_dir / f'iterations_analysis_{safe_name}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ График сохранен: {output_path}")


def analyze_iterations(results: List[Dict], model_name: str, output_dir: Path = None) -> None:
    """Анализирует влияние количества итераций на качество модели."""
    print(f"\n" + "="*70)
    print(f"АНАЛИЗ 2: Влияние количества итераций ({model_name})")
    print("="*70)
    
    df_data = []
    for result in results:
        df_data.append({
            'Итерации': result['max_iter'],
            'Accuracy': result['accuracy'],
            'F1-score': result['f1_score'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'Время (с)': result['training_time']
        })
    
    df = pd.DataFrame(df_data)
    print("\nРезультаты по итерациям:")
    print(df.to_string(index=False))
    
    # Находим оптимальное количество итераций
    best_result = max(results, key=lambda x: x['f1_score'])
    print(f"\n✓ Оптимальное количество итераций: {best_result['max_iter']}")
    print(f"  F1-score: {best_result['f1_score']:.4f}")
    print(f"  Accuracy: {best_result['accuracy']:.4f}")
    
    # Проверяем, есть ли значительное улучшение при увеличении итераций
    if len(results) > 1:
        improvements = []
        for i in range(1, len(results)):
            prev_f1 = results[i-1]['f1_score']
            curr_f1 = results[i]['f1_score']
            improvement = curr_f1 - prev_f1
            improvements.append((results[i-1]['max_iter'], results[i]['max_iter'], improvement))
        
        print("\nУлучшение при увеличении итераций:")
        for prev_iter, curr_iter, improvement in improvements:
            if improvement > 0.01:  # Значимое улучшение
                print(f"  {prev_iter} → {curr_iter}: +{improvement:.4f} F1-score")
            elif improvement < -0.01:  # Ухудшение
                print(f"  {prev_iter} → {curr_iter}: {improvement:.4f} F1-score (ухудшение)")
            else:
                print(f"  {prev_iter} → {curr_iter}: {improvement:.4f} F1-score (без значимых изменений)")
    
    # Строим графики
    if output_dir:
        plot_iterations_analysis(results, model_name, output_dir)


def plot_svm_mlp_comparison(svm_results: Dict, mlp_results: List[Dict], output_dir: Path) -> None:
    """Строит графики сравнения SVM и MLP."""
    # Находим лучшие результаты
    best_svm = None
    best_svm_score = 0
    for kernel, results in svm_results.items():
        for result in results:
            if result['f1_score'] > best_svm_score:
                best_svm_score = result['f1_score']
                best_svm = result
    
    best_mlp = max(mlp_results, key=lambda x: x['f1_score'])
    
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle('SVM vs MLP Comparison', fontsize=16, fontweight='bold')
    
    # Сравнение метрик
    ax1 = axes[0]
    metrics = ['F1-score', 'Accuracy', 'Precision', 'Recall']
    svm_values = [
        best_svm['f1_score'],
        best_svm['accuracy'],
        best_svm['precision'],
        best_svm['recall']
    ]
    mlp_values = [
        best_mlp['f1_score'],
        best_mlp['accuracy'],
        best_mlp['precision'],
        best_mlp['recall']
    ]
    
    x = np.arange(len(metrics))
    width = 0.35
    ax1.bar(x - width/2, svm_values, width, label=f'SVM ({best_svm["kernel"]})', color='#3498db')
    ax1.bar(x + width/2, mlp_values, width, label='MLP', color='#e74c3c')
    ax1.set_xlabel('Metrics')
    ax1.set_ylabel('Score')
    ax1.set_title('Metrics Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(metrics)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim([0, 1])
    
    # Время обучения
    ax2 = axes[1]
    models = [f'SVM\n({best_svm["kernel"]})', 'MLP']
    times = [best_svm['training_time'], best_mlp['training_time']]
    bars = ax2.bar(models, times, color=['#3498db', '#e74c3c'])
    ax2.set_ylabel('Training Time (seconds)')
    ax2.set_title('Training Time Comparison')
    ax2.grid(True, alpha=0.3, axis='y')
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom')
    
    # F1-score по итерациям
    ax3 = axes[2]
    # Собираем данные для SVM (лучший kernel)
    svm_kernel = best_svm['kernel']
    svm_iters_data = [(r['max_iter'], r['f1_score']) for r in svm_results[svm_kernel]]
    svm_iters_data.sort(key=lambda x: x[0])
    svm_iters = [x[0] for x in svm_iters_data]
    svm_f1s = [x[1] for x in svm_iters_data]
    
    mlp_iters_data = [(r['max_iter'], r['f1_score']) for r in mlp_results]
    mlp_iters_data.sort(key=lambda x: x[0])
    mlp_iters = [x[0] for x in mlp_iters_data]
    mlp_f1s = [x[1] for x in mlp_iters_data]
    
    ax3.plot(svm_iters, svm_f1s, marker='o', label=f'SVM ({svm_kernel})', 
             color='#3498db', linewidth=2, markersize=8)
    ax3.plot(mlp_iters, mlp_f1s, marker='s', label='MLP', 
             color='#e74c3c', linewidth=2, markersize=8)
    ax3.set_xlabel('Max Iterations')
    ax3.set_ylabel('F1-score')
    ax3.set_title('F1-score vs Iterations')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    output_path = output_dir / 'svm_mlp_comparison.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ График сохранен: {output_path}")


def plot_confusion_matrix(conf_matrix: List[List[int]], title: str, output_dir: Path, class_labels: List[str] = None) -> None:
    """Строит heatmap для confusion matrix."""
    if class_labels is None:
        class_labels = [f'Class {i}' for i in range(len(conf_matrix))]
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_labels, yticklabels=class_labels,
                ax=ax, cbar_kws={'label': 'Count'})
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    ax.set_title(title)
    
    plt.tight_layout()
    safe_title = title.lower().replace(' ', '_').replace('(', '').replace(')', '')
    output_path = output_dir / f'confusion_matrix_{safe_title}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ График сохранен: {output_path}")


def compare_svm_mlp(svm_results: Dict, mlp_results: List[Dict], output_dir: Path = None) -> None:
    """Сравнивает SVM и MLP."""
    print("\n" + "="*70)
    print("АНАЛИЗ 3: Сравнение SVM и MLP")
    print("="*70)
    
    # Находим лучший результат SVM
    best_svm = None
    best_svm_score = 0
    for kernel, results in svm_results.items():
        for result in results:
            if result['f1_score'] > best_svm_score:
                best_svm_score = result['f1_score']
                best_svm = result
    
    # Находим лучший результат MLP
    best_mlp = max(mlp_results, key=lambda x: x['f1_score'])
    
    print("\nЛучшие результаты:")
    print(f"\nSVM:")
    print(f"  Kernel: {best_svm['kernel']}")
    print(f"  Итерации: {best_svm['max_iter']}")
    print(f"  Accuracy: {best_svm['accuracy']:.4f}")
    print(f"  F1-score: {best_svm['f1_score']:.4f}")
    print(f"  Precision: {best_svm['precision']:.4f}")
    print(f"  Recall: {best_svm['recall']:.4f}")
    print(f"  Время обучения: {best_svm['training_time']:.2f}с")
    
    print(f"\nMLP:")
    print(f"  Итерации: {best_mlp['max_iter']}")
    print(f"  Accuracy: {best_mlp['accuracy']:.4f}")
    print(f"  F1-score: {best_mlp['f1_score']:.4f}")
    print(f"  Precision: {best_mlp['precision']:.4f}")
    print(f"  Recall: {best_mlp['recall']:.4f}")
    print(f"  Время обучения: {best_mlp['training_time']:.2f}с")
    
    print("\nСравнение:")
    f1_diff = best_svm['f1_score'] - best_mlp['f1_score']
    acc_diff = best_svm['accuracy'] - best_mlp['accuracy']
    time_diff = best_svm['training_time'] - best_mlp['training_time']
    
    if f1_diff > 0:
        print(f"  ✓ SVM превосходит MLP на {f1_diff:.4f} по F1-score")
    elif f1_diff < 0:
        print(f"  ✓ MLP превосходит SVM на {abs(f1_diff):.4f} по F1-score")
    else:
        print(f"  ≈ Модели показывают схожие результаты по F1-score")
    
    if abs(time_diff) > 1:
        if time_diff > 0:
            print(f"  ⚠ SVM обучается медленнее на {time_diff:.2f}с")
        else:
            print(f"  ⚠ MLP обучается медленнее на {abs(time_diff):.2f}с")
    
    # Строим графики
    if output_dir:
        plot_svm_mlp_comparison(svm_results, mlp_results, output_dir)
        # Строим confusion matrix для лучших моделей
        plot_confusion_matrix(best_svm['confusion_matrix'], 
                            f'SVM ({best_svm["kernel"]}) - Best Model', 
                            output_dir)
        plot_confusion_matrix(best_mlp['confusion_matrix'], 
                            'MLP - Best Model', 
                            output_dir)


def plot_dimensionality_analysis(dimension_results: Dict, output_dir: Path) -> None:
    """Строит графики анализа изменения размерности."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle('Dimensionality Analysis', fontsize=16, fontweight='bold')
    
    # Dropout analysis
    if 'dropout' in dimension_results:
        ax1 = axes[0]
        dropout_data = dimension_results['dropout']
        num_dropped = [r['num_dropped'] for r in dropout_data]
        f1_scores = [r['f1_score'] for r in dropout_data]
        accuracies = [r['accuracy'] for r in dropout_data]
        
        ax1_twin = ax1.twinx()
        line1 = ax1.plot(num_dropped, f1_scores, marker='o', color='#2ecc71', 
                        label='F1-score', linewidth=2, markersize=8)
        line2 = ax1_twin.plot(num_dropped, accuracies, marker='s', color='#3498db', 
                             label='Accuracy', linewidth=2, markersize=8)
        
        ax1.set_xlabel('Dimensions Dropped')
        ax1.set_ylabel('F1-score', color='#2ecc71')
        ax1_twin.set_ylabel('Accuracy', color='#3498db')
        ax1.set_title('Random Dimension Dropout')
        ax1.grid(True, alpha=0.3)
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper right')
    
    # PCA analysis
    if 'pca' in dimension_results:
        ax2 = axes[1]
        pca_data = dimension_results['pca']
        n_components = [r['n_components'] for r in pca_data]
        f1_scores = [r['f1_score'] for r in pca_data]
        accuracies = [r['accuracy'] for r in pca_data]
        
        ax2_twin = ax2.twinx()
        line1 = ax2.plot(n_components, f1_scores, marker='o', color='#2ecc71', 
                        label='F1-score', linewidth=2, markersize=8)
        line2 = ax2_twin.plot(n_components, accuracies, marker='s', color='#3498db', 
                             label='Accuracy', linewidth=2, markersize=8)
        
        ax2.set_xlabel('PCA Components')
        ax2.set_ylabel('F1-score', color='#2ecc71')
        ax2_twin.set_ylabel('Accuracy', color='#3498db')
        ax2.set_title('PCA Dimensionality Reduction')
        ax2.grid(True, alpha=0.3)
        ax2.invert_xaxis()  # Инвертируем ось X, чтобы показывать уменьшение размерности
        
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax2.legend(lines, labels, loc='upper right')
    
    # Mathematical features comparison
    if 'mathematical_features' in dimension_results:
        ax3 = axes[2]
        result = dimension_results['mathematical_features']
        original_dims = result['original_dimensions']
        new_dims = result['new_dimensions']
        
        # Сравниваем с базовым результатом (предполагаем, что он в dropout[0])
        baseline_f1 = 0
        baseline_acc = 0
        if 'dropout' in dimension_results and len(dimension_results['dropout']) > 0:
            baseline_f1 = dimension_results['dropout'][0]['f1_score']
            baseline_acc = dimension_results['dropout'][0]['accuracy']
        
        categories = ['Baseline\n(original)', 'With Math\nFeatures']
        f1_values = [baseline_f1, result['f1_score']]
        acc_values = [baseline_acc, result['accuracy']]
        
        x = np.arange(len(categories))
        width = 0.35
        
        bars1 = ax3.bar(x - width/2, f1_values, width, label='F1-score', color='#2ecc71')
        bars2 = ax3.bar(x + width/2, acc_values, width, label='Accuracy', color='#3498db')
        
        ax3.set_ylabel('Score')
        ax3.set_title('Mathematical Features Impact')
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories)
        ax3.legend()
        ax3.grid(True, alpha=0.3, axis='y')
        ax3.set_ylim([0, 1])
        
        # Добавляем значения на столбцы
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    output_path = output_dir / 'dimensionality_analysis.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ График сохранен: {output_path}")


def analyze_dimensionality(dimension_results: Dict, output_dir: Path = None) -> None:
    """Анализирует влияние изменения размерности на качество классификации."""
    print("\n" + "="*70)
    print("АНАЛИЗ 4: Влияние изменения размерности")
    print("="*70)
    
    # Анализ dropout
    if 'dropout' in dimension_results:
        print("\n--- Отбрасывание случайных размерностей ---")
        dropout_data = []
        for result in dimension_results['dropout']:
            dropout_data.append({
                'Отброшено': result['num_dropped'],
                'Осталось': result['remaining_dimensions'],
                'Accuracy': result['accuracy'],
                'F1-score': result['f1_score']
            })
        
        df_dropout = pd.DataFrame(dropout_data)
        print(df_dropout.to_string(index=False))
        
        baseline = dropout_data[0]  # Первый результат (без отбрасывания)
        baseline_f1 = float(baseline['F1-score']) if isinstance(baseline['F1-score'], str) else baseline['F1-score']
        baseline_acc = float(baseline['Accuracy']) if isinstance(baseline['Accuracy'], str) else baseline['Accuracy']
        print(f"\nБазовый результат (без отбрасывания): F1={baseline_f1:.4f}, Acc={baseline_acc:.4f}")
        
        for item in dropout_data[1:]:
            item_f1 = float(item['F1-score']) if isinstance(item['F1-score'], str) else item['F1-score']
            item_acc = float(item['Accuracy']) if isinstance(item['Accuracy'], str) else item['Accuracy']
            f1_change = item_f1 - baseline_f1
            acc_change = item_acc - baseline_acc
            print(f"  Отброшено {item['Отброшено']} размерностей ({item['Осталось']} осталось): F1={f1_change:+.4f}, Acc={acc_change:+.4f}")
    
    # Анализ PCA
    if 'pca' in dimension_results:
        print("\n--- Сокращение размерности через PCA ---")
        pca_data = []
        for result in dimension_results['pca']:
            pca_data.append({
                'Компоненты': result['n_components'],
                'Объясненная дисперсия': f"{result['explained_variance']:.2%}",
                'Accuracy': result['accuracy'],
                'F1-score': result['f1_score']
            })
        
        df_pca = pd.DataFrame(pca_data)
        print(df_pca.to_string(index=False))
        
        baseline = pca_data[0]  # Первый результат (исходная размерность)
        baseline_f1 = float(baseline['F1-score']) if isinstance(baseline['F1-score'], str) else baseline['F1-score']
        baseline_acc = float(baseline['Accuracy']) if isinstance(baseline['Accuracy'], str) else baseline['Accuracy']
        print(f"\nБазовый результат (100 компонент): F1={baseline_f1:.4f}, Acc={baseline_acc:.4f}")
        
        for item in pca_data[1:]:
            item_f1 = float(item['F1-score']) if isinstance(item['F1-score'], str) else item['F1-score']
            item_acc = float(item['Accuracy']) if isinstance(item['Accuracy'], str) else item['Accuracy']
            f1_change = item_f1 - baseline_f1
            acc_change = item_acc - baseline_acc
            print(f"  PCA до {item['Компоненты']} компонент ({item['Объясненная дисперсия']}): F1={f1_change:+.4f}, Acc={acc_change:+.4f}")
    
    # Анализ математических признаков
    if 'mathematical_features' in dimension_results:
        print("\n--- Добавление математических признаков ---")
        result = dimension_results['mathematical_features']
        print(f"Исходная размерность: {result['original_dimensions']}")
        print(f"Новая размерность: {result['new_dimensions']}")
        print(f"Accuracy: {result['accuracy']:.4f}")
        print(f"F1-score: {result['f1_score']:.4f}")
        
        # Сравнение с базовым результатом (нужно знать, какой был базовый)
        print("\nПримечание: Для сравнения с базовым результатом используйте лучший результат из анализа SVM/MLP")
    
    # Строим графики
    if output_dir:
        plot_dimensionality_analysis(dimension_results, output_dir)


def generate_summary_report(results: Dict) -> None:
    """Генерирует итоговый отчет с выводами."""
    print("\n" + "="*70)
    print("ИТОГОВЫЕ ВЫВОДЫ")
    print("="*70)
    
    conclusions = []
    
    # 1. Оптимальное количество итераций
    if 'svm_kernels' in results:
        all_svm_results = []
        for kernel, kernel_results in results['svm_kernels'].items():
            all_svm_results.extend(kernel_results)
        best_svm = max(all_svm_results, key=lambda x: x['f1_score'])
        conclusions.append(f"1. Оптимальное количество итераций для SVM ({best_svm['kernel']}): {best_svm['max_iter']}")
    
    if 'mlp' in results:
        best_mlp = max(results['mlp'], key=lambda x: x['f1_score'])
        conclusions.append(f"2. Оптимальное количество итераций для MLP: {best_mlp['max_iter']}")
    
    # 2. Лучший kernel
    if 'svm_kernels' in results:
        kernel_scores = {}
        for kernel, kernel_results in results['svm_kernels'].items():
            best = max(kernel_results, key=lambda x: x['f1_score'])
            kernel_scores[kernel] = best['f1_score']
        best_kernel = max(kernel_scores.items(), key=lambda x: x[1])
        conclusions.append(f"3. Лучший kernel для SVM: {best_kernel[0]} (F1-score: {best_kernel[1]:.4f})")
    
    # 3. Сравнение моделей
    if 'svm_kernels' in results and 'mlp' in results:
        all_svm_results = []
        for kernel, kernel_results in results['svm_kernels'].items():
            all_svm_results.extend(kernel_results)
        best_svm = max(all_svm_results, key=lambda x: x['f1_score'])
        best_mlp = max(results['mlp'], key=lambda x: x['f1_score'])
        
        if best_svm['f1_score'] > best_mlp['f1_score']:
            conclusions.append(f"4. Лучшая модель: SVM ({best_svm['kernel']}) с F1-score {best_svm['f1_score']:.4f}")
        else:
            conclusions.append(f"4. Лучшая модель: MLP с F1-score {best_mlp['f1_score']:.4f}")
    
    # 5. Влияние размерности
    if 'dimension_experiments' in results:
        dim_exp = results['dimension_experiments']
        if 'dropout' in dim_exp and len(dim_exp['dropout']) > 0:
            baseline_f1 = dim_exp['dropout'][0]['f1_score']
            worst_dropout = min(dim_exp['dropout'], key=lambda x: x['f1_score'])
            if worst_dropout['num_dropped'] > 0:
                conclusions.append(f"5. Отбрасывание размерностей: при отбрасывании {worst_dropout['num_dropped']} размерностей F1-score изменился с {baseline_f1:.4f} до {worst_dropout['f1_score']:.4f}")
        
        if 'pca' in dim_exp and len(dim_exp['pca']) > 0:
            baseline_f1 = dim_exp['pca'][0]['f1_score']
            best_pca = max(dim_exp['pca'], key=lambda x: x['f1_score'])
            if best_pca['n_components'] != 100:
                conclusions.append(f"6. PCA: оптимальное количество компонент - {best_pca['n_components']} (F1-score: {best_pca['f1_score']:.4f})")
    
    for conclusion in conclusions:
        print(f"\n{conclusion}")


def main():
    """Главная функция для анализа результатов."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Анализ результатов экспериментов классификации')
    parser.add_argument('--results', type=str, default='../assets/results.json',
                       help='Путь к файлу с результатами')
    parser.add_argument('--output_dir', type=str, default='../assets/plots',
                       help='Директория для сохранения графиков')
    args = parser.parse_args()
    
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"ОШИБКА: Файл {results_path} не найден")
        print("Сначала выполните эксперименты: python source/main.py")
        return
    
    # Создаем директорию для графиков
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Загрузка результатов...")
    results = load_results(results_path)
    
    # Анализ результатов
    if 'svm_kernels' in results:
        analyze_svm_kernels(results['svm_kernels'], output_dir)
        
        # Анализ итераций для каждого kernel
        for kernel, kernel_results in results['svm_kernels'].items():
            analyze_iterations(kernel_results, f"SVM ({kernel})", output_dir)
    
    if 'mlp' in results:
        analyze_iterations(results['mlp'], "MLP", output_dir)
    
    if 'svm_kernels' in results and 'mlp' in results:
        compare_svm_mlp(results['svm_kernels'], results['mlp'], output_dir)
    
    if 'dimension_experiments' in results:
        analyze_dimensionality(results['dimension_experiments'], output_dir)
    
    # Генерируем итоговый отчет
    generate_summary_report(results)
    
    print("\n" + "="*70)
    print("Анализ завершен!")
    print(f"Все графики сохранены в директории: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()

