"""
Скрипт для построения графика изменения perplexity в зависимости от количества тем
и полиномиальной аппроксимации с выбором оптимальной степени полинома по r-squared
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from pathlib import Path


def load_experiment_results(results_file, max_iter=20):
    """
    Загружает результаты экспериментов LDA
    
    Args:
        results_file: Путь к файлу с результатами
        max_iter: Количество итераций для фильтрации (по умолчанию 20)
    """
    with open(results_file, 'r', encoding='utf-8') as f:
        results = json.load(f)
    
    n_topics_list = []
    perplexities = []
    
    filtered_results = {}
    for _, data in results.items():
        if data.get('max_iter') == max_iter:
            filtered_results[data['n_topics']] = data
    
    for n_topics in sorted(filtered_results.keys()):
        data = filtered_results[n_topics]
        perplexity = data['perplexity']
        n_topics_list.append(n_topics)
        perplexities.append(perplexity)
    
    return np.array(n_topics_list), np.array(perplexities)


def find_optimal_polynomial_degree(x, y, max_degree=10):
    """
    Находит оптимальную степень полинома на основе метрики r-squared.
    
    Args:
        x: Массив значений количества тем
        y: Массив значений perplexity
        max_degree: Максимальная степень полинома для проверки
        
    Returns:
        tuple: (оптимальная_степень, лучший_r2, коэффициенты_полинома)
    """
    best_degree = 1
    best_r2 = -np.inf
    best_coefs = None
    
    print("\nПоиск оптимальной степени полинома:")
    print("-" * 50)
    
    for degree in range(1, max_degree + 1):
        # Создаем полиномиальные признаки
        poly_features = PolynomialFeatures(degree=degree)
        x_poly = poly_features.fit_transform(x.reshape(-1, 1))
        
        model = LinearRegression()
        model.fit(x_poly, y)
        
        y_pred = model.predict(x_poly)

        r2 = r2_score(y, y_pred)
        
        print(f"Степень {degree}: R² = {r2:.6f}")
        
        if r2 > best_r2:
            best_r2 = r2
            best_degree = degree
            best_coefs = model.coef_
    
    print(f"\nОптимальная степень полинома: {best_degree} (R² = {best_r2:.6f})")
    
    return best_degree, best_r2, best_coefs


def plot_perplexity_with_polynomial_fit(n_topics, perplexities, optimal_degree, output_dir, max_iter):
    """
    Строит график perplexity и полиномиальную аппроксимацию.
    
    Args:
        n_topics: Массив количества тем
        perplexities: Массив значений perplexity
        optimal_degree: Оптимальная степень полинома
        output_dir: Директория для сохранения графика
    """
    poly_features = PolynomialFeatures(degree=optimal_degree)
    x_poly = poly_features.fit_transform(n_topics.reshape(-1, 1))
    
    model = LinearRegression()
    model.fit(x_poly, perplexities)
    
    x_smooth = np.linspace(n_topics.min(), n_topics.max(), 200)
    x_smooth_poly = poly_features.transform(x_smooth.reshape(-1, 1))
    y_smooth = model.predict(x_smooth_poly)
    
    plt.figure(figsize=(12, 8))
    
    plt.scatter(n_topics, 
        perplexities, 
        color='blue', 
        s=100, 
        zorder=3, 
        label='Экспериментальные данные',
        edgecolors='black',
        linewidth=1.5
    )
    
    # Полиномиальная аппроксимация
    plt.plot(x_smooth,
        y_smooth, 
        color='red', 
        linewidth=2.5, 
        label=f'Полиномиальная аппроксимация (степень {optimal_degree})',
        zorder=2
    )
    
    # Настройки графика
    plt.xlabel('Количество тем', fontsize=14, fontweight='bold')
    plt.ylabel('Perplexity', fontsize=14, fontweight='bold')
    plt.title('Зависимость Perplexity от количества тем в модели LDA', 
              fontsize=16, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(fontsize=12, loc='best', framealpha=0.9)
    
    # Добавляем аннотации с значениями
    for i, (x_val, y_val) in enumerate(zip(n_topics, perplexities)):
        plt.annotate(f'{y_val:.2f}', 
                    (x_val, y_val), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center',
                    fontsize=9,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    
    # Сохраняем график
    output_path = output_dir / f'perplexity_plot_{max_iter}.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nГрафик сохранен в {output_path}")
    
    plt.close()


if __name__ == '__main__':
    base_dir = Path(__file__).parent.parent
    source_dir = base_dir / 'source'
    output_dir = source_dir / 'output'
    
    results_file = output_dir / 'lda_experiments_results.json'
    
    if not results_file.exists():
        print(f"Ошибка: файл {results_file} не найден!")
        print("Сначала запустите lda_experiments.py для проведения экспериментов.")
        exit()
    
    for max_iter in [10, 20, 40]:
        print("=" * 60)
        print("Построение графика изменения perplexity")
        print("=" * 60)
        
        # Загружаем результаты для 20 итераций
        print(f"\nЗагрузка результатов из {results_file}...")
        print(f"Используются результаты для {max_iter} итераций")
        n_topics, perplexities = load_experiment_results(results_file, max_iter=max_iter)
        
        print(f"\nЗагружено {len(n_topics)} экспериментов:")
        for n, p in zip(n_topics, perplexities):
            print(f"  {n} тем: perplexity = {p:.4f}")

        print(f"Минимальная perplexity {min(perplexities)}")
        
        # Находим оптимальную степень полинома
        optimal_degree, best_r2, best_coefs = find_optimal_polynomial_degree(n_topics, perplexities)
        
        # Строим график
        print("\nПостроение графика...")
        plot_perplexity_with_polynomial_fit(n_topics, perplexities, optimal_degree, output_dir, max_iter)
        
        # Сохраняем информацию об аппроксимации
        poly_info = {
            'optimal_degree': int(optimal_degree),
            'r_squared': float(best_r2),
            'polynomial_coefficients': best_coefs.tolist() if best_coefs is not None else None
        }
        
        poly_info_file = output_dir / f'polynomial_approximation_info_{max_iter}.json'
        with open(poly_info_file, 'w', encoding='utf-8') as f:
            json.dump(poly_info, f, ensure_ascii=False, indent=2)