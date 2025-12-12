import json
from pathlib import Path
from collections import Counter
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import r2_score


def load_results(results_dir: Path) -> Dict:
    results_by_iter = {}
    
    for results_file in sorted(results_dir.glob("results_*topics*.json")):
        with open(results_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            n_topics = data['n_topics']
            max_iter = data['max_iter']
            
            if max_iter not in results_by_iter:
                results_by_iter[max_iter] = {}
            
            results_by_iter[max_iter][n_topics] = data
    
    return results_by_iter


def polynomial_func(x, *params):
    return sum(p * (x ** i) for i, p in enumerate(params))


def find_best_polynomial_degree(x: np.ndarray, y: np.ndarray, max_degree: int = 5, min_r2_improvement: float = 0.01) -> tuple:
    results = []
    
    for degree in range(1, max_degree + 1):
        try:
            initial_params = [1.0] * (degree + 1)
            params, _ = curve_fit(polynomial_func, x, y, p0=initial_params, maxfev=10000)
            y_pred = polynomial_func(x, *params)
            r2 = r2_score(y, y_pred)
            results.append((degree, params, r2))
        except:
            continue
    
    if not results:
        return 1, None, 0.0
    
    results.sort(key=lambda x: x[0])
    best_degree, best_params, best_r2 = results[0]
    
    for i in range(1, len(results)):
        degree, params, r2 = results[i]
        r2_improvement = r2 - best_r2
        
        if r2_improvement > min_r2_improvement:
            best_degree, best_params, best_r2 = degree, params, r2
        else:
            break
    
    return best_degree, best_params, best_r2


def plot_perplexity_analysis(results_by_iter: Dict, output_dir: Path):
    fig, ax = plt.subplots(figsize=(14, 8))
    
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E']
    markers = ['o', 's', '^', 'D', 'v']
    
    all_best_results = {}
    
    for idx, (max_iter, results) in enumerate(sorted(results_by_iter.items())):
        n_topics_list = sorted(results.keys())
        perplexities = [results[n]['perplexity'] for n in n_topics_list]
        
        x = np.array(n_topics_list)
        y = np.array(perplexities)
        
        best_degree, best_params, best_r2 = find_best_polynomial_degree(x, y)
        x_smooth = np.linspace(x.min(), x.max(), 100)
        y_smooth = polynomial_func(x_smooth, *best_params)
        
        best_idx = np.argmin(perplexities)
        best_n_topics = n_topics_list[best_idx]
        best_perplexity = perplexities[best_idx]
        all_best_results[max_iter] = (best_n_topics, best_perplexity)
        
        color = colors[idx % len(colors)]
        marker = markers[idx % len(markers)]
        
        ax.plot(x, y, marker=marker, linestyle='-', label=f'max_iter={max_iter}', 
                linewidth=2.5, markersize=10, color=color)
        ax.plot(x_smooth, y_smooth, '--', linewidth=1.5, color=color, alpha=0.5)
        ax.plot(best_n_topics, best_perplexity, marker='*', markersize=20, 
                color=color, markeredgecolor='black', markeredgewidth=1)
    
    ax.set_xlabel('Количество тем', fontsize=14, fontweight='bold')
    ax.set_ylabel('Perplexity', fontsize=14, fontweight='bold')
    ax.set_title('Зависимость Perplexity от количества тем для разных max_iter', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=11, loc='best')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    all_n_topics = set()
    for results in results_by_iter.values():
        all_n_topics.update(results.keys())
    ax.set_xticks(sorted(all_n_topics))
    
    plt.tight_layout()
    output_file = output_dir / "perplexity_analysis_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Сравнительный график perplexity сохранен: {output_file}")
    plt.close()
    
    return all_best_results


def plot_iterations_analysis(results_by_iter: Dict, output_dir: Path):
    fig, ax = plt.subplots(figsize=(12, 7))
    
    all_n_topics = set()
    for results in results_by_iter.values():
        all_n_topics.update(results.keys())
    n_topics_list = sorted(all_n_topics)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(n_topics_list)))
    markers = ['o', 's', '^', 'D', 'v', 'p']
    
    for idx, n_topics in enumerate(n_topics_list):
        iterations = []
        perplexities = []
        
        for max_iter in sorted(results_by_iter.keys()):
            if n_topics in results_by_iter[max_iter]:
                iterations.append(max_iter)
                perplexities.append(results_by_iter[max_iter][n_topics]['perplexity'])
        
        if iterations:
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            ax.plot(iterations, perplexities, marker=marker, linestyle='-', 
                   label=f'{n_topics} тем', linewidth=2.5, markersize=10, color=color)
    
    ax.set_xlabel('Количество итераций (max_iter)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Perplexity', fontsize=14, fontweight='bold')
    ax.set_title('Зависимость Perplexity от количества итераций для разных количеств тем', 
                 fontsize=16, fontweight='bold')
    ax.legend(fontsize=10, loc='best', ncol=2)
    ax.grid(True, alpha=0.3, linestyle='--')
    
    all_iterations = sorted(results_by_iter.keys())
    ax.set_xticks(all_iterations)
    
    plt.tight_layout()
    output_file = output_dir / "perplexity_vs_iterations.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"График perplexity vs итерации сохранен: {output_file}")
    plt.close()


def analyze_top_words(results_by_iter: Dict, output_dir: Path):
    all_words = Counter()
    words_by_topic_count = {}
    
    for max_iter, results in results_by_iter.items():
        for n_topics, data in results.items():
            if n_topics not in words_by_topic_count:
                words_by_topic_count[n_topics] = {}
            for topic_idx, topic_words in enumerate(data['top_words']):
                words_by_topic_count[n_topics][topic_idx] = topic_words
                all_words.update(topic_words)
    
    top_20_words = [word for word, count in all_words.most_common(20)]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    word_counts = [count for word, count in all_words.most_common(20)]
    ax.barh(range(len(top_20_words)), word_counts, color='#F18F01')
    ax.set_yticks(range(len(top_20_words)))
    ax.set_yticklabels(top_20_words, fontsize=10)
    ax.set_xlabel('Частота появления в топ-10 тем', fontsize=12, fontweight='bold')
    ax.set_title('Топ-20 слов, наиболее часто встречающихся в топ-10 тем', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    output_file = output_dir / "top_words_frequency.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"График частоты слов сохранен: {output_file}")
    plt.close()
    
    diversity_by_n_topics = {}
    all_n_topics = set()
    for results in results_by_iter.values():
        all_n_topics.update(results.keys())
    
    for n_topics in sorted(all_n_topics):
        all_unique_words = set()
        for max_iter, results in results_by_iter.items():
            if n_topics in results:
                for topic_words in results[n_topics]['top_words']:
                    all_unique_words.update(topic_words)
        diversity_by_n_topics[n_topics] = len(all_unique_words)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    n_topics_list = sorted(diversity_by_n_topics.keys())
    diversities = [diversity_by_n_topics[n] for n in n_topics_list]
    
    ax.plot(n_topics_list, diversities, 'o-', linewidth=2.5, markersize=10, color='#C73E1D')
    ax.set_xlabel('Количество тем', fontsize=12, fontweight='bold')
    ax.set_ylabel('Количество уникальных слов в топ-10', fontsize=12, fontweight='bold')
    ax.set_title('Разнообразие слов в зависимости от количества тем', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.set_xticks(n_topics_list)
    
    plt.tight_layout()
    output_file = output_dir / "word_diversity.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"График разнообразия слов сохранен: {output_file}")
    plt.close()


def create_topic_visualization(results_by_iter: Dict, n_topics: int, max_iter: int, output_dir: Path):
    if max_iter not in results_by_iter or n_topics not in results_by_iter[max_iter]:
        print(f"Результаты для {n_topics} тем и max_iter={max_iter} не найдены")
        return
    
    data = results_by_iter[max_iter][n_topics]
    top_words = data['top_words']
    
    n_cols = min(3, n_topics)
    n_rows = (n_topics + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    if n_topics == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    for topic_idx, topic_words in enumerate(top_words):
        ax = axes[topic_idx]
        words = topic_words[:10]
        y_pos = np.arange(len(words))
        ax.barh(y_pos, range(len(words), 0, -1), color=plt.cm.viridis(np.linspace(0, 1, len(words))))
        ax.set_yticks(y_pos)
        ax.set_yticklabels(words, fontsize=9)
        ax.set_xlabel('Ранг', fontsize=9)
        ax.set_title(f'Тема {topic_idx + 1}', fontsize=11, fontweight='bold')
        ax.invert_yaxis()
        ax.set_xticks([])
    
    for idx in range(n_topics, len(axes)):
        axes[idx].axis('off')
    
    plt.suptitle(f'Топ-10 ключевых слов для каждой темы (n_topics={n_topics}, max_iter={max_iter})', 
                 fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    output_file = output_dir / f"topics_visualization_{n_topics}topics_iter{max_iter}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Визуализация тем сохранена: {output_file}")
    plt.close()


def generate_report(results_by_iter: Dict, output_dir: Path):
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("АНАЛИЗ РЕЗУЛЬТАТОВ ТЕМАТИЧЕСКОГО МОДЕЛИРОВАНИЯ")
    report_lines.append("=" * 80)
    report_lines.append("")
    
    all_n_topics = set()
    for results in results_by_iter.values():
        all_n_topics.update(results.keys())
    n_topics_list = sorted(all_n_topics)
    iterations_list = sorted(results_by_iter.keys())
    
    report_lines.append("АНАЛИЗ PERPLEXITY:")
    report_lines.append("-" * 80)
    
    best_overall = None
    best_perplexity_overall = float('inf')
    
    for max_iter in iterations_list:
        results = results_by_iter[max_iter]
        report_lines.append(f"\nmax_iter = {max_iter}:")
        perplexities = [results[n]['perplexity'] for n in n_topics_list if n in results]
        topics_for_iter = [n for n in n_topics_list if n in results]
        
        if perplexities:
            best_idx = np.argmin(perplexities)
            best_n_topics = topics_for_iter[best_idx]
            best_perplexity = perplexities[best_idx]
            
            if best_perplexity < best_perplexity_overall:
                best_perplexity_overall = best_perplexity
                best_overall = (max_iter, best_n_topics, best_perplexity)
            
            report_lines.append(f"  Оптимальное количество тем: {best_n_topics} (perplexity={best_perplexity:.2f})")
            report_lines.append("  Perplexity по количеству тем:")
            for n, p in zip(topics_for_iter, perplexities):
                marker = " <-- ОПТИМАЛЬНО" if n == best_n_topics else ""
                report_lines.append(f"    {n:2d} тем: {p:12.2f}{marker}")
    
    if best_overall:
        max_iter_best, n_topics_best, perplexity_best = best_overall
        report_lines.append("")
        report_lines.append(f"ОБЩИЙ ОПТИМУМ:")
        report_lines.append(f"  max_iter={max_iter_best}, n_topics={n_topics_best}, perplexity={perplexity_best:.2f}")
    
    report_lines.append("")
    report_lines.append("АНАЛИЗ ВЛИЯНИЯ КОЛИЧЕСТВА ИТЕРАЦИЙ:")
    report_lines.append("-" * 80)
    
    for n_topics in n_topics_list:
        report_lines.append(f"\n{n_topics} тем:")
        for max_iter in iterations_list:
            if max_iter in results_by_iter and n_topics in results_by_iter[max_iter]:
                p = results_by_iter[max_iter][n_topics]['perplexity']
                report_lines.append(f"  max_iter={max_iter:2d}: perplexity={p:12.2f}")
    
    report_lines.append("")
    report_lines.append("АНАЛИЗ ТОП-СЛОВ:")
    report_lines.append("-" * 80)
    
    all_words = Counter()
    for results in results_by_iter.values():
        for data in results.values():
            for topic_words in data['top_words']:
                all_words.update(topic_words)
    
    report_lines.append("Топ-15 самых частых слов во всех темах:")
    for idx, (word, count) in enumerate(all_words.most_common(15), 1):
        report_lines.append(f"  {idx:2d}. {word:15s} - {count:3d} раз")
    report_lines.append("")
    
    report_lines.append("Разнообразие слов (количество уникальных слов в топ-10):")
    for n_topics in n_topics_list:
        all_unique_words = set()
        for max_iter, results in results_by_iter.items():
            if n_topics in results:
                for topic_words in results[n_topics]['top_words']:
                    all_unique_words.update(topic_words)
        report_lines.append(f"  {n_topics:2d} тем: {len(all_unique_words):4d} уникальных слов")
    report_lines.append("")
    
    if best_overall:
        max_iter_best, n_topics_best, _ = best_overall
        report_lines.append(f"ПРИМЕРЫ ТЕМ ДЛЯ ОПТИМАЛЬНОЙ КОНФИГУРАЦИИ (max_iter={max_iter_best}, n_topics={n_topics_best}):")
        report_lines.append("-" * 80)
        for topic_idx, topic_words in enumerate(results_by_iter[max_iter_best][n_topics_best]['top_words']):
            report_lines.append(f"Тема {topic_idx + 1}: {', '.join(topic_words[:10])}")
        report_lines.append("")
    
    report_lines.append("=" * 80)
    
    report_text = "\n".join(report_lines)
    
    report_file = output_dir / "analysis_report.txt"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("\n" + report_text)
    print(f"\nОтчет сохранен: {report_file}")


def main():
    script_dir = Path(__file__).parent.resolve()
    assets_dir = script_dir.parent / "assets"
    
    if not assets_dir.exists():
        print(f"Директория с результатами не найдена: {assets_dir}")
        return
    
    print("Загрузка результатов...")
    results_by_iter = load_results(assets_dir)
    
    if not results_by_iter:
        print("Результаты не найдены!")
        return
    
    iterations_list = sorted(results_by_iter.keys())
    all_n_topics = set()
    for results in results_by_iter.values():
        all_n_topics.update(results.keys())
    
    print(f"Загружено результатов для {len(iterations_list)} значений max_iter: {iterations_list}")
    print(f"Количества тем: {sorted(all_n_topics)}")
    
    print("\nПостроение графиков...")
    all_best_results = plot_perplexity_analysis(results_by_iter, assets_dir)
    plot_iterations_analysis(results_by_iter, assets_dir)
    analyze_top_words(results_by_iter, assets_dir)
    
    print("\nСоздание визуализаций тем...")
    for max_iter in iterations_list:
        for n_topics in [2, 4, 5, 10, 20, 40]:
            if max_iter in results_by_iter and n_topics in results_by_iter[max_iter]:
                create_topic_visualization(results_by_iter, n_topics, max_iter, assets_dir)
    
    print("\nГенерация отчета...")
    generate_report(results_by_iter, assets_dir)
    
    print("\n" + "=" * 80)
    print("АНАЛИЗ ЗАВЕРШЕН")
    print("=" * 80)
    
    print("\nКраткая сводка оптимальных конфигураций:")
    for max_iter, (best_n_topics, best_perplexity) in sorted(all_best_results.items()):
        print(f"  max_iter={max_iter:2d}: оптимально {best_n_topics} тем (perplexity={best_perplexity:.2f})")


if __name__ == "__main__":
    main()

