import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import curve_fit
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import r2_score

from corpus_loader import load_corpus_from_annotated_dir, load_corpus_from_csv
from term_document_matrix import build_term_document_matrix
from utils import tokenize_and_lemmatize


def run_lda_experiment(
    train_matrix,
    test_matrix,
    n_topics: int,
    max_iter: int = 10,
    random_state: int = 42,
) -> Tuple[LatentDirichletAllocation, float, np.ndarray]:
    print(f"  Обучение LDA с {n_topics} темами...")

    if train_matrix.shape[0] == 0 or train_matrix.shape[1] == 0:
        raise ValueError(f"Обучающая матрица пуста: {train_matrix.shape}")
    if train_matrix.nnz == 0:
        raise ValueError("Обучающая матрица не содержит ненулевых элементов")

    if test_matrix.shape[0] == 0:
        raise ValueError(f"Тестовая матрица пуста: {test_matrix.shape}")

    print(f"    Размер обучающей матрицы: {train_matrix.shape}, ненулевых элементов: {train_matrix.nnz}")
    print(f"    Размер тестовой матрицы: {test_matrix.shape}, ненулевых элементов: {test_matrix.nnz}")

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=max_iter,
        random_state=random_state,
        n_jobs=-1,
        verbose=0,
        learning_method='online',
        learning_decay=0.7,
        doc_topic_prior=0.1,
        topic_word_prior=0.01,
    )

    lda.fit(train_matrix)
    perplexity = lda.perplexity(test_matrix)
    doc_topic_probs = lda.transform(train_matrix)

    return lda, perplexity, doc_topic_probs


def get_top_words(model: LatentDirichletAllocation, feature_names: List[str], n_words: int = 10) -> List[List[str]]:
    top_words_per_topic = []
    for topic_idx, topic in enumerate(model.components_):
        top_indices = topic.argsort()[-n_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        top_words_per_topic.append(top_words)
    return top_words_per_topic


def save_results(
    output_dir: Path,
    n_topics: int,
    top_words: List[List[str]],
    perplexity: float,
    doc_topic_probs: np.ndarray,
    max_iter: int,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    top_docs_per_topic = []
    for topic_idx in range(n_topics):
        topic_probs = doc_topic_probs[:, topic_idx]
        top_doc_indices = topic_probs.argsort()[-10:][::-1]
        top_docs_per_topic.append(
            {
                'topic': topic_idx,
                'top_documents': [int(idx) for idx in top_doc_indices],
                'probabilities': [float(topic_probs[idx]) for idx in top_doc_indices],
            },
        )

    results_data = {
        'n_topics': n_topics,
        'max_iter': max_iter,
        'perplexity': float(perplexity),
        'top_words': top_words,
        'top_documents': top_docs_per_topic,
    }

    results_file = output_dir / f"results_{n_topics}topics_iter{max_iter}.json"
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)

    probs_file = output_dir / f"doc_topic_probs_{n_topics}topics_iter{max_iter}.tsv"
    with open(probs_file, 'w', encoding='utf-8') as f:
        for doc_idx, probs in enumerate(doc_topic_probs):
            line = [str(doc_idx)] + [f"{p:.6f}" for p in probs]
            f.write("\t".join(line) + "\n")


def polynomial_func(x, *params):
    return sum(p * (x ** i) for i, p in enumerate(params))


def find_best_polynomial_degree(x: np.ndarray, y: np.ndarray, max_degree: int = 5, min_r2_improvement: float = 0.01) -> \
    Tuple[int, np.ndarray, float]:
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


def plot_perplexity(results: Dict[int, Dict], output_dir: Path, max_iter: int):
    import matplotlib.pyplot as plt

    n_topics_list = sorted(results.keys())
    perplexities = [results[n]['perplexity'] for n in n_topics_list]

    x = np.array(n_topics_list)
    y = np.array(perplexities)

    best_degree, best_params, best_r2 = find_best_polynomial_degree(x, y)
    print(f"\nВыбрана степень полинома: {best_degree} (R² = {best_r2:.4f})")

    x_smooth = np.linspace(x.min(), x.max(), 100)
    y_smooth = polynomial_func(x_smooth, *best_params)

    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'o-', label='Perplexity', linewidth=2, markersize=8)
    plt.plot(
        x_smooth,
        y_smooth,
        '--',
        label=f'Полиномиальная аппроксимация (степень {best_degree}, R²={best_r2:.4f})',
        linewidth=2,
    )
    plt.xlabel('Количество тем', fontsize=12)
    plt.ylabel('Perplexity', fontsize=12)
    plt.title(f'Зависимость Perplexity от количества тем (max_iter={max_iter})', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_file = output_dir / f"perplexity_plot_iter{max_iter}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"График сохранен: {output_file}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='LDA Topic Modeling Experiments')
    parser.add_argument(
        '--train_corpus', type=str, default='../../lab2/assets/annotated-corpus/train',
        help='Путь к обучающему корпусу (аннотированному)',
    )
    parser.add_argument(
        '--test_dataset', type=str, default='../../lab2/assets/source-corpus/test.csv',
        help='Путь к тестовому датасету (CSV)',
    )
    parser.add_argument(
        '--output', type=str, default='../assets',
        help='Директория для сохранения результатов',
    )
    parser.add_argument(
        '--n_classes', type=int, default=4,
        help='Количество классов в датасете',
    )
    parser.add_argument(
        '--max_iter', type=int, default=10,
        help='Количество итераций обучения LDA',
    )
    parser.add_argument(
        '--experiment_iterations', action='store_true',
        help='Провести эксперименты с разным количеством итераций',
    )
    parser.add_argument(
        '--n_topics', type=int, default=None,
        help='Количество тем для одного эксперимента (если указано, запускается только этот эксперимент)',
    )
    parser.add_argument(
        '--max_docs', type=int, default=None,
        help='Максимальное количество документов для загрузки (для ускорения обучения)',
    )
    args = parser.parse_args()

    script_dir = Path(__file__).parent.resolve()

    train_corpus_dir = (script_dir / args.train_corpus).resolve() if not Path(
        args.train_corpus,
    ).is_absolute() else Path(args.train_corpus)
    test_dataset_path = (script_dir / args.test_dataset).resolve() if not Path(
        args.test_dataset,
    ).is_absolute() else Path(args.test_dataset)
    output_dir = (script_dir / args.output).resolve() if not Path(args.output).is_absolute() else Path(args.output)

    print("Загрузка обучающего корпуса...")
    if args.max_docs:
        print(f"  Ограничение размера корпуса: максимум {args.max_docs} документов")
    train_documents = load_corpus_from_annotated_dir(train_corpus_dir, max_docs=args.max_docs)
    print(f"Загружено {len(train_documents)} документов")

    non_empty_train = [doc for doc in train_documents if doc]
    print(f"Непустых документов: {len(non_empty_train)}")
    if non_empty_train:
        print(f"Пример первого документа (первые 10 токенов): {non_empty_train[0][:10]}")

    print("\nПостроение матрицы термин-документ для обучающей выборки...")
    train_matrix, feature_names, vectorizer = build_term_document_matrix(
        train_documents,
        min_df=3,
        max_df=0.85,
    )
    print(f"Размерность матрицы: {train_matrix.shape}")
    print(f"Количество признаков: {len(feature_names)}")

    if train_matrix.shape[0] == 0 or train_matrix.shape[1] == 0:
        raise ValueError(f"Матрица обучающей выборки пуста: {train_matrix.shape}")

    if train_matrix.nnz == 0:
        raise ValueError("Матрица обучающей выборки не содержит ненулевых элементов")

    print("\nЗагрузка тестового датасета...")
    test_texts, test_classes = load_corpus_from_csv(test_dataset_path)
    print(f"Загружено {len(test_texts)} тестовых документов")

    print("\nТокенизация тестовых документов...")
    test_tokenized = [tokenize_and_lemmatize(text) for text in test_texts]

    print("\nПостроение матрицы термин-документ для тестовой выборки...")
    non_empty_test = [doc for doc in test_tokenized if doc]
    print(f"Непустых тестовых документов: {len(non_empty_test)} из {len(test_tokenized)}")

    test_matrix = vectorizer.transform(non_empty_test)
    print(f"Размерность тестовой матрицы: {test_matrix.shape}")

    if test_matrix.shape[0] == 0:
        raise ValueError("Тестовая матрица пуста после преобразования")

    if test_matrix.nnz == 0:
        print("Предупреждение: тестовая матрица не содержит терминов из словаря обучающей выборки")
        print("Это может привести к проблемам при вычислении perplexity")

    if args.n_topics:
        n_topics_list = [args.n_topics]
        print(f"\nРежим одного эксперимента: {args.n_topics} тем, {args.max_iter} итераций")
    else:
        n_topics_list = [2, 5, 10, 20, 40]
        if args.n_classes not in n_topics_list:
            n_topics_list.append(args.n_classes)
        n_topics_list.sort()

    if args.n_topics:
        iterations_list = [args.max_iter]
    elif args.experiment_iterations:
        iterations_list = [args.max_iter // 2, args.max_iter, args.max_iter * 2]
    else:
        iterations_list = [args.max_iter]

    all_results = {}

    for max_iter in iterations_list:
        print(f"\n{'=' * 60}")
        print(f"Эксперименты с max_iter={max_iter}")
        print(f"{'=' * 60}")

        results = {}

        for n_topics in n_topics_list:
            print(f"\nЭксперимент: {n_topics} тем")

            model, perplexity, doc_topic_probs = run_lda_experiment(
                train_matrix,
                test_matrix,
                n_topics=n_topics,
                max_iter=max_iter,
            )

            top_words = get_top_words(model, feature_names, n_words=10)

            print(f"  Perplexity: {perplexity:.4f}")
            print(f"  Топ-5 слов для первой темы: {', '.join(top_words[0][:5])}")

            save_results(
                output_dir,
                n_topics,
                top_words,
                perplexity,
                doc_topic_probs,
                max_iter,
            )

            results[n_topics] = {
                'perplexity': perplexity,
                'top_words': top_words,
                'doc_topic_probs': doc_topic_probs,
            }

        all_results[max_iter] = results

        if not args.n_topics:
            print(f"\nПостроение графика для max_iter={max_iter}...")
            plot_perplexity(results, output_dir, max_iter)

    print(f"\n{'=' * 60}")
    print("АНАЛИЗ РЕЗУЛЬТАТОВ")
    print(f"{'=' * 60}")

    for max_iter in iterations_list:
        results = all_results[max_iter]
        n_topics_list_sorted = sorted(results.keys())
        perplexities = [results[n]['perplexity'] for n in n_topics_list_sorted]

        best_idx = np.argmin(perplexities)
        best_n_topics = n_topics_list_sorted[best_idx]
        best_perplexity = perplexities[best_idx]

        print(f"\nДля max_iter={max_iter}:")
        print(f"  Оптимальное количество тем: {best_n_topics}")
        print(f"  Perplexity: {best_perplexity:.4f}")
        print(f"  Все значения perplexity:")
        for n, p in zip(n_topics_list_sorted, perplexities):
            print(f"    {n} тем: {p:.4f}")

    summary_file = output_dir / "summary.json"
    summary = {
        'n_classes': args.n_classes,
        'experiments': {},
    }
    for max_iter in iterations_list:
        summary['experiments'][max_iter] = {
            n_topics: {
                'perplexity': float(result['perplexity']),
            }
            for n_topics, result in all_results[max_iter].items()
        }

    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\nСводка результатов сохранена: {summary_file}")


if __name__ == "__main__":
    main()
