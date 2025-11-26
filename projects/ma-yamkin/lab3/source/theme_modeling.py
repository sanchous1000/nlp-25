import time
import json
import numpy as np
import pandas as pd
import pickle
import os
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import r2_score


def load_data():
    path = '../../assets/annotated-corpus/term_doc_matrix'

    loader = np.load(path+'_train/term_document_matrix.npz')
    td_matrix_train = csr_matrix(
        (loader['data'], loader['indices'], loader['indptr']),
        shape=loader['shape']
    )

    with open(path+'_train/term_to_index.pkl', 'rb') as f:
        term_to_index_train = pickle.load(f)

    index_to_term_train = {idx: term for term, idx in term_to_index_train.items()}

    if td_matrix_train.shape[0] == len(index_to_term_train):
        td_matrix_train = td_matrix_train.T

    column_names = ['class', 'title', 'text']
    df = pd.read_csv('../../train.csv', header=None, names=column_names)

    return td_matrix_train, index_to_term_train, len(df)


def get_top_words_per_topic(lda_model, feature_names, n_top_words=10):
    topics_words = []
    for topic_idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        topics_words.append(top_words)
    return topics_words


def save_document_topic_probs(doc_topic_probs, doc_ids, n_topics, out_dir='../lda_results'):
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f'doc_topic_probs_K{n_topics}.tsv')
    with open(filename, 'w', encoding='utf-8') as f:

        header = ['doc_id'] + [f'topic_{i}' for i in range(n_topics)]
        f.write('\t'.join(header) + '\n')

        for doc_id, probs in zip(doc_ids, doc_topic_probs):
            row = [str(doc_id)] + [f'{p:.6f}' for p in probs]
            f.write('\t'.join(row) + '\n')
    print(f"Сохранено: {filename}")


def save_top_docs_per_topic(doc_topic_dists, n_top_docs=1):
    top_docs = {}
    n_topics = doc_topic_dists.shape[1]

    for topic_idx in range(n_topics):
        # Получаем вероятности для текущей темы
        probs = doc_topic_dists[:, topic_idx]
        # Сортируем по убыванию
        top_indices = probs.argsort()[-n_top_docs:][::-1]
        top_docs[topic_idx] = [
            (i, float(probs[i])) for i in top_indices
        ]

    # Сохранение в файл
    os.makedirs('../lda_results', exist_ok=True)
    with open(f'../lda_results/top_docs_K{n_topics}.tsv', 'w', encoding='utf-8') as f:
        f.write("topic_id\tdoc_id\tprobability\n")
        for topic_id, docs in top_docs.items():
            for doc_id, prob in docs:
                f.write(f"{topic_id}\t{doc_id}\t{prob:.6f}\n")


def run_experiment(td_matrix, index_to_term, n_topics_list, len_train):
    results = []
    train_doc_ids = [i for i in range(len_train)]

    for n_topics in n_topics_list:
        print(f"\n=== Обучение LDA с K={n_topics} тем ===")

        lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=42,
            n_jobs=1,
            max_iter=100
        )

        td_matrix_train = td_matrix[:120000]
        td_matrix_test = td_matrix[120000:]

        # Замер времени обучения
        start_time = time.time()
        doc_topic_dists = lda.fit_transform(td_matrix_train)
        end_time = time.time()
        train_time = end_time - start_time

        # Топ-10 слов
        top_words = get_top_words_per_topic(lda, index_to_term, n_top_words=10)
        for i, words in enumerate(top_words):
            print(f"Тема {i}: {', '.join(words)}")

        # Сохранение топ-10 слов в JSON
        out_dir = '../lda_results'
        os.makedirs(out_dir, exist_ok=True)
        json_filename = os.path.join(out_dir, f'top_words_K{n_topics}.json')
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump({f'topic_{i}': words for i, words in enumerate(top_words)}, f, ensure_ascii=False, indent=4)
        print(f"Сохранено: {json_filename}")

        # Perplexity на тестовой выборке
        ppl = lda.perplexity(td_matrix_test)

        print(f"Perplexity на тесте: {ppl:.2f}")
        print(f"Время обучения: {train_time:.2f} секунд")

        # Сохранение вероятностей тем для train
        save_document_topic_probs(doc_topic_dists, train_doc_ids, n_topics)

        # Пример использования в цикле экспериментов:
        save_top_docs_per_topic(doc_topic_dists, n_top_docs=5)

        # Сохраняем результаты
        results.append({
            'n_topics': n_topics,
            'perplexity': ppl,
            'train_time': train_time,
            'model': lda,
            'top_words': top_words
        })

    return results


def build_graph(results):
    # Извлекаем данные для графика perplexity
    n_vals = np.array([r['n_topics'] for r in results])
    ppl_vals = np.array([r['perplexity'] for r in results])

    # Удаляем NaN, если есть
    valid_mask = ~np.isnan(ppl_vals)
    n_vals_clean = n_vals[valid_mask]
    ppl_vals_clean = ppl_vals[valid_mask]

    # Построение графика perplexity
    plt.figure(figsize=(10, 6))
    plt.plot(n_vals_clean, ppl_vals_clean, 'bo-', label='Perplexity')
    plt.xlabel('Количество тем (K)')
    plt.ylabel('Perplexity')
    plt.title('Зависимость Perplexity от количества тем в LDA')
    plt.grid(True)
    plt.legend()
    plt.savefig('../lda_results/perplexity_vs_topics.png', dpi=150)
    plt.show()

    # График времени обучения
    train_times = np.array([r['train_time'] for r in results])
    plt.figure(figsize=(10, 6))
    plt.plot(n_vals, train_times, 'go-', label='Время обучения (сек)')
    plt.xlabel('Количество тем (K)')
    plt.ylabel('Время обучения (сек)')
    plt.title('Зависимость времени обучения LDA от количества тем')
    plt.grid(True)
    plt.legend()
    plt.savefig('../lda_results/train_time_vs_topics.png', dpi=150)
    plt.show()

    # Полиномиальная аппроксимация для perplexity (остаётся без изменений)
    best_degree = 1
    best_r2 = -np.inf
    best_coefs = None

    for degree in range(1, 6):
        coeffs = np.polyfit(n_vals_clean, ppl_vals_clean, degree)
        ppl_pred = np.polyval(coeffs, n_vals_clean)
        r2 = r2_score(ppl_vals_clean, ppl_pred)
        print(f"Степень {degree}: R² = {r2:.4f}")
        if r2 > best_r2:
            best_r2 = r2
            best_degree = degree
            best_coefs = coeffs

    print(f"\nЛучшая степень полинома: {best_degree} (R² = {best_r2:.4f})")

    x_fine = np.linspace(n_vals_clean.min(), n_vals_clean.max(), 200)
    y_fine = np.polyval(best_coefs, x_fine)

    plt.figure(figsize=(10, 6))
    plt.plot(n_vals_clean, ppl_vals_clean, 'bo', label='Исходные данные')
    plt.plot(x_fine, y_fine, 'r--', label=f'Полином степени {best_degree} (R²={best_r2:.3f})')
    plt.xlabel('Количество тем (K)')
    plt.ylabel('Perplexity')
    plt.title('Полиномиальная аппроксимация зависимости Perplexity от K')
    plt.grid(True)
    plt.legend()
    plt.savefig('../lda_results/perplexity_approximation.png', dpi=150)
    plt.show()


if __name__ == '__main__':
    td_matrix_train, index_to_term_train, len_train = load_data()

    n_topics_list = [2, 4, 5, 10, 20, 40]

    results = run_experiment(td_matrix_train, index_to_term_train, n_topics_list, len_train)

    build_graph(results)
