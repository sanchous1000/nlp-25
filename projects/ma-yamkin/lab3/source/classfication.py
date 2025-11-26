import numpy as np
import pandas as pd
import time
from sklearn.svm import SVC, LinearSVC
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


def load_embeddings(file_path):
    df = pd.read_csv(file_path, sep='\t', header=None)
    doc_ids = df.iloc[:, 0].values
    embeddings = df.iloc[:, 1:].values.astype(np.float32)
    return doc_ids, embeddings


def load_labels(file_path):
    column_names = ['class', 'title', 'text']
    df = pd.read_csv(file_path, header=None, names=column_names)
    return df['class'].to_dict()


def load_data():
    train_doc_ids, X_train = load_embeddings('../assets/annotated-corpus/train.tsv')
    test_doc_ids, X_test = load_embeddings('../assets/annotated-corpus/test.tsv')

    train_labels_dict = load_labels('../train.csv')
    test_labels_dict = load_labels('../test.csv')

    y_train = np.array([train_labels_dict[doc_id] for doc_id in train_doc_ids])
    y_test = np.array([test_labels_dict[doc_id] for doc_id in test_doc_ids])

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test


def calculate_metrics(y_true, y_pred):
    classes = np.unique(y_true)
    tp = np.zeros(len(classes))
    fp = np.zeros(len(classes))
    fn = np.zeros(len(classes))

    for i, cls in enumerate(classes):
        true_mask = (y_true == cls)
        pred_mask = (y_pred == cls)
        tp[i] = np.sum(true_mask & pred_mask)
        fp[i] = np.sum(~true_mask & pred_mask)
        fn[i] = np.sum(true_mask & ~pred_mask)

    precisions = np.array([tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0 for i in range(len(classes))])
    recalls = np.array([tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0 for i in range(len(classes))])
    f1s = np.array([2 * (precisions[i] * recalls[i]) / (precisions[i] + recalls[i])
                    if (precisions[i] + recalls[i]) > 0 else 0 for i in range(len(classes))])

    tp_total = np.sum(tp)
    fp_total = np.sum(fp)
    fn_total = np.sum(fn)
    micro_precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0
    micro_recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0
    micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) \
        if (micro_precision + micro_recall) > 0 else 0

    precision_macro = np.mean(precisions)
    recall_macro = np.mean(recalls)
    f1_macro = np.mean(f1s)
    accuracy = np.sum(y_true == y_pred) / len(y_true)

    return precision_macro, recall_macro, f1_macro, accuracy, micro_f1


def plot_metrics_comparison(results_list, filename):
    filename += '.png'

    metrics_names = ['accuracy', 'error rate', 'micro f1', 'macro f1', 'macro recall', 'macro precision']
    n_experiments = len(results_list)
    if n_experiments == 0:
        print("Нет данных для визуализации.")
        return

    metric_values = {metric: [] for metric in metrics_names}

    for res in results_list:
        metric_values['accuracy'].append(res['accuracy'])
        metric_values['error rate'].append(1.0 - res['accuracy'])
        metric_values['micro f1'].append(res.get('micro_f1', res['f1_score']))
        metric_values['macro f1'].append(res['f1_score'])
        metric_values['macro recall'].append(res['recall'])
        metric_values['macro precision'].append(res['precision'])

    x = np.arange(len(metrics_names))
    width = 0.8 / max(n_experiments, 1)
    colors = plt.cm.tab10(np.linspace(0, 1, min(n_experiments, 10)))

    fig, ax = plt.subplots(figsize=(14, 8))
    for i in range(n_experiments):
        offset = (i - n_experiments / 2) * width + width / 2
        values = [metric_values[metric][i] for metric in metrics_names]
        label = f"{results_list[i].get('kernel', 'exp')}@{results_list[i].get('max_iter', 'N/A')}"
        ax.bar(x + offset, values, width, label=label, color=colors[i], edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value', fontsize=12, fontweight='bold')
    ax.set_title('Сравнение метрик', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names, rotation=45, ha='right')
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(title='Experiment', loc='upper left', bbox_to_anchor=(1.02, 1.0))
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.show()


def run_svm_hyperparameter_search(
        X_train, X_test, y_train, y_test,
        kernels=['linear', 'rbf'],
        max_iters=[5, 10, 100],
        C=1.0,
        gamma='scale',
        degree=3
):
    best_iters_per_kernel = {}

    for kernel in kernels:
        print(f"\nЭксперименты для ядра: {kernel.upper()}")
        kernel_results = []

        for max_iter in max_iters:
            print(f"max_iter = {max_iter}")

            # Настройка модели
            if kernel == 'linear':
                model = LinearSVC(
                    C=C,
                    max_iter=max_iter,
                    dual=False,
                    random_state=42
                )
            else:
                model = SVC(
                    kernel=kernel,
                    C=C,
                    gamma=gamma,
                    degree=degree,
                    max_iter=max_iter if kernel != 'linear' else -1,
                    random_state=42
                )

            # Обучение
            start_time = time.time()
            model.fit(X_train, y_train)
            training_time = time.time() - start_time

            # Предсказание
            y_pred = model.predict(X_test)
            precision, recall, f1_macro, accuracy, micro_f1 = calculate_metrics(y_test, y_pred)

            result = {
                'kernel': kernel,
                'max_iter': max_iter,
                'C': C,
                'gamma': gamma,
                'degree': degree,
                'accuracy': accuracy,
                'precision': precision,
                'recall': recall,
                'f1_score': f1_macro,
                'micro_f1': micro_f1,
                'training_time': training_time
            }
            kernel_results.append(result)

        plot_metrics_comparison(kernel_results, 'metrics_comparison_'+kernel)

        best = max(kernel_results, key=lambda x: x['f1_score'])
        best_iters_per_kernel[kernel] = best['max_iter']
        print(f"Лучшее max_iter для {kernel}: {best['max_iter']} "
              f"(F1_macro={best['f1_score']:.4f})"
              f"(F1_micro={best['micro_f1']:.4f})"
              f"(recall={best['recall']:.4f})"
              f"(precision={best['precision']:.4f})"
              f"(accuracy={best['accuracy']:.4f})"
              f"training_time={best['training_time']}")

    return best_iters_per_kernel


def pca_experiment(
        X_train,
        y_train,
        X_test,
        y_test,
        optimal_max_iter,
        C=1.0,
        gamma='scale',
        degree=3
):
    for key, value in optimal_max_iter.items():
        print(f'ядро: {key}')

        original_dim = X_train.shape[1]
        dimensions = [10, 30, 60, 90, original_dim]  # Целевые размерности
        dim_results = []

        for dim in dimensions:
            if dim > original_dim:
                dim = original_dim

            # Применение PCA
            pca = PCA(n_components=dim, random_state=42)
            X_train_pca = pca.fit_transform(X_train)
            X_test_pca = pca.transform(X_test)

            # Настройка модели
            if key == 'linear':
                model = LinearSVC(
                    C=C,
                    max_iter=value,
                    dual=False,
                    random_state=42
                )
            else:
                model = SVC(
                    kernel=key,
                    C=C,
                    gamma=gamma,
                    degree=degree,
                    max_iter=value if key != 'linear' else -1,
                    random_state=42
                )

            start_time = time.time()
            model.fit(X_train_pca, y_train)
            training_time = time.time() - start_time

            # Предсказание и расчёт метрик
            y_pred = model.predict(X_test_pca)
            precision, recall, f1, accuracy, f1_micro = calculate_metrics(y_test, y_pred)

            # Сохранение результатов
            dim_results.append({
                'dimension': dim,
                'precision': precision,
                'recall': recall,
                'f1_macro': f1,
                'f1_micro': f1_micro,
                'accuracy': accuracy,
                'explained_variance': np.sum(pca.explained_variance_ratio_),
                'training_time': training_time
            })

            print(
                f"Dimension={dim}: F1={f1:.4f}, Accuracy={accuracy:.4f}, Explained Var={np.sum(pca.explained_variance_ratio_):.4f}")

        # Визуализация зависимости метрик от размерности
        dims = [res['dimension'] for res in dim_results]
        f1_scores = [res['f1_macro'] for res in dim_results]
        accuracies = [res['accuracy'] for res in dim_results]
        explained_vars = [res['explained_variance'] for res in dim_results]

        plt.figure(figsize=(12, 6))
        plt.plot(dims, f1_scores, 'o-', label='F1-score')
        plt.plot(dims, accuracies, 's--', label='Accuracy')
        plt.plot(dims, explained_vars, 'd-.', label='Explained Variance')
        plt.xscale('log')
        plt.xlabel('Размерность')
        plt.ylabel('Значение метрик')
        plt.title(f'Зависимость метрик от размерности признакового пространства {key}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'dimension_vs_metrics_{key}.png', dpi=150)
        plt.show()


def run_pipeline():
    X_train, y_train, X_test, y_test = load_data()

    optimal_max_iter = run_svm_hyperparameter_search(
        X_train, X_test, y_train, y_test
    )

    pca_experiment(X_train, y_train, X_test, y_test, optimal_max_iter)


if __name__ == '__main__':
    run_pipeline()

