import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from metrics import calculate_metrics_per_class, classification_report
import time
import random
import matplotlib.pyplot as plt
# from sklearn.metrics import classification_report

def get_data(file_name, select_features=False, n_to_remove=0):
    data = pd.read_csv(file_name, sep='\t', header=None)
    X = data.iloc[:, 1:].values
    y = data.iloc[:, 0].values
    if select_features:
        idxs2drop = random.sample(range(X.shape[1]), k=n_to_remove)
        X = np.delete(X, idxs2drop, axis=1)
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


agg_precision, agg_recall, agg_f1, agg_accuracy = [], [], [], []
n_feature_to_remove = range(1, 16)
for n_to_remove in n_feature_to_remove:
    X_train, X_test, y_train, y_test = get_data('assets/embeddings_with_labels.tsv', select_features=True, n_to_remove=n_to_remove)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

# param_grid = {
#     'kernel': ['linear', 'rbf', 'poly'],
#     'max_iter': [2500, 5000, 7500],
#     'gamma': [0.001, 0.01, 0.1, 1],
#     'degree': [2, 3, 4, 5]
# }
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# svc = SVC()

# t1 = time.time()
# grid_search = GridSearchCV(svc, param_grid, cv=cv, scoring='accuracy')
# grid_search.fit(X_train, y_train)
# t2 = time.time()
# print(f"Время обучения: {(t2-t1):.2f} секунд")
# print(f"Лучшие параметры: {grid_search.best_params_}")
# print(f"Лучшая точность: {grid_search.best_score_:.2f}")
 
# clf = grid_search.best_estimator_

    clf = SVC(
        kernel='poly',
        degree=2,
        gamma=0.1,
        max_iter=5000
    )

    t1 = time.time()
    clf.fit(X_train, y_train)
    t2 = time.time()
    print(f'Время обучения: {(t2-t1):.2f} секунд')

    y_pred = clf.predict(X_test)
    # report из sklearn выдает те же метрики
    # print(classification_report(y_test, y_pred))
    precision, recall, f1, accuracy = calculate_metrics_per_class(y_test, y_pred)
    classification_report(precision, recall, f1, accuracy)
    agg_precision.append(np.mean(precision))
    agg_recall.append(np.mean(recall))
    agg_f1.append(np.mean(f1))
    agg_accuracy.append(np.mean(accuracy))

fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 3))
axes.plot(n_feature_to_remove, agg_precision, '-o', label="mean precision")
axes.plot(n_feature_to_remove, agg_recall, '-o', label="mean recall")
axes.plot(n_feature_to_remove, agg_f1, '-o', label="mean f1")
axes.plot(n_feature_to_remove, agg_accuracy, '-o', label="accuracy")
axes.legend(loc="lower left")
axes.set_xlabel("Число удаленных признаков")
plt.tight_layout()
plt.savefig("assets/metrics.png")