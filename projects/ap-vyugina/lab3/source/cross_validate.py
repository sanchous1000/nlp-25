import pandas as pd
import numpy as np
from sklearn.svm import SVC

from utils import calculate_metrics_per_class
import random
from tqdm import tqdm
import matplotlib.pyplot as plt


TRAIN_SIZE = 10000
TEST_SIZE = 4000

def get_data(file_name):
    data = pd.read_csv(file_name, sep='\t', header=None).sample(frac=1)
    X = data.iloc[:, 1:].values.astype(np.float32)
    y = data.iloc[:, 0].astype(str).apply(lambda x: x.split('.')[0]).astype(int).values
    return X, y

def select_features(X, n_features):
    idxs2drop = random.sample(range(X.shape[1]), k=n_features)
    return np.delete(X, idxs2drop, axis=1), idxs2drop

num_features_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 15, 17, 20, 25]
metrics = {
    "precision": [],
    "recall": [],
    "f1": [],
    "accuracy": []
}

for num_features in tqdm(num_features_list):
    X, y = get_data('assets/tokenized_sentences_train.tsv')
    X, feature_idxs2drop = select_features(X, n_features=num_features)

    clf = SVC()
    clf.fit(X[:TRAIN_SIZE], y[:TRAIN_SIZE])

    X_test, y_test = get_data('assets/tokenized_sentences_test.tsv')
    X_test = np.delete(X_test, feature_idxs2drop, axis=1)

    y_pred = clf.predict(X_test[:TEST_SIZE])

    precision, recall, f1_score, accuracy = calculate_metrics_per_class(y_test[:TEST_SIZE], y_pred, 4)
    metrics["precision"].append(np.mean(precision))
    metrics["recall"].append(np.mean(recall))
    metrics["f1"].append(np.mean(f1_score))
    metrics["accuracy"].append(accuracy)


fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 3))
axes.plot(num_features_list, metrics["precision"], '-o', label="mean precision")
axes.plot(num_features_list, metrics["recall"], '-o', label="mean recall")
axes.plot(num_features_list, metrics["f1"], '-o', label="mean f1")
axes.plot(num_features_list, metrics["accuracy"], '-o', label="accuracy")
axes.legend()
axes.set_xlabel("Number of deleted features")

plt.tight_layout()
plt.savefig("assets/metrics-feature-drop.png", dpi=100)
