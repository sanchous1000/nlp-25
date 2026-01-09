import pandas as pd
import numpy as np
from sklearn.svm import SVC

from utils import calculate_metrics_per_class, classification_report
import time


TRAIN_SIZE = 10000
TEST_SIZE = 4000


def get_data(file_name):
    data = pd.read_csv(file_name, sep='\t', header=None).sample(frac=1)
    X = data.iloc[:, 1:].values.astype(np.float32)
    y = data.iloc[:, 0].astype(str).apply(lambda x: x.split('.')[0]).astype(int).values
    return X, y


X, y = get_data('assets/tokenized_sentences_train.tsv')

t1 = time.time()
clf = SVC(kernel="rbf")
clf.fit(X[:TRAIN_SIZE], y[:TRAIN_SIZE])
t2 = time.time()
print(f"Time: {(t2-t1):.3f} s")

X_test, y_test = get_data('assets/tokenized_sentences_test.tsv')
y_pred = clf.predict(X_test[:TEST_SIZE])

precision, recall, f1_score, accuracy = calculate_metrics_per_class(y_test[:TEST_SIZE], y_pred, 4)
classification_report(precision, recall, f1_score, accuracy)
