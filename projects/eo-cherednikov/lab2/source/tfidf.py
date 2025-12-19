import pickle
import random
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from utils import read_all_train_files


if __name__ == "__main__":
    print("Загрузка обучающей выборки...")
    corpus = read_all_train_files("assets/annotated-corpus/train")
    print(f"Загружено {len(corpus)} предложений")
    
    random.shuffle(corpus)

    print("Вычисление TF-IDF...")
    texts = [" ".join(sentence) for sentence in corpus]

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    os.makedirs('assets', exist_ok=True)

    with open('assets/tf-idf.pkl', 'wb') as file:
        pickle.dump((tfidf_matrix, feature_names), file)

    tfidf_matrix_max = tfidf_matrix.max(axis=0).todense().reshape(-1, 1)

    vocab = vectorizer.get_feature_names_out()
    term_weights = {}
    for term, index in zip(vocab, range(len(tfidf_matrix_max))):
        term_weights[term.lower()] = float(tfidf_matrix_max[index, 0])

    with open('assets/term_weights.pkl', 'wb') as file:
        pickle.dump(term_weights, file)
    
    print(f"Вычислено весов для {len(term_weights)} терминов")