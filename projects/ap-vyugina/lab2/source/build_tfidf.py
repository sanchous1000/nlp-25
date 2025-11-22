import pickle
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from utils import read_and_group


if __name__ == "__main__":

    corpus = [
        *read_and_group("assets/annotated-corpus/train/1.tsv"),
        *read_and_group("assets/annotated-corpus/train/2.tsv"),
        *read_and_group("assets/annotated-corpus/train/3.tsv"),
        *read_and_group("assets/annotated-corpus/train/4.tsv"),
    ]
    random.shuffle(corpus)


    texts = [" ".join(sentence) for sentence in corpus]

    vectorizer = TfidfVectorizer()

    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    
    with open('assets/tf-idf.pkl', 'wb') as file:
        pickle.dump((tfidf_matrix, feature_names), file)

    tfidf_matrix = tfidf_matrix.max(axis=0).todense().reshape(-1, 1)

    vocab = vectorizer.get_feature_names_out()
    term_weights = {}
    for term, index in zip(vocab, range(len(tfidf_matrix))):
        term_weights[term] = tfidf_matrix[index]

    with open('assets/term_weights.pkl', 'wb') as file:
        pickle.dump(term_weights, file)
    
