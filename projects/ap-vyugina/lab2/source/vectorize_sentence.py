import pickle
import random

from gensim.models import Word2Vec
from utils import read_and_group, weighted_average_vector


if __name__ == "__main__":

    corpus = [
        *read_and_group("assets/annotated-corpus/train/1.tsv"),
        *read_and_group("assets/annotated-corpus/train/2.tsv"),
        *read_and_group("assets/annotated-corpus/train/3.tsv"),
        *read_and_group("assets/annotated-corpus/train/4.tsv"),
    ]
    random.shuffle(corpus)


    with open('assets/term_weights.pkl', 'rb') as file:
        term_weights = pickle.load(file)

    model = Word2Vec.load('assets/word2vec.model')

    test_sentence = corpus[0]
    weighted_avg = weighted_average_vector(test_sentence, model, term_weights)
    print(weighted_avg.shape)
        