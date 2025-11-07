import random

from gensim.models import Word2Vec
from utils import read_and_group


if __name__ == "__main__":
    corpus = [
        *read_and_group("assets/annotated-corpus/train/1.tsv"),
        *read_and_group("assets/annotated-corpus/train/2.tsv"),
        *read_and_group("assets/annotated-corpus/train/3.tsv"),
        *read_and_group("assets/annotated-corpus/train/4.tsv"),
    ]
    random.shuffle(corpus)

    model = Word2Vec(
        corpus,
        vector_size=30,# Размер вектора
        window=5,      # Окно контекста
        min_count=1,   # Минимальное число повторений слова
        workers=4,     # Количество потоков
        sg=1           # Алгоритм Skip-Gram (sg=1); иначе CBOW (sg=0)
    )

    model.save("assets/word2vec.model")

    # тест
    similar_words = model.wv.most_similar('vote')
    for w, sim in similar_words:
        print(f"{w}: {sim:.4f}")
                        
       