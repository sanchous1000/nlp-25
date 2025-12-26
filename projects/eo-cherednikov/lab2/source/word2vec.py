import random

from gensim.models import Word2Vec
from utils import read_all_train_files


if __name__ == "__main__":
    corpus = read_all_train_files("assets/annotated-corpus/train")
    print(f"Загружено {len(corpus)} предложений")
    
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

    similar_words = model.wv.most_similar('vote')
    for w, sim in similar_words:
        print(f"{w}: {sim:.4f}")