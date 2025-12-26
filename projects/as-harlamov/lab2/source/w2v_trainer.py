from typing import List

from gensim.models import Word2Vec


def train_word2vec(sentences: List[List[str]], vector_size=100, window=5, min_count=2, sg=1):
    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,  # размерность векторного пространства
        window=window,  # контекстное окно (сколько слов слева и справа от целевого учитывается)
        min_count=min_count,  # игнорирует слова с частотой меньше min_count
        sg=sg,  # архитектура модели (1 - skip-gram, 0 - continuous bag of words)
        epochs=10,
        workers=4,
    )
    return model
