from typing import List

from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity

from w2v_trainer import train_word2vec


def demo_semantic_similarity(model: Word2Vec, test_word: str = "doctor"):
    similar = ["physician", "surgeon"]
    related = ["hospital", "patient", "clinic"]
    unrelated = ["car", "computer", "ocean"]

    if test_word not in model.wv:
        print(f"Слово '{test_word}' отсутствует в модели.")
        return

    target_vec = model.wv[test_word].reshape(1, -1)

    def get_scores(words):
        scores = []
        for w in words:
            if w in model.wv:
                sim = float(cosine_similarity(target_vec, model.wv[w].reshape(1, -1))[0][0])
                scores.append((w, sim))
        return sorted(scores, key=lambda x: -x[1])

    print(f"Тестовое слово: {test_word}")
    print("  Близкие:", get_scores(similar))
    print("  Связанные:", get_scores(related))
    print("  Несвязанные:", get_scores(unrelated))


def run_hyperparam_experiments(sentences: List[List[str]]):
    configs = [
        {"name": "small_", "vector_size": 50, "window": 3, "sg": 1},
        {"name": "default", "vector_size": 100, "window": 5, "sg": 1},
        {"name": "large", "vector_size": 200, "window": 7, "sg": 1},
        {"name": "cbow", "vector_size": 100, "window": 5, "sg": 0},
    ]

    for config in configs:
        name = config.pop("name")
        print(f"\n--- Обучение: {name} ---")
        model = train_word2vec(sentences, **config)
        model.save(f"assets/models/w2v_{name}.model")
        demo_semantic_similarity(model, test_word="doctor")
