from sklearn.metrics.pairwise import cosine_similarity
from gensim.models import Word2Vec


def demonstrate_cosine_distances(model, examples):
    for target, similar, same_domain, different in examples:
        print(f"\n{'-' * 70}")
        print(f"ИСХОДНЫЙ ТОКЕН: {target.upper()}")
        print("-" * 70)

        if target not in model.wv:
            print(f"  ⚠️ Токен '{target}' отсутствует в словаре модели.")
            continue

        target_vec = model.wv[target].reshape(1, -1)
        results = []

        # Группы для сравнения
        groups = [similar, same_domain, different]

        for tokens in groups:
            group_results = []

            for token in tokens:
                if token in model.wv:
                    token_vec = model.wv[token].reshape(1, -1)
                    cos_sim = cosine_similarity(target_vec, token_vec)[0][0]
                    cos_dist = 1 - cos_sim
                    group_results.append((token, cos_dist, 'in_vocab'))
                else:
                    print(f"  • {token:15} | ❌ ТОКЕН ОТСУТСТВУЕТ В СЛОВАРЕ")
                    group_results.append((token, float('inf'), 'not_in_vocab'))

            results.extend([(token, dist) for token, dist, status in group_results if status == 'in_vocab'])

        # Ранжированный список всех расстояний
        if results:
            print(f"\nРАНЖИРОВАННЫЙ СПИСОК (ближайшие -> дальние):")
            sorted_results = sorted([(token, dist) for token, dist in results if dist < float('inf')],
                                    key=lambda x: x[1])

            for i, (token, dist) in enumerate(sorted_results[:10], 1):
                print(f"  {i}. {token:15} | расстояние: {dist:.4f}")


def main():
    model = Word2Vec.load('word2vec_model.model')

    examples = [
        (
            "Hurricane",
            ["typhoon", "cyclone", "storm"],
            ["wind", "rain", "flood", "disaster", "evacuation"],
            ["stock", "software", "Olympics", "judo", "Google"]
        ),
        (
            "Olympics",
            ["Games", "athletics", "competition"],
            ["medal", "swimming", "gymnastics", "judo", "relay"],
            ["kernel", "nanotech", "mortgage", "mutual fund", "dollar"]
        ),
        (
            "oil",
            ["petroleum", "crude", "gas"],
            ["pipeline", "price", "Venezuela", "export", "OPEC"],
            ["bacteria", "cycling", "email", "phishing", "dollar"]
        ),
        (
            "phishing",
            ["spam", "spyware", "malware"],
            ["email", "security", "virus", "firewall", "hacker"],
            ["basketball", "typhoon", "sprint", "medal", "tornado"]
        )

    ]

    demonstrate_cosine_distances(model, examples)


if __name__ == "__main__":
    main()
    