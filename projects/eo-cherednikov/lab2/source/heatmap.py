import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
from gensim.models import Word2Vec
import os

os.makedirs('assets', exist_ok=True)

model = Word2Vec.load('assets/word2vec.model')
test_tokens = {
    'vote': {
        'similar': ['election', 'ballot', 'poll'],
        'same_domain': ['politics', 'democracy', 'government'],
        'different': ['animal', 'tree', 'sentence']
    },
    'study': {
        'similar': ['learn', 'research', 'examine'],
        'same_domain': ['university', 'professor', 'education'],
        'different': ['vote', 'animal', 'creation']
    }
}

def get_cosine_distance(word1, word2, model):
    try:
        vec1 = model.wv[word1].reshape(1, -1)
        vec2 = model.wv[word2].reshape(1, -1)
        distance = cosine_distances(vec1, vec2)[0][0]
        return distance
    except KeyError:
        return None

all_words = ['vote', 'election', 'politics', "study", "university", "professor", "animal", "park", "tree"]

available_words = [w for w in all_words if w in model.wv]
print(f"Доступные слова в модели: {available_words}")

if len(available_words) >= 2:
    vectors = [model.wv[word] for word in available_words]
    distances = cosine_distances(vectors)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(distances, annot=True, xticklabels=available_words, yticklabels=available_words, 
                cmap="YlGnBu", fmt='.3f')
    plt.title('Косинусные расстояния между словами')
    plt.tight_layout()
    plt.savefig('assets/heatmap.png', dpi=300, bbox_inches='tight')
    print("Тепловая карта сохранена в assets/heatmap.png")