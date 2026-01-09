import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
from gensim.models import Word2Vec


words = ['vote', 'election', 'politics', "study", "university", "professor", "animal", "park", "tree"]
model = Word2Vec.load('assets/word2vec.model') 

vectors = [model.wv[word] for word in words]
distances = cosine_distances(vectors) 

sns.heatmap(distances, annot=True, xticklabels=words, yticklabels=words, cmap="YlGnBu")
plt.title('Косинусные расстояния между словами')
plt.savefig('assets/heatmap.png', dpi=300, bbox_inches='tight')