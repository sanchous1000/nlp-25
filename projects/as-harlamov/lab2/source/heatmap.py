from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
from gensim.models import Word2Vec
import seaborn as sns


words = ['python', 'doctor', 'patient', 'snake', 'dog', 'cat', 'milk', 'cow', 'dish', 'restaurant']
model = Word2Vec.load('../assets/models/w2v_large.model')

vectors = [model.wv[word] for word in words]
distances = cosine_distances(vectors)

sns.heatmap(distances, annot=True, xticklabels=words, yticklabels=words, cmap="YlGnBu")
plt.title('Косинусные расстояния между словами')
plt.savefig('../assets/heatmap.png', dpi=300, bbox_inches='tight')