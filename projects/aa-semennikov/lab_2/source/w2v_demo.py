from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_distances
import seaborn as sns
import matplotlib.pyplot as plt

words = ["wildlife", "nature", "tree", "animal", "law", "power", "government", "president", "study", "university"]

w2v = Word2Vec.load("assets/word2vec_model.bin")

vectors = [w2v.wv[word] for word in words]
cosine_distances = cosine_distances(vectors)

plt.figure(figsize=(8, 6))
sns.heatmap(cosine_distances, annot=True, xticklabels=words, yticklabels=words, cbar=False, cmap="crest")
plt.show()