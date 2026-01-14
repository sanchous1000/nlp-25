import re
import numpy as np
from datasets import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

STOP_WORDS = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(t) for t in tokens if t not in STOP_WORDS and len(t) > 2]
    return " ".join(tokens)

ds_train = load_dataset("ag_news", split="train[:10000]")
documents = [preprocess_text(row["text"]) for row in ds_train]

vectorizer = CountVectorizer(min_df=5, max_df=0.9)
X = vectorizer.fit_transform(documents)

X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

def lda_experiment(X_train, X_test, n_topics, max_iter=10, random_state=42):
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=max_iter,
        learning_method="batch",
        random_state=random_state
    )
    lda.fit(X_train)
    feature_names = np.array(vectorizer.get_feature_names_out())
    top_words = [feature_names[topic.argsort()[::-1][:10]] for topic in lda.components_]
    perplexity = lda.perplexity(X_test)
    doc_topic_probs = lda.transform(X_train)
    top_docs_per_topic = {topic: doc_topic_probs[:, topic].argsort()[::-1][:5].tolist() for topic in range(n_topics)}
    return {
        "lda_model": lda,
        "top_words": top_words,
        "perplexity": perplexity,
        "doc_topic_probs": doc_topic_probs,
        "top_docs_per_topic": top_docs_per_topic
    }

num_topics_list = [2, 4, 5, 10, 20, 40]
iteration_values = [5, 10, 20]
all_results = {}

for max_iter in iteration_values:
    iter_results = []
    for n_topics in num_topics_list:
        res = lda_experiment(X_train, X_test, n_topics, max_iter=max_iter)
        iter_results.append(res)
        print(f"\nIter={max_iter} Topics={n_topics} Perplexity={res['perplexity']:.2f}")
        for i, words in enumerate(res["top_words"]):
            print(f"  Topic {i}: {', '.join(words)}")
    all_results[max_iter] = iter_results

plt.figure(figsize=(12, 6))
for max_iter in iteration_values:
    perplexities = [res["perplexity"] for res in all_results[max_iter]]
    coeffs = np.polyfit(num_topics_list, perplexities, 3)
    poly = np.poly1d(coeffs)
    x_fit = np.linspace(min(num_topics_list), max(num_topics_list), 100)
    y_fit = poly(x_fit)
    plt.plot(x_fit, y_fit, label=f"Iter={max_iter} Poly Fit")
    plt.scatter(num_topics_list, perplexities, label=f"Iter={max_iter} Points")
plt.xlabel("Number of Topics")
plt.ylabel("Perplexity")
plt.title("Perplexity vs Number of Topics (Polynomial Fit)")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(12, 6))
for max_iter in iteration_values:
    perplexities = [res["perplexity"] for res in all_results[max_iter]]
    plt.plot(num_topics_list, perplexities, marker='o', label=f"Iter={max_iter}")
plt.xlabel("Number of Topics")
plt.ylabel("Perplexity")
plt.title("Perplexity vs Number of Topics")
plt.legend()
plt.grid(True)
plt.show()

for max_iter in iteration_values:
    perplexities = [res["perplexity"] for res in all_results[max_iter]]
    coeffs = np.polyfit(num_topics_list, perplexities, 3)
    poly = np.poly1d(coeffs)
    y_pred = poly(num_topics_list)
    r2 = r2_score(perplexities, y_pred)
    print(f"Max_iter={max_iter} Polynomial R²={r2:.3f}")

np.savetxt("doc_topic_probs_5topics.tsv", all_results[10][2]["doc_topic_probs"], delimiter="\t")
