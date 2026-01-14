import re
import numpy as np
from collections import Counter
from gensim.models import Word2Vec
from sklearn.decomposition import PCA
from datasets import load_dataset

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zа-я0-9\s]', ' ', text)
    tokens = text.split()
    return tokens

def build_vocab_and_term_doc_matrix(documents, min_freq=2):
    vocab_counter = Counter()
    tokenized_docs = []

    for doc in documents:
        tokens = preprocess_text(doc)
        tokenized_docs.append(tokens)
        vocab_counter.update(tokens)

    filtered_tokens = [w for w,c in vocab_counter.items() if c >= min_freq]

    vocab = {w:i for i,w in enumerate(filtered_tokens)}

    term_doc_matrix = np.zeros((len(documents), len(vocab)), dtype=float)

    for i, tokens in enumerate(tokenized_docs):
        counts = Counter(tokens)
        for token, count in counts.items():
            idx = vocab.get(token)
            if idx is not None:
                term_doc_matrix[i, idx] = count

    return vocab, term_doc_matrix, tokenized_docs

def compute_tfidf(term_doc_matrix):
    tf = term_doc_matrix / (term_doc_matrix.sum(axis=1, keepdims=True) + 1e-9)
    df = np.count_nonzero(term_doc_matrix, axis=0)
    idf = np.log((term_doc_matrix.shape[0]+1)/(df+1)) + 1
    tfidf = tf * idf
    return tfidf

def train_word2vec(tokenized_docs, vector_size=100, window=5, min_count=2):
    model = Word2Vec(sentences=tokenized_docs, vector_size=vector_size, window=window, min_count=min_count)
    return model

def document_vector(doc_tokens, w2v_model):
    vectors = [w2v_model.wv[token] for token in doc_tokens if token in w2v_model.wv]
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return np.zeros(w2v_model.vector_size)

def cosine_similarity(a, b):
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return np.dot(a, b) / (na * nb)

def save_embeddings_tsv(file_path, doc_ids, embeddings):
    with open(file_path, "w", encoding="utf-8") as f:
        for doc_id, emb in zip(doc_ids, embeddings):
            f.write(f"{doc_id}\t" + "\t".join(map(str, emb)) + "\n")

def reduce_dimensionality(embeddings, n_components=50):
    pca = PCA(n_components=n_components)
    reduced = pca.fit_transform(embeddings)
    return reduced

def vectorize_sentence(tokens, vocab, term_doc_matrix=None):
    vec = np.zeros(len(vocab))
    counts = Counter(tokens)
    for token, count in counts.items():
        if token in vocab:
            vec[vocab[token]] = count
    if term_doc_matrix is not None:
        tf = vec / (vec.sum() + 1e-9)
        df = np.count_nonzero(term_doc_matrix, axis=0)
        idf = np.log((term_doc_matrix.shape[0]+1)/(df+1)) + 1
        vec = tf * idf
    return vec

ds_train = load_dataset("ag_news", split="train[:10000]")
documents = [row["text"] for row in ds_train]
doc_ids = [f"{i:05d}" for i in range(len(documents))]

vocab, term_doc_matrix, tokenized_docs = build_vocab_and_term_doc_matrix(documents)
tfidf_matrix = compute_tfidf(term_doc_matrix)
w2v_model = train_word2vec(tokenized_docs)

w2v_doc_embeddings = np.array([document_vector(tokens, w2v_model) for tokens in tokenized_docs])
save_embeddings_tsv("ag_news_w2v_embeddings.tsv", doc_ids, w2v_doc_embeddings)

reduced_embeddings = reduce_dimensionality(w2v_doc_embeddings, n_components=50)
save_embeddings_tsv("ag_news_w2v_embeddings_pca.tsv", doc_ids, reduced_embeddings)

test_words = ["news", "market", "technology", "government"]
for word in test_words:
    print(f"Nearest neighbors for '{word}':")
    if word in w2v_model.wv:
        sims = w2v_model.wv.most_similar(word, topn=5)
        for w, score in sims:
            print(f"  {w}: {score:.4f}")

sentence_vectors = []
for tokens in tokenized_docs:
    vec = np.array([w2v_model.wv[t] for t in tokens if t in w2v_model.wv])
    if len(vec) > 0:
        sentence_vectors.append(np.mean(vec, axis=0))
    else:
        sentence_vectors.append(np.zeros(w2v_model.vector_size))
sentence_vectors = np.array(sentence_vectors)
save_embeddings_tsv("ag_news_sentence_vectors.tsv", doc_ids, sentence_vectors)
