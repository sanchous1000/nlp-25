import random
from pathlib import Path
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

BASE_DIR = Path(__file__).parent.parent.parent
CORPUS_DIR = BASE_DIR / "assets" / "annotated-corpus"
OUTPUT_DIR = Path(__file__).parent.parent / "output"

TRAIN_SIZE = 10000
TEST_SIZE = 2000


def load_document(filepath: Path) -> str:
    tokens = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            if len(parts) >= 3:
                lemma = parts[2].lower()
                if len(lemma) > 2 and not lemma.isdigit():
                    tokens.append(lemma)
    return ' '.join(tokens)


def load_corpus(corpus_dir: Path, sample_size: int = None):
    documents = []
    doc_ids = []
    
    all_files = []
    for class_dir in corpus_dir.iterdir():
        if class_dir.is_dir():
            for f in class_dir.glob("*.tsv"):
                all_files.append((f, class_dir.name))
    
    if sample_size and len(all_files) > sample_size:
        random.seed(42)
        all_files = random.sample(all_files, sample_size)
    
    for filepath, class_name in all_files:
        text = load_document(filepath)
        if text:
            documents.append(text)
            doc_ids.append(filepath.stem)
    
    return documents, doc_ids


def calc_rsquared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


def run_lda_experiment(X_train, X_test, vectorizer, n_topics):
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=20,
        learning_method='batch'
    )
    
    lda.fit(X_train)
    
    perplexity = lda.perplexity(X_test)
    
    feature_names = vectorizer.get_feature_names_out()
    top_words = []
    for topic_idx, topic in enumerate(lda.components_):
        top_indices = topic.argsort()[:-11:-1]
        words = [feature_names[i] for i in top_indices]
        top_words.append(words)
    
    doc_topics = lda.transform(X_train)
    
    return {
        'n_topics': n_topics,
        'perplexity': perplexity,
        'top_words': top_words,
        'doc_topics': doc_topics,
        'model': lda
    }


def save_doc_topics(doc_topics, doc_ids, filepath):
    with open(filepath, 'w') as f:
        for doc_id, probs in zip(doc_ids, doc_topics):
            probs_str = '\t'.join(f"{p:.6f}" for p in probs)
            f.write(f"{doc_id}\t{probs_str}\n")


def main():
    print("Загрузка корпуса...")
    train_docs, train_ids = load_corpus(CORPUS_DIR / "train", TRAIN_SIZE)
    test_docs, test_ids = load_corpus(CORPUS_DIR / "test", TEST_SIZE)
    print(f"Train: {len(train_docs)}, Test: {len(test_docs)}")
    
    print("Построение term-document matrix...")
    vectorizer = CountVectorizer(
        max_features=5000, 
        min_df=5, 
        max_df=0.8,
        stop_words='english'
    )
    X_train = vectorizer.fit_transform(train_docs)
    X_test = vectorizer.transform(test_docs)
    print(f"Vocabulary: {len(vectorizer.get_feature_names_out())}")
    
    topic_counts = [2, 4, 5, 10, 20, 40]
    results = []
    
    print("\nЭксперименты с LDA:")
    for n_topics in topic_counts:
        print(f"\n--- {n_topics} topics ---")
        result = run_lda_experiment(X_train, X_test, vectorizer, n_topics)
        results.append(result)
        
        print(f"Perplexity: {result['perplexity']:.2f}")
        print("Top words per topic:")
        for i, words in enumerate(result['top_words']):
            print(f"  Topic {i}: {', '.join(words[:5])}")
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    best_result = min(results, key=lambda x: x['perplexity'])
    save_doc_topics(best_result['doc_topics'], train_ids, 
                    OUTPUT_DIR / f"doc_topics_{best_result['n_topics']}.tsv")
    
    x = np.array([r['n_topics'] for r in results])
    y = np.array([r['perplexity'] for r in results])
    
    plt.figure(figsize=(10, 6))
    plt.plot(x, y, 'bo-', label='Perplexity', markersize=8)
    
    best_degree = 2
    best_r2 = -np.inf
    for degree in [1, 2]:
        coeffs = np.polyfit(x, y, degree)
        y_pred = np.polyval(coeffs, x)
        r2 = calc_rsquared(y, y_pred)
        if r2 > best_r2:
            best_r2 = r2
            best_degree = degree
    
    coeffs = np.polyfit(x, y, best_degree)
    x_smooth = np.linspace(x.min(), x.max(), 100)
    y_smooth = np.polyval(coeffs, x_smooth)
    plt.plot(x_smooth, y_smooth, 'r--', label=f'Poly fit (deg={best_degree}, R²={best_r2:.4f})')
    
    plt.xlabel('Number of Topics')
    plt.ylabel('Perplexity')
    plt.title('LDA Perplexity vs Number of Topics')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(OUTPUT_DIR / 'perplexity_plot.png', dpi=150)
    print(f"\nГрафик сохранён: {OUTPUT_DIR / 'perplexity_plot.png'}")
    
    with open(OUTPUT_DIR / "results.txt", 'w') as f:
        f.write("LDA Topic Modeling Results\n\n")
        
        f.write("Perplexity by n_topics:\n")
        f.write("topics\tperplexity\n")
        for r in results:
            f.write(f"{r['n_topics']}\t{r['perplexity']:.2f}\n")
        
        f.write(f"\nPolynomial fit: degree={best_degree}, R²={best_r2:.4f}\n")
        f.write(f"Best n_topics (min perplexity): {best_result['n_topics']}\n\n")
        
        f.write("Top-10 words per topic:\n")
        for r in results:
            f.write(f"\n=== {r['n_topics']} topics ===\n")
            for i, words in enumerate(r['top_words']):
                f.write(f"Topic {i}: {', '.join(words)}\n")
    
    print(f"Результаты: {OUTPUT_DIR / 'results.txt'}")
    print(f"\nЛучшее количество тем: {best_result['n_topics']} (perplexity={best_result['perplexity']:.2f})")


if __name__ == "__main__":
    main()

