from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

from main import TextVectorizer, TRAIN_DIR


def build_heatmap(vectorizer: TextVectorizer, words: list):
    n = len(words)
    matrix = np.zeros((n, n))
    
    for i, w1 in enumerate(words):
        for j, w2 in enumerate(words):
            if i == j:
                matrix[i][j] = 0
            else:
                matrix[i][j] = vectorizer.cosine_distance(w1, w2)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, xticklabels=words, yticklabels=words, 
                annot=True, fmt='.2f', cmap='RdYlGn_r')
    plt.title('Косинусное расстояние между словами')
    plt.tight_layout()
    
    out_path = Path(__file__).parent.parent / 'output' / 'heatmap.png'
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f'Сохранено: {out_path}')
    plt.show()


def main():
    vectorizer = TextVectorizer()
    
    docs, _ = vectorizer.load_corpus(TRAIN_DIR)
    sentences = vectorizer.prepare_sentences(docs)
    
    vectorizer.train_word2vec(sentences, vector_size=100, window=5, min_count=2, epochs=20)
    
    words = ['company', 'firm', 'market', 'president', 'government', 
             'game', 'team', 'oil', 'price', 'computer']
    
    available = [w for w in words if vectorizer.word_exists(w)]
    print(f'Слова для heatmap: {available}')
    
    build_heatmap(vectorizer, available)


if __name__ == '__main__':
    main()

