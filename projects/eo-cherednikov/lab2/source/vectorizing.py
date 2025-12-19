import pickle
import random

from gensim.models import Word2Vec
from utils import read_all_train_files, weighted_average_vector, vectorize_document
from pathlib import Path


if __name__ == "__main__":
    with open('assets/term_weights.pkl', 'rb') as file:
        term_weights = pickle.load(file)

    model = Word2Vec.load('assets/word2vec.model')

    print("\nТестирование векторизации одного предложения:")
    corpus = read_all_train_files("assets/annotated-corpus/train")
    random.shuffle(corpus)
    
    test_sentence = corpus[0]
    print(f"Тестовое предложение: {' '.join(test_sentence[:10])}...")
    weighted_avg = weighted_average_vector(test_sentence, model, term_weights)
    print(f"Размерность вектора предложения: {weighted_avg.shape}")

    print("\nТестирование полной векторизации документа:")
    train_dir = Path("assets/annotated-corpus/train")
    test_file = None
    for subdir in ['0', '1', '2', '3']:
        subdir_path = train_dir / subdir
        if subdir_path.exists():
            files = list(subdir_path.glob("*.tsv"))
            if files:
                test_file = files[0]
                break

    if test_file:
        print(f"Тестовый файл: {test_file}")
        doc_vector = vectorize_document(str(test_file), model, term_weights)
        print(f"Размерность вектора документа: {doc_vector.shape}")
        print(f"Первые 5 компонентов: {doc_vector[:5]}")