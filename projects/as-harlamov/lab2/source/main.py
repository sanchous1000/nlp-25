# main.py
import argparse
from pathlib import Path

import pandas as pd

from corpus_loader import load_corpus_from_annotated_dir
from semantic_demo import demo_semantic_similarity, run_hyperparam_experiments
from vectorizer import document_to_vector
from w2v_trainer import train_word2vec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_corpus', type=str, default='../assets/annotated-corpus/train')
    parser.add_argument('--test_dataset', type=str, required=True)
    parser.add_argument('--output', type=str, default='../assets/embeddings/test_vectors.tsv')
    parser.add_argument('--tune', action='store_true', help='Запустить эксперименты с гиперпараметрами')
    args = parser.parse_args()

    train_corpus_dir = Path(args.train_corpus)
    sentences = load_corpus_from_annotated_dir(train_corpus_dir)

    if args.tune:
        print("Запуск экспериментов с гиперпараметрами Word2Vec...")
        run_hyperparam_experiments(sentences)
        print("Эксперименты завершены. Модели сохранены.")
        return

    print("Обучение основной Word2Vec модели...")
    w2v_model = train_word2vec(
        sentences,
        vector_size=100,
        window=5,
        min_count=2,
        sg=1,
    )
    w2v_model.save("assets/models/w2v_trained.model")
    print("Модель сохранена как w2v_trained.model")

    print("\nДемонстрация семантической близости:")
    demo_semantic_similarity(w2v_model)

    print("\nВекторизация тестовой выборки...")
    test_df = pd.read_csv(args.test_dataset, header=None, names=['class', 'title', 'text'])

    vector_size = 100
    output_lines = []
    for i, row in test_df.iterrows():
        doc_id = str(i)
        vec = document_to_vector(row['text'], w2v_model, vector_size)
        line = [doc_id] + [f"{x:.6f}" for x in vec.tolist()]
        output_lines.append("\t".join(line))

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"Векторы сохранены в {output_path}")


if __name__ == "__main__":
    main()
