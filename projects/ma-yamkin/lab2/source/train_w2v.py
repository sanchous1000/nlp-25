import os
import time
from tqdm import tqdm
from gensim.models import Word2Vec


def read_annotated_corpus():
    train_dir = os.path.join('..', 'assets', 'annotated-corpus', 'train')

    subdirs = ['1', '2', '3', '4']
    sentences = []

    # Обрабатываем каждую метку
    for subdir in subdirs:
        subdir_path = os.path.join(train_dir, subdir)

        tsv_files = [f for f in os.listdir(subdir_path) if f.endswith('.tsv')]

        for filename in tqdm(tsv_files, desc=f"Чтение файлов из train/{subdir}", leave=False):
            filepath = os.path.join(subdir_path, filename)

            with open(filepath, 'r', encoding='utf-8') as f:
                content = f.read().strip()

            if not content:
                continue

            # Разделение на предложения по двойному переносу строки
            raw_sentences = [s.strip() for s in content.split('\n\n') if s.strip()]

            # Обрабатываем отдельно каждое предложение
            for raw_sentence in raw_sentences:
                tokens = []
                lines = [l.strip() for l in raw_sentence.split('\n') if l.strip()]

                # Обрабатываем отдельно каждое слово
                for line in lines:
                    parts = [p.strip() for p in line.split('\t') if p.strip()]
                    if parts:
                        # Берем только токен слова
                        token = parts[0]
                        if token:
                            tokens.append(token)
                if tokens:
                    sentences.append(tokens)

    return sentences


def train_word2vec(sentences, vector_size=100, window=5, min_count=1, epochs=10):
    """Обучение модели Word2Vec"""

    model = Word2Vec(
        sentences=sentences,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        epochs=epochs,
        seed=42,
    )

    return model


def main():
    sentences = read_annotated_corpus()

    start_time = time.time()
    model = train_word2vec(
        sentences,
        vector_size=100,
        window=5,
        min_count=1,
        epochs=10
    )
    end_time = time.time()

    print(end_time - start_time)

    model.save('source/word2vec_model.model')


if __name__ == "__main__":
    main()
