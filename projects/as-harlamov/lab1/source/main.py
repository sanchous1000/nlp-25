import argparse
from pathlib import Path

import pandas as pd
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from tqdm import tqdm

from utils import _setup_nltk, tokenize, process_tokens


def main(dataset_path: str) -> None:
    # Инициализация NLTK при запуске
    _setup_nltk()

    # Инициализация NLP-инструментов
    stemmer = SnowballStemmer("english")
    lemmatizer = WordNetLemmatizer()

    # Чтение данных
    data = pd.read_csv(dataset_path, header=None, names=['class', 'title', 'text'])

    # Определение выходной директории
    dataset_name = Path(dataset_path).stem
    base_output_dir = Path('assets/annotated-corpus') / dataset_name

    for i, row in tqdm(data.iterrows(), total=len(data), desc="Processing documents"):
        try:
            tokenized_sentences = tokenize(row['text'])
            annotated_sentences = process_tokens(tokenized_sentences, stemmer, lemmatizer)

            output_path = base_output_dir / str(row['class']) / f'{i}.tsv'
            output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(output_path, "w", encoding="utf-8") as f:
                for sentence in annotated_sentences:
                    for tpl in sentence:
                        f.write('\t'.join(map(str, tpl)) + '\n')
                    f.write("\n")
        except Exception as e:
            print(f"Ошибка при обработке документа {i}: {e}")
            continue


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Аннотация текстового корпуса.")
    parser.add_argument('--dataset', type=str, required=True, help="Путь к CSV-датасету")
    args = parser.parse_args()
    main(args.dataset)
