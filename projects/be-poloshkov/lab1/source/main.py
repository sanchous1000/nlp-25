import ssl
from pathlib import Path

import pandas as pd
from nltk.stem import SnowballStemmer, WordNetLemmatizer
import nltk
from tqdm import tqdm

from text import tokenize, process_tokens

dataset_name = "test"  # train or test
dataset_dir = Path("../dataset")
out_path = Path(f"../assets/annotated-corpus/{dataset_name}")

def setup_nltk():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    required_resources = [
        'punkt',
        'averaged_perceptron_tagger_eng',
        'wordnet',
        'omw-1.4',
    ]
    for resource in required_resources:
        nltk.download(resource, quiet=True)

def main() -> None:
    stemmer = SnowballStemmer("english")
    lemmatizer = WordNetLemmatizer()

    data = pd.read_csv(dataset_dir / f'{dataset_name}.csv',
                       header=None,
                       names=['class', 'title', 'text']
                       )

    base_output_dir = Path(out_path)

    for i, row in tqdm(data.iterrows(), total=len(data), desc="processing text"):
        tokenized_sentences = tokenize(row['text'])
        annotations = process_tokens(tokenized_sentences, stemmer, lemmatizer)

        output_path = base_output_dir / str(row['class']) / f'{i}.tsv'
        output_path.parent.mkdir(parents=True, exist_ok=True)

        write_to_disk(output_path, annotations)

def write_to_disk(path_to_write: Path, annotations: list[list[tuple[str, str, str]]]):
    with open(path_to_write, "w", encoding="utf-8") as f:
        for sentence in annotations:
            for tpl in sentence:
                token, stem, lemma = map(str, tpl)
                f.write('\t'.join([token, stem, lemma]) + '\n')
            f.write("\n")


if __name__ == "__main__":
    setup_nltk()
    main()
