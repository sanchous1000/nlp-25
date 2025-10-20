import pandas as pd
import argparse
from tqdm import tqdm
import time

from make_token import advanced_tokenize
from make_lemma_and_stem import process_tokens
from annotate import save_annotated_corpus


def main_dataset(args):
    documents = []
    column_names = ['class', 'title', 'text']
    data = pd.read_csv(f"{args.dataset}.csv", header=None, names=column_names)

    for i, row in tqdm(data.iterrows(), total=len(data)):
        text = row['text']
        doc_id = i
        label = row['class']

        token = advanced_tokenize(text)

        token_lemma_stem = process_tokens(token)

        documents.append(
            {
                'doc_id': doc_id,
                'label': label,
                'split': args.dataset,
                'sentences': token_lemma_stem

            }
        )

    save_annotated_corpus(documents)

def main_manual():
    documents = []

    for i in range(1):
        text = """
        A dove flew by. 
        She dove into the pool. 
        I want to ask Mr. Daimon for limonade. 
        +7-952-333-33-33 is my phone number.
        mas@asc.ru is my mail.
        8(952)9999999 is number.
        a=(b+c)^2.
        """

        label = 'omonim'
        doc_id = i

        token = advanced_tokenize(text)

        token_lemma_stem = process_tokens(token)

        documents.append(
            {
                'doc_id': doc_id,
                'label': label,
                'split': 'manual',
                'sentences': token_lemma_stem

            }
        )

    save_annotated_corpus(documents)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)

    args = parser.parse_args()

    start_time = time.time()
    if args.dataset in ['train', 'test']:
        main_dataset(args)
    else:
        main_manual()
    
    end_time = time.time()
    print(f"Время выполнения: {end_time - start_time:.4f} секунд")
    # 350 секунд train
    # 25 секунд test


