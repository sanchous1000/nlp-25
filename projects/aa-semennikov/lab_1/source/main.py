import pandas as pd
import time
from pathlib import Path
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from tqdm import tqdm
from data_processing import tokenize, process_tokens


def main_solo_case():
    sample_text = """
    Dr. Smith conducted a research that shows running improves health! 
    Contact him at doc.smith@gmail.com or +1-555-123-4567.
    You may also visit his website at https://www.drsmith.com - it's way more informative than popular medical TV shows.

    He saw the saw he had left on the left of the nearest log.
    """
    print(f"\nИсходный текст:\n{sample_text}\n")
    stemmer = SnowballStemmer("english")
    lemmatizer = WordNetLemmatizer()

    start_time = time.time()
    tokenized_sentences = tokenize(sample_text)
    print(f"Количество предложений: {len(tokenized_sentences)}\n")
    annotated_sentences = process_tokens(tokenized_sentences, stemmer, lemmatizer)
    
    for i, sentence in enumerate(annotated_sentences, 1):
        print(f"Предложение {i}:")
        print(f"{'Токен':<20} {'Стемма':<20} {'Лемма':<20}")
        print("-" * 60)
        for token, stem, lemma in sentence:
            print(f"{token:<20} {stem:<20} {lemma:<20}")
        print()
    
    res_time = time.time() - start_time
    print(f"Время обработки: {res_time:.2f} сек.")


def main(path):
    data = pd.read_csv(path, header=None, names=['class', 'title', 'text'])
    stemmer = SnowballStemmer("english")
    lemmatizer = WordNetLemmatizer()
    dataset_name = Path(path).stem
    base_output_dir = Path('assets/annotated-corpus') / dataset_name

    start_time = time.time()
    for i, row in tqdm(data.iterrows(), total=len(data)):
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
            print(f"Ошибка при обработке {i}-го документа: {e}")
            continue
    
    res_time = time.time() - start_time
    print(f"Время обработки датасета: {res_time:.2f} сек.")


if __name__ == "__main__":    
    main_solo_case()
    # main('data/train.csv')
    # main('data/test.csv')