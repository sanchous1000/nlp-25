from time import time

import pandas as pd
from source.text_processing import TextProcessor
from tqdm import tqdm


processor = TextProcessor()

for set_ in ["train", "test"]:

    df = pd.read_csv(f'../assets/raw/{set_}.csv', 
                    header=None, names=['category', 'title', 'text'])

    files = {
        c: open(f'../assets/annotated-corpus/{set_}/{c}.tsv', mode='a', encoding='utf-8')
        for c in sorted(df["category"].unique())
    }

    t1 = time()
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        result = processor.process_text(row['text'])
        for t, st, l in zip(result['tokens'], result['stemmed_tokens'], result['lemmatized_tokens']):
            tsv_line = '\t'.join([t, st, l]) + '\n'
            files[row["category"]].write(tsv_line)
        files[row["category"]].write('\n')

    t2 = time()

    for c, f in files.items():
        f.close()

    print(f"Set: {set_} [{len(df)} lines] took {(t2-t1):.3f} s")