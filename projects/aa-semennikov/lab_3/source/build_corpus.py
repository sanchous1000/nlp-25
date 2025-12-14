from gensim.corpora import Dictionary
import glob
import pandas as pd
import numpy as np
import pickle
import os


def load_annotated_corpus(train_dir='assets/annotated-corpus/train'):
    documents = []
    
    for class_dir in sorted(os.listdir(train_dir)):
        class_path = os.path.join(train_dir, class_dir)
        if not os.path.isdir(class_path):
            continue
        
        tsv_files = glob.glob(os.path.join(class_path, '*.tsv'))
        
        for tsv_file in tsv_files:
            df = pd.read_csv(tsv_file, sep='\t', header=None, names=['word', 'stem', 'lemma'])
            words = df['lemma'].dropna().astype(str).tolist()
            words = [w.strip().lower() for w in words if w.strip()]
            
            if words:
                documents.append(words)
    
    return documents


def build_corpus(documents):

    corpus_path = 'assets/corpus.pkl'
    dictionary_path = 'assets/dictionary.pkl'
    dictionary = Dictionary(documents)
    corpus = [dictionary.doc2bow(doc) for doc in documents]
    
    with open(corpus_path, 'wb') as f:
        pickle.dump(corpus, f)
    with open(dictionary_path, 'wb') as f:
        pickle.dump(dictionary, f)


if __name__ == '__main__':
    documents = load_annotated_corpus()
    build_corpus(documents)
    with open('assets/corpus.pkl', 'rb') as f:
        corpus = pickle.load(f)
    with open('assets/dictionary.pkl', 'rb') as f:
        dictionary = pickle.load(f)
    print(f"Количество терминов: {len(dictionary)}")
    print(f"Количество документов: {len(corpus)}")