import numpy as np
from gensim.models import Word2Vec
import os
from pathlib import Path
from tqdm import tqdm
import time


def make_sentences(filename):
    with open(filename, 'r', encoding='utf-8') as file:
        sentence = []
        corpus = []

        for line in file:
            stripped_line = line.strip() 
            if stripped_line != '': 
                words = stripped_line.split("\t")
                sentence.append(words[2])
            else:
                if sentence:
                    corpus.append(sentence)
                    sentence = []
        
        if sentence:
            corpus.append(sentence)
            
    return corpus


def vectorize_token(token, model):
    try:
        return model.wv[token.lower()]
    except KeyError:
        return None


def vectorize_sentence(sentence, model):
    vectors = []
    for token in sentence:
        vec = vectorize_token(token, model)
        if vec is not None:
            vectors.append(vec)
    
    if len(vectors) == 0:
        return np.zeros(model.wv.vector_size)
    
    return np.mean(vectors, axis=0)


def vectorize_document(document, model):
    sentence_vectors = []
    for sentence in document:
        sent_vec = vectorize_sentence(sentence, model)
        sentence_vectors.append(sent_vec)
    
    if len(sentence_vectors) == 0:
        return np.zeros(model.wv.vector_size)
    
    return np.mean(sentence_vectors, axis=0)


def process_test_corpus(test_dir, model, output_file):
    start_time = time.time()
    results = []
    test_path = Path(test_dir)
    subdirs = sorted([d for d in test_path.iterdir() if d.is_dir()])
    
    for subdir in subdirs:        
        tsv_files = sorted(subdir.glob("*.tsv"))
        for tsv_file in tqdm(tsv_files, desc=f"Папка {subdir.name}"):
            doc_id = tsv_file.stem
            document = make_sentences(str(tsv_file))
            doc_vector = vectorize_document(document, model)
            results.append((doc_id, doc_vector))
    
    results.sort(key=lambda x: int(x[0]))
    
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc_id, vector in results:
            vector_str = '\t'.join(map(str, vector))
            f.write(f"{doc_id}\t{vector_str}\n")
    
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    print(f"Обработано {len(results)} документов, размерность векторов: {len(results[0][1]) if results else 0}")
    print(f"Время обработки: {elapsed_time:.2f} секунд")


if __name__=="__main__":
    model = Word2Vec.load("assets/word2vec_model.bin")
    test_dir = "assets/annotated-corpus/test"
    output_file = "assets/test_embeddings.tsv"
    process_test_corpus(test_dir, model, output_file)