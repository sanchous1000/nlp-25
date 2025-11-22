from gensim.models import Word2Vec
import pandas as pd
import re
from nltk.tokenize import word_tokenize
import nltk
from tqdm import tqdm
import time

df = pd.read_csv("../lab_1/data/train.csv", header=None, names=["label", "title", "text"])

def clean_text(text):
    if pd.isna(text):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

corpus = []
for idx, row in tqdm(df.iterrows(), total=len(df)):
    combined_text = str(row['title']) + " " + str(row['text'])
    cleaned_text = clean_text(combined_text)
    if cleaned_text:
        corpus.append(cleaned_text)

sentences = [text.split() for text in corpus]

w2v = Word2Vec(
    vector_size=30,      # размерность векторов
    window=5,            # размер контекстного окна
    min_count=5,         # минимальная частота слова
    workers=4,           # количество потоков
    sg=1                # 1 = skip-gram, 0 = CBOW
)

# Построение словаря
w2v.build_vocab(sentences)
print(f"Размер словаря: {len(w2v.wv)}")

# Обучение модели на обучающей выборке
start_time = time.time()
w2v.train(
    sentences,
    total_examples=w2v.corpus_count,
    epochs=30
)
end_time = time.time()
training_time = end_time - start_time
print(f"Время обучения: {training_time:.2f} секунд")

w2v.save("assets/word2vec_model.bin")