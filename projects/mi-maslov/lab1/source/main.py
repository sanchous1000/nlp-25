import os
from pathlib import Path
import re
import nltk
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize
from multiprocessing import Pool, cpu_count

nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)
nltk.download('wordnet', quiet=True)

stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

TOKEN_PATTERNS = [
    r'\+?\d[\d\-\(\) ]+\d',
    r'\b\w+@\w+\.\w+\b',
    r'(?:Dr|Mr|Ms|Mrs|Sen|Rep)\.',
    r'\b[A-Za-z]+\s+St\.\s+[A-Za-z]+',
    r'[:;=8][\-o\*\']?[\)\]\(\[dDpP/:\}\{@\|\\]',
    r'\b[a-zA-Z]\s*=\s*[a-zA-Z0-9\+\-\*/\^\(\)]+\b',
    r'\w+'
]
TOKEN_REGEX = re.compile('|'.join(TOKEN_PATTERNS))

def tokenize(text):
    return TOKEN_REGEX.findall(text)

def process_text(text):
    sentences = sent_tokenize(text, language='english')
    annotated_sentences = []
    for sentence in sentences:
        tokens = tokenize(sentence)
        annotated = []
        for token in tokens:
            stem = stemmer.stem(token)
            lemma = lemmatizer.lemmatize(token.lower())
            annotated.append((token, stem, lemma))
        annotated_sentences.append(annotated)
    return annotated_sentences

def save_tsv(doc_id, class_name, annotated_sentences, base_dir, split_name):
    class_dir = os.path.join(base_dir, split_name, class_name)
    os.makedirs(class_dir, exist_ok=True)
    file_path = os.path.join(class_dir, f"{doc_id}.tsv")
    with open(file_path, "w", encoding="utf-8") as f:
        for sentence in annotated_sentences:
            for token, stem, lemma in sentence:
                f.write(f"{token}\t{stem}\t{lemma}\n")
            f.write("\n")

def process_file(args):
    fpath, base_dir, split_name, class_name = args
    with open(fpath, "r", encoding="latin1") as f:
        text = f.read()
    doc_id = fpath.stem
    annotated = process_text(text)
    save_tsv(doc_id, class_name, annotated, base_dir, split_name)
    return doc_id

base_dir = "./annotated-corpus"

splits = {
    "train": "/content/20news-bydate-train",
    "test": "/content/20news-bydate-test"
}

for split_name, split_path in splits.items():
    categories = [d for d in os.listdir(split_path) if os.path.isdir(os.path.join(split_path, d))]
    for cat in categories:
        cat_path = os.path.join(split_path, cat)
        files = list(Path(cat_path).glob("*"))
        args_list = [(fpath, base_dir, split_name, cat) for fpath in files]

        with Pool(processes=cpu_count()) as pool:
            for doc_id in pool.imap_unordered(process_file, args_list):
                print(f"Обработан: {doc_id}")