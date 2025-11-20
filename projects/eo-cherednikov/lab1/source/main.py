import re
from nltk.stem.snowball import SnowballStemmer
from pymorphy2 import MorphAnalyzer
import os

stemmer = SnowballStemmer("english")
morph = MorphAnalyzer()

email = r'(?P<EMAIL>[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,})'
phone = r'(?P<PHONE>\+?\d[\d\-\(\)\s]{6,}\d)'
emot = r'(?P<EMOT>:\-\)|:\)|;\)|:-\(|:\(|:D|=\)|\(:|\);)'
abbr = r'(?P<ABBR>[A-ZА-Я][a-zа-я]{0,3}\.)'

token = r'(?P<WORD>[A-Za-zА-Яа-яЁё0-9]+)|(?P<PUNCT>[^A-Za-zА-Яа-яЁё0-9\s])'

master = f"{email}|{phone}|{emot}|{abbr}|{token}"

ABBR = r'([A-ZА-Я][a-zа-я]{0,3}\.)'

def tokenize(text):
    return [m.group(0) for m in re.finditer(master, text)]

def stem(t):
    return stemmer.stem(t.lower())

def lemma(t):
    p = morph.parse(t)
    return p[0].normal_form

def segment(text):
    tmp = re.sub(ABBR, lambda m: m.group(1).replace('.', '<DOT>'), text)
    sents = re.split(r'(?<=[.!?])\s+', tmp)
    return [s.replace('<DOT>', '.') for s in sents]

def process_document(text):
    out_lines = []
    for sent in segment(text):
        toks = tokenize(sent)
        for t in toks:
            out_lines.append(f"{t}\t{stem(t)}\t{lemma(t)}")
        out_lines.append("")
    return "\n".join(out_lines).rstrip()

def write_split(df, split):
    base = f"assets/annotated-corpus/{split}"
    for _, row in df.iterrows():
        label = str(row["labels"])
        text = str(row["text"])

        doc_id = str(row["id"]) if "id" in row else str(_)

        dir_path = os.path.join(base, label)
        os.makedirs(dir_path, exist_ok=True)

        out_path = os.path.join(dir_path, f"{doc_id}.tsv")

        content = process_document(text)

        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)

train_df = pd.read_csv("assets/raw/train.csv")
test_df  = pd.read_csv("assets/raw/test.csv")

write_split(train_df, "train")
write_split(test_df, "test")
