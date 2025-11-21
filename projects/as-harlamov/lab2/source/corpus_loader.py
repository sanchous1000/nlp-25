from pathlib import Path
from typing import List


def load_corpus_from_annotated_dir(annotated_dir: Path) -> List[List[str]]:
    sentences = []
    for tsv_file in annotated_dir.rglob("*.tsv"):
        with open(tsv_file, "r", encoding="utf-8") as f:
            sentence = []
            for line in f:
                line = line.strip()
                if not line:
                    if sentence:
                        sentences.append(sentence)
                        sentence = []
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    lemma = parts[2]
                    if lemma.isalpha():
                        sentence.append(lemma.lower())
            if sentence:
                sentences.append(sentence)
    return sentences
