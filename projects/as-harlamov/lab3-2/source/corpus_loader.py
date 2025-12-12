from pathlib import Path
from typing import List, Tuple

import nltk
import pandas as pd
from nltk.corpus import stopwords


try:
    nltk.download('stopwords', quiet=True)
    STOP_WORDS = set(stopwords.words('english'))
except:
    STOP_WORDS = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from', 'as', 'is',
        'was', 'are', 'were', 'been', 'be', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'should',
        'could', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we',
        'they', 'me', 'him', 'her', 'us', 'them',
    }

HTML_ARTIFACTS = {'quot', 'lt', 'gt', 'amp', 'nbsp', 'mdash', 'ndash', 'hellip', 'rsquo', 'lsquo', 'rdquo', 'ldquo'}

DAYS_OF_WEEK = {
    'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday', 'mon', 'tue', 'wed', 'thu', 'fri',
    'sat', 'sun',
}

STOP_WORDS.update(HTML_ARTIFACTS)
STOP_WORDS.update(DAYS_OF_WEEK)


def load_corpus_from_annotated_dir(annotated_dir: Path, max_docs: int = None) -> List[List[str]]:
    documents = []
    total_files = 0
    empty_files = 0
    skipped_tokens = 0
    total_lines = 0
    lines_with_lemma = 0

    if not annotated_dir.exists():
        raise ValueError(f"Директория не существует: {annotated_dir}")

    tsv_files = list(annotated_dir.rglob("*.tsv"))
    if not tsv_files:
        raise ValueError(f"Не найдено TSV файлов в директории: {annotated_dir}")

    print(f"  Найдено TSV файлов: {len(tsv_files)}")

    for tsv_file in sorted(tsv_files):
        total_files += 1
        document = []
        with open(tsv_file, "r", encoding="utf-8") as f:
            for line in f:
                total_lines += 1
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 3:
                    lemma = parts[2].strip()
                    lines_with_lemma += 1
                    if lemma and len(lemma) > 0:
                        lemma_clean = lemma.lower().strip()
                        if (lemma_clean and len(lemma_clean) > 0 and
                            lemma_clean not in STOP_WORDS and
                            len(lemma_clean) > 1 and
                            not lemma_clean.isdigit()):
                            document.append(lemma_clean)
                        else:
                            skipped_tokens += 1
        if document:
            documents.append(document)
        else:
            empty_files += 1

        if total_files <= 5 and documents:
            print(f"    Файл {tsv_file.name}: {len(document)} токенов")

        if max_docs and len(documents) >= max_docs:
            print(f"  Достигнут лимит документов ({max_docs}), остановка загрузки")
            break

    print(f"  Обработано файлов: {total_files}")
    print(f"  Всего строк: {total_lines}, строк с леммами: {lines_with_lemma}")
    print(f"  Пустых файлов: {empty_files}, файлов с содержимым: {len(documents)}")
    if skipped_tokens > 0:
        print(f"  Пропущено токенов (стоп-слова, HTML-артефакты, числа, однобуквенные): {skipped_tokens}")
    if documents:
        print(f"  Пример первого документа (первые 20 токенов): {documents[0][:20]}")
        print(f"  Средняя длина документа: {sum(len(d) for d in documents) / len(documents):.1f} токенов")
    else:
        print("  ВНИМАНИЕ: Все документы пусты после загрузки!")
        print(f"  Проверьте формат TSV файлов. Ожидается: слово\\tстемма\\тлемма")

    return documents


def load_corpus_from_csv(csv_path: Path) -> Tuple[List[str], List[int]]:
    df = pd.read_csv(csv_path, header=None, names=['class', 'title', 'text'])
    texts = df['text'].tolist()
    classes = df['class'].astype(int).tolist()
    return texts, classes
