from pathlib import Path
import re

def sanitize_label(label: str) -> str:
    # Заменяем недопустимые символы на подчёркивания или удаляем
    # Особенно важно: < > : " / \ | ? * и, в Windows, также $
    sanitized = re.sub(r'[<>:"/\\|?*$]', '_', label)
    # Также убираем начальные/конечные пробелы и точки
    sanitized = sanitized.strip().strip('.')
    # Заменяем множественные пробелы/подчёркивания на одно подчёркивание
    sanitized = re.sub(r'[_\s]+', '_', sanitized)
    return sanitized


def save_annotated_corpus(documents):
    base_dir = Path(f"./assets/annotated-corpus")
    
    for doc in documents:
        doc_id = sanitize_label(str(doc["doc_id"]))
        label = str(doc["label"])
        split = doc["split"]  # 'train' или 'test'
        sentences = doc["sentences"]  # list of list of (token, stem, lemma)
        
        output_path = base_dir / split / label / f"{doc_id}.tsv"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w", encoding="utf-8") as f:
            for i, sentence in enumerate(sentences):
                if i > 0:
                    f.write("\n")  # пустая строка между предложениями
                for token, stem, lemma in sentence:
                    f.write(f"{token}\t{stem}\t{lemma}\n")