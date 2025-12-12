import sys
import warnings
from pathlib import Path

import pandas as pd
from gensim.models import Word2Vec


warnings.filterwarnings('ignore')


def _import_lab2_modules(lab2_source_path: Path):
    import importlib.util

    utils_spec = importlib.util.spec_from_file_location("utils", lab2_source_path / "utils.py")
    utils_module = importlib.util.module_from_spec(utils_spec)
    sys.modules["utils"] = utils_module
    utils_spec.loader.exec_module(utils_module)

    vectorizer_spec = importlib.util.spec_from_file_location("vectorizer", lab2_source_path / "vectorizer.py")
    vectorizer_module = importlib.util.module_from_spec(vectorizer_spec)
    sys.modules["vectorizer"] = vectorizer_module
    vectorizer_spec.loader.exec_module(vectorizer_module)

    return vectorizer_module.document_to_vector


def document_to_vector(text: str, w2v_model, vector_size: int):
    lab2_path = Path(__file__).parent.parent.parent / 'lab2' / 'source'
    doc_to_vec = _import_lab2_modules(lab2_path)
    return doc_to_vec(text, w2v_model, vector_size)


def generate_vectors_for_dataset(
    csv_path: Path,
    w2v_model_path: Path,
    output_path: Path,
    vector_size: int = 100,
):
    print(f"Загрузка Word2Vec модели из {w2v_model_path}...")
    w2v_model = Word2Vec.load(str(w2v_model_path))

    print(f"Загрузка датасета из {csv_path}...")
    df = pd.read_csv(csv_path, header=None, names=['class', 'title', 'text'])

    print(f"Генерация векторов для {len(df)} документов...")
    output_lines = []
    for i, row in df.iterrows():
        doc_id = str(i)
        vec = document_to_vector(row['text'], w2v_model, vector_size)
        line = [doc_id] + [f"{x:.6f}" for x in vec.tolist()]
        output_lines.append("\t".join(line))
        if (i + 1) % 1000 == 0:
            print(f"  Обработано {i + 1}/{len(df)} документов...")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write("\n".join(output_lines))

    print(f"Векторы сохранены в {output_path}")
