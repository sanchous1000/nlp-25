import pickle
from pathlib import Path
from tqdm import tqdm

from gensim.models import Word2Vec
from utils import vectorize_document


def vectorize_test_set(test_dir="assets/annotated-corpus/test", 
                       output_file="assets/test_embeddings.tsv",
                       model_path="assets/word2vec.model",
                       term_weights_path="assets/term_weights.pkl"):
    
    model = Word2Vec.load(model_path)
    with open(term_weights_path, 'rb') as file:
        term_weights = pickle.load(file)
    test_path = Path(test_dir)
    test_files = []
    for subdir in ['0', '1', '2', '3']:
        subdir_path = test_path / subdir
        if subdir_path.exists():
            files = list(subdir_path.glob("*.tsv"))
            test_files.extend(files)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    embeddings = []
    failed_files = []
    
    for test_file in tqdm(test_files, desc="Обработка документов"):
        try:
            doc_id = test_file.stem
            doc_vector = vectorize_document(str(test_file), model, term_weights)
            if doc_vector is not None:
                embeddings.append((doc_id, doc_vector))
            else:
                failed_files.append(doc_id)
        except Exception:
            failed_files.append(test_file.stem)

    if failed_files:
        print(f"\nНе удалось обработать {len(failed_files)} файлов")
    if not embeddings:
        raise RuntimeError("Не удалось векторизовать ни один документ")
    vector_size = embeddings[0][1].shape[0]
    for doc_id, vec in embeddings:
        if vec.shape[0] != vector_size:
            raise ValueError(f"Несовпадение размерности векторов: документ {doc_id}")
    
    print(f"\nУспешно векторизовано {len(embeddings)} документов")
    print(f"Размерность векторов: {vector_size}")
    with open(output_file, 'w', encoding='utf-8') as f:
        for doc_id, doc_vector in tqdm(embeddings, desc="Сохранение"):
            components = '\t'.join([f"{component:.6f}" for component in doc_vector])
            f.write(f"{doc_id}\t{components}\n")
    
    print(f"\nРезультаты сохранены в {output_file}")
    print(f"Всего документов: {len(embeddings)}")
    print(f"Размерность векторов: {vector_size}")

    if embeddings:
        print("\nПример первой строки:")
        doc_id, doc_vector = embeddings[0]
        print(f"{doc_id}\t" + "\t".join([f"{c:.6f}" for c in doc_vector[:5]]) + "\t...")

if __name__ == "__main__":
    vectorize_test_set()
