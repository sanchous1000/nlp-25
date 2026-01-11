import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from gensim.models import Word2Vec
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class TextVectorizer:

    def __init__(self):
        self.model: Optional[Word2Vec] = None
        self.vocab: List[str] = []

    # ---------- Загрузка данных ----------

    def load_annotated_document(self, filepath: Path) -> List[List[str]]:
        """
        Загружает TSV-документ и возвращает список предложений,
        где каждое предложение — список лемм
        """
        sentences = []
        current_sentence = []

        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()

                # Пустая строка = граница предложения
                if not line:
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
                    continue

                parts = line.split('\t')
                if len(parts) >= 3:
                    lemma = parts[2].lower()
                    if lemma.isalpha():
                        current_sentence.append(lemma)

        if current_sentence:
            sentences.append(current_sentence)

        return sentences

    def load_corpus(self, corpus_dir: Path):
        """
        Загружает корпус в формате ЛР1:
        train/pos/*.tsv, train/neg/*.tsv

        Возвращает:
        - sentences: List[List[str]] — для Word2Vec
        - doc_ids: List[str]
        """
        all_sentences = []
        doc_ids = []

        for class_dir in sorted(corpus_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            logger.info(f"Загрузка из {class_dir.name}...")

            for doc_file in sorted(class_dir.glob("*.tsv")):
                doc_id = doc_file.stem
                sentences = self.load_annotated_document(doc_file)

                if sentences:
                    all_sentences.extend(sentences)
                    doc_ids.append(doc_id)

        logger.info(f"Загружено предложений: {len(all_sentences)}")
        return all_sentences, doc_ids

    # ---------- Обучение модели ----------

    def train_word2vec(
        self,
        sentences: List[List[str]],
        vector_size: int = 100,
        window: int = 5,
        min_count: int = 2,
        workers: int = 3,
        epochs: int = 20,
    ) -> None:
        """Обучение Word2Vec строго на предложениях"""

        logger.info(
            f"Word2Vec: size={vector_size}, window={window}, min_count={min_count}, epochs={epochs}"
        )

        self.model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=1,          # skip-gram
            epochs=epochs,
        )

        self.vocab = list(self.model.wv.key_to_index.keys())
        logger.info(f"Словарь: {len(self.vocab)} слов")

    # ---------- Работа со словами ----------

    def word_exists(self, word: str) -> bool:
        return self.model is not None and word in self.model.wv

    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        if self.word_exists(word):
            return self.model.wv[word]
        return None

    def cosine_distance(self, word1: str, word2: str) -> float:
        """Косинусное расстояние (0 = идентичны)"""
        v1 = self.get_word_vector(word1)
        v2 = self.get_word_vector(word2)

        if v1 is None or v2 is None:
            return float('inf')

        sim = cosine_similarity([v1], [v2])[0][0]
        return 1.0 - sim

    def find_similar(self, word: str, topn: int = 10):
        if self.word_exists(word):
            return self.model.wv.most_similar(word, topn=topn)
        return []

    # ---------- Векторизация документов ----------

    def vectorize_sentence(self, sentence: List[str]) -> Optional[np.ndarray]:
        """Усреднение токенов предложения"""
        vectors = [self.model.wv[w] for w in sentence if w in self.model.wv]
        if not vectors:
            return None
        return np.mean(vectors, axis=0)

    def vectorize_document(self, sentences: List[List[str]]) -> Optional[np.ndarray]:
        """Документ = усреднение векторов предложений"""
        sent_vectors = []
        for sent in sentences:
            v = self.vectorize_sentence(sent)
            if v is not None:
                sent_vectors.append(v)

        if not sent_vectors:
            return None

        return np.mean(sent_vectors, axis=0)

    def vectorize_corpus(self, corpus_dir: Path) -> Dict[str, np.ndarray]:
        """Векторизация всего корпуса документов"""
        embeddings: Dict[str, np.ndarray] = {}
        vec_size = self.model.wv.vector_size

        for class_dir in sorted(corpus_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            logger.info(f"Векторизация {class_dir.name}...")

            for doc_file in tqdm(list(class_dir.glob("*.tsv")), desc=class_dir.name):
                doc_id = doc_file.stem
                sentences = self.load_annotated_document(doc_file)

                if not sentences:
                    embeddings[doc_id] = np.zeros(vec_size)
                    continue

                vec = self.vectorize_document(sentences)
                embeddings[doc_id] = vec if vec is not None else np.zeros(vec_size)

        return embeddings

    # ---------- Экспорт ----------

    def save_embeddings_tsv(self, embeddings: Dict[str, np.ndarray], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            for doc_id, emb in sorted(embeddings.items()):
                values = '\t'.join(f"{x:.6f}" for x in emb)
                f.write(f"{doc_id}\t{values}\n")

        logger.info(f"Сохранено: {output_path} (документов: {len(embeddings)})")

    # ----------- TDM ------------

    def build_term_document_matrix(
            self,
            corpus_dir: Path,
            min_df: int = 5,
            max_features=5000,
            vectorizer : CountVectorizer = None
    ) -> Tuple[List[str], List[str], np.ndarray, CountVectorizer]:
        """
        Строит матрицу термин-документ (term-document matrix)

        Возвращает:
        - doc_ids: List[str]
        - vocab: List[str]
        - tdm: np.ndarray shape (n_docs, n_terms)
        """

        logger.info("Построение term-document matrix...")

        documents = []
        doc_ids = []

        # 1. Загружаем документы
        for class_dir in sorted(corpus_dir.iterdir()):
            if not class_dir.is_dir():
                continue

            for doc_file in sorted(class_dir.glob("*.tsv")):
                # 1. Загружаем документы
                doc_id = doc_file.stem
                sentences = self.load_annotated_document(doc_file)
                # разворачиваем предложения в документ
                tokens = [w for sent in sentences for w in sent]
                if tokens:
                    documents.append(" ".join(tokens))
                    doc_ids.append(doc_file.stem)

        logger.info(f"Документов: {len(documents)}")

        # # 2. Строим словарь
        # df_counter = Counter()

        if vectorizer is None:
            vectorizer = CountVectorizer(
                max_features=max_features,
                stop_words='english',
                max_df=0.8,
                min_df=min_df
            )
        tdm = vectorizer.fit_transform(documents)
        vocab = vectorizer.get_feature_names_out()

        logger.info(f"TDM built: {tdm.shape}")
        return doc_ids, vocab, tdm, vectorizer

    # ---- save ----
    def save_tdm_tsv(
            self,
            doc_ids: List[str],
            tdm: np.ndarray,
            output_path: Path
    ) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            if isinstance(tdm, csr_matrix):
                # Для CSR матрицы итерируем построчно
                for i, doc_id in enumerate(doc_ids):
                    row = tdm[i].toarray().flatten()
                    values = '\t'.join(str(int(x)) for x in row)
                    f.write(f"{doc_id}\t{values}\n")
            else:
                for doc_id, row in zip(doc_ids, tdm):
                    values = '\t'.join(str(int(x)) for x in row)
                    f.write(f"{doc_id}\t{values}\n")

        logger.info(f"TDM сохранена: {output_path}")

    def save_vocab(self, vocab: List[str], output_path: Path) -> None:
        with open(output_path, 'w', encoding='utf-8') as f:
            for word in vocab:
                f.write(word + '\n')

        logger.info(f"Словарь сохранён: {output_path}")


