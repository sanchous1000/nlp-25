import logging
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import numpy as np
from gensim.models import Word2Vec
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent.parent.parent
ANNOTATED_CORPUS_DIR = BASE_DIR / "assets" / "annotated-corpus"
TRAIN_DIR = ANNOTATED_CORPUS_DIR / "train"
TEST_DIR = ANNOTATED_CORPUS_DIR / "test"
OUTPUT_DIR = BASE_DIR / "lab2" / "output"


class TextVectorizer:
    
    def __init__(self):
        self.model: Optional[Word2Vec] = None
        self.vocab: List[str] = []
    
    def load_annotated_document(self, filepath: Path) -> List[str]:
        """Загружает токены из TSV файла (берем леммы из 3й колонки)"""
        tokens = []
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        lemma = parts[2].lower()
                        if len(lemma) > 1 and not lemma.isdigit():
                            tokens.append(lemma)
        except Exception as e:
            logger.warning(f"Ошибка чтения {filepath}: {e}")
        return tokens
    
    def load_corpus(self, corpus_dir: Path) -> Tuple[List[List[str]], List[str]]:
        """Загружает корпус документов из директории"""
        documents = []
        doc_ids = []
        
        for class_dir in sorted(corpus_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            
            logger.info(f"Загрузка из {class_dir.name}...")
            
            for doc_file in sorted(class_dir.glob("*.tsv")):
                tokens = self.load_annotated_document(doc_file)
                if tokens:
                    documents.append(tokens)
                    doc_ids.append(doc_file.stem)
        
        logger.info(f"Загружено {len(documents)} документов")
        return documents, doc_ids
    
    def prepare_sentences(self, documents: List[List[str]]) -> List[List[str]]:
        sentences = [doc for doc in documents if len(doc) >= 3]
        logger.info(f"Предложений: {len(sentences)}, токенов: {sum(len(s) for s in sentences)}")
        return sentences
    
    def train_word2vec(self, sentences: List[List[str]], vector_size: int = 100,
                       window: int = 5, min_count: int = 3, workers: int = 4,
                       epochs: int = 20) -> None:
        logger.info(f"Word2Vec: size={vector_size}, window={window}, min_count={min_count}")
        
        self.model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            sg=1,
            epochs=epochs
        )
        self.vocab = list(self.model.wv.key_to_index.keys())
        logger.info(f"Словарь: {len(self.vocab)} слов")
    
    def get_word_vector(self, word: str) -> Optional[np.ndarray]:
        if self.model and word in self.model.wv:
            return self.model.wv[word]
        return None
    
    def cosine_distance(self, word1: str, word2: str) -> float:
        vec1 = self.get_word_vector(word1)
        vec2 = self.get_word_vector(word2)
        
        if vec1 is None or vec2 is None:
            return float('inf')
        
        similarity = cosine_similarity([vec1], [vec2])[0][0]
        return 1 - similarity
    
    def find_similar(self, word: str, topn: int = 10) -> List[Tuple[str, float]]:
        if self.model and word in self.model.wv:
            return self.model.wv.most_similar(word, topn=topn)
        return []
    
    def word_exists(self, word: str) -> bool:
        return self.model is not None and word in self.model.wv
    
    def vectorize_document(self, tokens: List[str]) -> Optional[np.ndarray]:
        """Усредняет векторы токенов документа"""
        vectors = []
        for token in tokens:
            vec = self.get_word_vector(token.lower())
            if vec is not None:
                vectors.append(vec)
        
        if not vectors:
            return None
        
        return np.mean(vectors, axis=0)
    
    def vectorize_corpus(self, corpus_dir: Path) -> Tuple[Dict[str, np.ndarray], int]:
        """Векторизует весь корпус"""
        embeddings = {}
        vector_size = self.model.wv.vector_size if self.model else 100
        
        for class_dir in sorted(corpus_dir.iterdir()):
            if not class_dir.is_dir():
                continue
            
            logger.info(f"Векторизация {class_dir.name}...")
            
            for doc_file in tqdm(list(class_dir.glob("*.tsv")), desc=class_dir.name):
                doc_id = doc_file.stem
                tokens = self.load_annotated_document(doc_file)
                
                if tokens:
                    embedding = self.vectorize_document(tokens)
                    if embedding is not None:
                        embeddings[doc_id] = embedding
                    else:
                        embeddings[doc_id] = np.zeros(vector_size)
        
        return embeddings, vector_size
    
    def save_embeddings_tsv(self, embeddings: Dict[str, np.ndarray], output_path: Path) -> None:
        """Сохраняет эмбеддинги в TSV"""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for doc_id, embedding in sorted(embeddings.items()):
                components = '\t'.join(f"{x:.6f}" for x in embedding)
                f.write(f"{doc_id}\t{components}\n")
        
        logger.info(f"Сохранено в {output_path}, документов: {len(embeddings)}")


def show_semantic_demo(vectorizer: TextVectorizer):
    print("Демонстрация семантических отношений")

    test_groups = [
        {
            'word': 'company',
            'similar': ['corporation', 'firm', 'business'],
            'domain': ['market', 'stock', 'industry'],
            'different': ['football', 'weather', 'music']
        },
        {
            'word': 'president',
            'similar': ['leader', 'chief', 'chairman'],
            'domain': ['government', 'election', 'political'],
            'different': ['computer', 'sport', 'movie']
        },
        {
            'word': 'game',
            'similar': ['match', 'play', 'sport'],
            'domain': ['team', 'player', 'win'],
            'different': ['economy', 'science', 'oil']
        }
    ]
    
    for group in test_groups:
        base = group['word']
        
        if not vectorizer.word_exists(base):
            print(f"\n'{base}' нет в словаре")
            continue
        
        print(f"\n--- {base} ---")
        
        all_dists = []
        
        print("Близкие:")
        for w in group['similar']:
            if vectorizer.word_exists(w):
                d = vectorizer.cosine_distance(base, w)
                all_dists.append((w, 'близкие', d))
                print(f"  {w}: {d:.4f}")
        
        print("Та же область:")
        for w in group['domain']:
            if vectorizer.word_exists(w):
                d = vectorizer.cosine_distance(base, w)
                all_dists.append((w, 'область', d))
                print(f"  {w}: {d:.4f}")
        
        print("Разные:")
        for w in group['different']:
            if vectorizer.word_exists(w):
                d = vectorizer.cosine_distance(base, w)
                all_dists.append((w, 'разные', d))
                print(f"  {w}: {d:.4f}")
        
        if all_dists:
            print(f"\nОтсортировано:")
            for i, (w, cat, d) in enumerate(sorted(all_dists, key=lambda x: x[2]), 1):
                print(f"  {i}. {w}: {d:.4f} ({cat})")
            
            close = [d for w, c, d in all_dists if c == 'близкие']
            domain = [d for w, c, d in all_dists if c == 'область']
            diff = [d for w, c, d in all_dists if c == 'разные']
            
            if close and domain and diff:
                print(f"\nСредние: близкие={np.mean(close):.4f}, область={np.mean(domain):.4f}, разные={np.mean(diff):.4f}")
    
    print("\n" + "=" * 60)
    print("Похожие слова из модели")
    print("=" * 60)
    
    for word in ['company', 'president', 'game', 'market', 'oil', 'year']:
        if vectorizer.word_exists(word):
            similar = vectorizer.find_similar(word, topn=5)
            if similar:
                print(f"\n{word}:")
                for w, sim in similar:
                    print(f"  {w}: {sim:.4f}")


def run_experiments(sentences: List[List[str]]) -> Tuple[Dict, TextVectorizer]:
    """Эксперименты с гиперпараметрами"""
    
    print("\n" + "=" * 60)
    print("Эксперименты с гиперпараметрами")
    print("=" * 60)
    
    configs = [
        {'vector_size': 50, 'window': 3, 'min_count': 2, 'epochs': 15},
        {'vector_size': 100, 'window': 5, 'min_count': 2, 'epochs': 20},
        {'vector_size': 100, 'window': 7, 'min_count': 3, 'epochs': 20},
        {'vector_size': 150, 'window': 5, 'min_count': 2, 'epochs': 25},
    ]
    
    test_pairs = {
        'close': [
            ('company', 'corporation'), ('company', 'firm'),
            ('president', 'leader'), ('president', 'chief'),
            ('game', 'match'), ('game', 'play')
        ],
        'domain': [
            ('company', 'market'), ('company', 'stock'),
            ('president', 'government'), ('president', 'election'),
            ('game', 'team'), ('game', 'player')
        ],
        'different': [
            ('company', 'football'), ('company', 'weather'),
            ('president', 'computer'), ('president', 'sport'),
            ('game', 'economy'), ('game', 'oil')
        ]
    }
    
    results = []
    
    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}] {cfg}")
        
        vec = TextVectorizer()
        vec.train_word2vec(sentences, **cfg)
        
        avgs = {}
        for cat, pairs in test_pairs.items():
            dists = []
            for w1, w2 in pairs:
                if vec.word_exists(w1) and vec.word_exists(w2):
                    d = vec.cosine_distance(w1, w2)
                    if d != float('inf'):
                        dists.append(d)
            avgs[cat] = np.mean(dists) if dists else float('inf')
        
        score = avgs['different'] - avgs['close'] if all(v != float('inf') for v in avgs.values()) else -float('inf')
        
        result = {
            'idx': i + 1,
            'cfg': cfg,
            'close': avgs['close'],
            'domain': avgs['domain'],
            'different': avgs['different'],
            'vocab': len(vec.vocab),
            'score': score
        }
        results.append((result, vec))
        
        print(f"  vocab={result['vocab']}, close={result['close']:.4f}, domain={result['domain']:.4f}, diff={result['different']:.4f}, score={score:.4f}")
    
    valid = [(r, v) for r, v in results if r['score'] != -float('inf')]
    
    if valid:
        best, best_vec = max(valid, key=lambda x: x[0]['score'])
        print(f"\nЛучший: эксперимент {best['idx']}, score={best['score']:.4f}")
        return best, best_vec
    
    return results[-1]


def main():
    vectorizer = TextVectorizer()
    
    print("\nЗагрузка данных...")
    if not TRAIN_DIR.exists():
        logger.error(f"Не найдено: {TRAIN_DIR}")
        return
    
    docs, _ = vectorizer.load_corpus(TRAIN_DIR)
    if not docs:
        logger.error("Пустая выборка")
        return
    
    sentences = vectorizer.prepare_sentences(docs)
    
    best, vectorizer = run_experiments(sentences)
    
    show_semantic_demo(vectorizer)
    
    print("Векторизация тестовой выборки")

    if not TEST_DIR.exists():
        logger.error(f"Не найдено: {TEST_DIR}")
        return
    
    embeddings, vec_size = vectorizer.vectorize_corpus(TEST_DIR)
    
    out_file = OUTPUT_DIR / "test_embeddings.tsv"
    vectorizer.save_embeddings_tsv(embeddings, out_file)

    print('done')

if __name__ == "__main__":
    main()
