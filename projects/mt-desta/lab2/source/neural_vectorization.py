"""
Task 3: Neural network-based vectorization (Word2Vec or GloVe).
"""

import os
from typing import List, Dict
import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from gensim.models.callbacks import CallbackAny2Vec
import gensim.downloader as api


class NeuralVectorizer:
    """
    Neural network-based text vectorization using Word2Vec or GloVe.
    """
    
    def __init__(self, model_type: str = 'word2vec', vector_size: int = 100, window: int = 5, 
                 min_count: int = 2, workers: int = 4, sg: int = 0):
        """
        Initialize neural vectorizer.
        
        Args:
            model_type: 'word2vec' or 'glove'
            vector_size: Dimensionality of word vectors
            window: Maximum distance between current and predicted word
            min_count: Minimum frequency of words to include
            workers: Number of worker threads
            sg: Training algorithm: 0 for CBOW, 1 for skip-gram
        """
        self.model_type = model_type
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.sg = sg
        self.model = None
        self.wv = None  # Word vectors
    
    def train(self, documents: List[List[str]], epochs: int = 10):
        """
        Train Word2Vec model on documents.
        
        Args:
            documents: List of documents, each document is a list of tokens
            epochs: Number of training epochs
        """
        print(f"Training {self.model_type} model...")
        print(f"  Vector size: {self.vector_size}")
        print(f"  Window: {self.window}")
        print(f"  Min count: {self.min_count}")
        print(f"  Training algorithm: {'Skip-gram' if self.sg == 1 else 'CBOW'}")
        print(f"  Number of documents: {len(documents)}")
        
        if self.model_type == 'word2vec':
            self.model = Word2Vec(
                sentences=documents,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                workers=self.workers,
                sg=self.sg,
                epochs=epochs
            )
            self.wv = self.model.wv
        elif self.model_type == 'glove':
            # Gensim doesn't have built-in GloVe, but we can use pre-trained GloVe
            # or train Word2Vec and use it as approximation
            print("Note: Using Word2Vec as GloVe approximation (Gensim doesn't support GloVe training)")
            self.model = Word2Vec(
                sentences=documents,
                vector_size=self.vector_size,
                window=self.window,
                min_count=self.min_count,
                workers=self.workers,
                sg=self.sg,
                epochs=epochs
            )
            self.wv = self.model.wv
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        print(f"Training completed! Vocabulary size: {len(self.wv)}")
    
    def get_word_vector(self, word: str) -> np.ndarray:
        """
        Get vector representation for a word.
        
        Args:
            word: Word token
            
        Returns:
            Word vector of size vector_size
        """
        if self.wv is None:
            raise ValueError("Model not trained. Call train() first.")
        
        word_lower = word.lower().strip()
        if word_lower in self.wv:
            return self.wv[word_lower]
        else:
            # Return zero vector for unknown words
            return np.zeros(self.vector_size, dtype=np.float32)
    
    def get_sentence_vector(self, sentence: List[str], method: str = 'average') -> np.ndarray:
        """
        Get vector representation for a sentence.
        
        Args:
            sentence: List of tokens in the sentence
            method: 'average' for simple average, 'tfidf' for weighted average (requires tfidf weights)
            
        Returns:
            Sentence vector of size vector_size
        """
        vectors = []
        for token in sentence:
            vec = self.get_word_vector(token)
            if np.any(vec != 0):  # Only include known words
                vectors.append(vec)
        
        if not vectors:
            return np.zeros(self.vector_size, dtype=np.float32)
        
        if method == 'average':
            return np.mean(vectors, axis=0)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def get_sentence_vector_tfidf(self, sentence: List[str], tfidf_weights: np.ndarray) -> np.ndarray:
        """
        Get sentence vector using TF-IDF weighted average.
        
        Args:
            sentence: List of tokens in the sentence
            tfidf_weights: TF-IDF weights for each token (must match sentence length)
            
        Returns:
            Sentence vector of size vector_size
        """
        if len(sentence) != len(tfidf_weights):
            raise ValueError("Sentence length must match TF-IDF weights length")
        
        vectors = []
        weights = []
        
        for token, weight in zip(sentence, tfidf_weights):
            vec = self.get_word_vector(token)
            if np.any(vec != 0) and weight > 0:
                vectors.append(vec)
                weights.append(weight)
        
        if not vectors:
            return np.zeros(self.vector_size, dtype=np.float32)
        
        # Weighted average
        weights = np.array(weights)
        weights = weights / np.sum(weights)  # Normalize weights
        
        vectors = np.array(vectors)
        return np.average(vectors, axis=0, weights=weights)
    
    def save(self, filepath: str):
        """Save trained model."""
        if self.model is None:
            raise ValueError("Model not trained. Call train() first.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.model.save(filepath)
        print(f"Saved model to {filepath}")
    
    def load(self, filepath: str):
        """Load trained model."""
        self.model = Word2Vec.load(filepath)
        self.wv = self.model.wv
        self.vector_size = self.model.vector_size
        print(f"Loaded model from {filepath}")
    
    def load_pretrained_glove(self, name: str = 'glove-wiki-gigaword-100'):
        """
        Load pre-trained GloVe model from Gensim.
        
        Args:
            name: Name of pre-trained model
        """
        print(f"Loading pre-trained GloVe model: {name}")
        self.wv = api.load(name)
        self.vector_size = self.wv.vector_size
        self.model_type = 'glove'
        print(f"Loaded model with vocabulary size: {len(self.wv)}")

