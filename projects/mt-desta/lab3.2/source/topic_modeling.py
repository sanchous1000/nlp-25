"""
Topic Modeling implementation for Lab 3.2

Implements LDA (Latent Dirichlet Allocation) using scikit-learn.
"""

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import LatentDirichletAllocation
from typing import List, Tuple, Dict, Optional


class TopicModeler:
    """
    LDA Topic Modeler wrapper for scikit-learn.
    
    Handles matrix format conversion: lab2 uses (vocab_size, num_documents)
    but sklearn expects (num_documents, vocab_size).
    """
    
    def __init__(self, n_topics: int = 10, n_iter: int = 10, random_state: int = 42):
        """
        Initialize LDA model.
        
        Args:
            n_topics: Number of topics
            n_iter: Number of iterations
            random_state: Random seed
        """
        self.n_topics = n_topics
        self.n_iter = n_iter
        self.random_state = random_state
        self.model = None
        self.is_trained = False
        self.expected_vocab_size = None
    
    def train(self, term_doc_matrix: csr_matrix) -> float:
        """
        Train LDA model on term-document matrix.
        
        Args:
            term_doc_matrix: Term-document matrix (vocab_size, num_documents)
        
        Returns:
            Training time in seconds
        """
        import time
        
        # Store expected vocabulary size
        vocab_size, num_docs = term_doc_matrix.shape
        self.expected_vocab_size = vocab_size
        
        # sklearn expects (num_documents, vocab_size), so transpose
        X = term_doc_matrix.T  # Now (num_documents, vocab_size)
        
        # Initialize model
        self.model = LatentDirichletAllocation(
            n_components=self.n_topics,
            max_iter=self.n_iter,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # Train
        start_time = time.time()
        self.model.fit(X)
        training_time = time.time() - start_time
        
        self.is_trained = True
        
        return training_time
    
    def get_top_words(
        self,
        vocabulary: List[str],
        n_words: int = 10
    ) -> Dict[int, List[Tuple[str, float]]]:
        """
        Get top words for each topic.
        
        Args:
            vocabulary: List of vocabulary words
            n_words: Number of top words per topic
        
        Returns:
            Dictionary mapping topic_id -> list of (word, weight) tuples
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Verify vocabulary size matches model
        if len(vocabulary) != self.expected_vocab_size:
            # Adjust vocabulary to match model
            if len(vocabulary) > self.expected_vocab_size:
                vocabulary = vocabulary[:self.expected_vocab_size]
            else:
                # Extend vocabulary if needed (shouldn't happen)
                vocabulary = vocabulary + [f"word_{i}" for i in range(len(vocabulary), self.expected_vocab_size)]
        
        # Get topic-word distribution (n_topics, vocab_size)
        topic_word_dist = self.model.components_
        
        top_words = {}
        
        for topic_idx in range(self.n_topics):
            topic_weights = topic_word_dist[topic_idx]
            top_indices = np.argsort(topic_weights)[-n_words:][::-1]
            
            top_words[topic_idx] = [
                (vocabulary[idx], topic_weights[idx])
                for idx in top_indices
            ]
        
        return top_words
    
    def get_document_topic_distribution(
        self,
        term_doc_matrix: csr_matrix
    ) -> np.ndarray:
        """
        Get document-topic probability distribution.
        
        Args:
            term_doc_matrix: Term-document matrix (vocab_size, num_documents)
        
        Returns:
            Array of shape (num_documents, n_topics) with topic probabilities
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Verify vocabulary size
        vocab_size, num_docs = term_doc_matrix.shape
        if vocab_size != self.expected_vocab_size:
            raise ValueError(
                f"Vocabulary size mismatch: matrix has {vocab_size} features, "
                f"but model expects {self.expected_vocab_size} features."
            )
        
        # sklearn expects (num_documents, vocab_size), so transpose
        X = term_doc_matrix.T  # Now (num_documents, vocab_size)
        
        # Get document-topic distribution
        doc_topic_dist = self.model.transform(X)
        
        return doc_topic_dist
    
    def get_perplexity(self, term_doc_matrix: csr_matrix) -> float:
        """
        Calculate perplexity on test set.
        
        Args:
            term_doc_matrix: Term-document matrix (vocab_size, num_documents)
        
        Returns:
            Perplexity score
        """
        if not self.is_trained:
            raise ValueError("Model not trained. Call train() first.")
        
        # Verify vocabulary size
        original_shape = term_doc_matrix.shape
        vocab_size, num_docs = original_shape
        
        if vocab_size != self.expected_vocab_size:
            raise ValueError(
                f"Vocabulary size mismatch: matrix has {vocab_size} features, "
                f"but model expects {self.expected_vocab_size} features. "
                f"Matrix shape: {original_shape}"
            )
        
        # sklearn expects (num_documents, vocab_size), so transpose
        X = term_doc_matrix.T  # Now (num_documents, vocab_size)
        
        # Calculate perplexity
        perplexity = self.model.perplexity(X)
        
        return perplexity
    
    def get_top_documents_per_topic(
        self,
        doc_topic_dist: np.ndarray,
        n_docs: int = 10
    ) -> Dict[int, List[Tuple[int, float]]]:
        """
        Get documents with highest probability for each topic.
        
        Args:
            doc_topic_dist: Document-topic distribution (num_documents, n_topics)
            n_docs: Number of top documents per topic
        
        Returns:
            Dictionary mapping topic_id -> list of (doc_idx, probability) tuples
        """
        top_documents = {}
        
        for topic_idx in range(self.n_topics):
            topic_probs = doc_topic_dist[:, topic_idx]
            top_indices = np.argsort(topic_probs)[-n_docs:][::-1]
            
            top_documents[topic_idx] = [
                (int(doc_idx), float(topic_probs[doc_idx]))
                for doc_idx in top_indices
            ]
        
        return top_documents

