"""
Task 7: Document vectorization pipeline.
Segments text into sentences and tokens, gets vector for each token,
calculates sentence vectors, then document vector.
"""

import re
import numpy as np
from typing import List, Tuple
try:
    from .neural_vectorization import NeuralVectorizer
    from .basic_vectorization import BasicVectorizer
except ImportError:
    from neural_vectorization import NeuralVectorizer
    from basic_vectorization import BasicVectorizer


class DocumentVectorizer:
    """
    Document vectorization using neural network models.
    """
    
    def __init__(self, neural_model: NeuralVectorizer, basic_vectorizer: BasicVectorizer = None):
        """
        Initialize document vectorizer.
        
        Args:
            neural_model: Trained NeuralVectorizer instance
            basic_vectorizer: Optional BasicVectorizer for TF-IDF weights
        """
        self.neural_model = neural_model
        self.basic_vectorizer = basic_vectorizer
    
    def segment_text(self, text: str) -> List[List[str]]:
        """
        Segment text into sentences and tokens.
        
        Args:
            text: Input text string
            
        Returns:
            List of sentences, each sentence is a list of tokens
        """
        # Simple sentence segmentation
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Tokenize each sentence
        tokenized_sentences = []
        for sentence in sentences:
            # Simple tokenization (split on whitespace and punctuation)
            tokens = re.findall(r'\b\w+\b', sentence.lower())
            if tokens:
                tokenized_sentences.append(tokens)
        
        return tokenized_sentences
    
    def vectorize_document(self, text: str, use_tfidf_weights: bool = False, 
                          term_doc_matrix=None) -> np.ndarray:
        """
        Vectorize a document using the complete pipeline.
        
        Args:
            text: Input text string
            use_tfidf_weights: If True, use TF-IDF weighted average for sentences
            term_doc_matrix: Term-document matrix (required if use_tfidf_weights=True)
            
        Returns:
            Document vector
        """
        # Step 1: Segment text into sentences and tokens
        sentences = self.segment_text(text)
        
        if not sentences:
            # Return zero vector for empty text
            return np.zeros(self.neural_model.vector_size, dtype=np.float32)
        
        # Step 2: Get vector representation for each token
        # Step 3: Calculate sentence vectors
        sentence_vectors = []
        
        for sentence in sentences:
            if use_tfidf_weights and self.basic_vectorizer is not None and term_doc_matrix is not None:
                # Calculate TF-IDF weights for this sentence
                tfidf_weights = self.basic_vectorizer.tf_idf_vector(sentence, term_doc_matrix)
                
                # Get sentence vector using TF-IDF weighted average
                sentence_vec = self.neural_model.get_sentence_vector_tfidf(sentence, tfidf_weights)
            else:
                # Simple average of token vectors
                sentence_vec = self.neural_model.get_sentence_vector(sentence, method='average')
            
            sentence_vectors.append(sentence_vec)
        
        # Step 4: Calculate document vector from sentence vectors
        # Simple average of sentence vectors
        document_vector = np.mean(sentence_vectors, axis=0)
        
        return document_vector
    
    def vectorize_document_from_tokens(self, sentences: List[List[str]], 
                                      use_tfidf_weights: bool = False,
                                      term_doc_matrix=None) -> np.ndarray:
        """
        Vectorize a document from pre-tokenized sentences.
        
        Args:
            sentences: List of sentences, each sentence is a list of tokens
            use_tfidf_weights: If True, use TF-IDF weighted average
            term_doc_matrix: Term-document matrix (required if use_tfidf_weights=True)
            
        Returns:
            Document vector
        """
        if not sentences:
            return np.zeros(self.neural_model.vector_size, dtype=np.float32)
        
        sentence_vectors = []
        
        for sentence in sentences:
            if use_tfidf_weights and self.basic_vectorizer is not None and term_doc_matrix is not None:
                tfidf_weights = self.basic_vectorizer.tf_idf_vector(sentence, term_doc_matrix)
                sentence_vec = self.neural_model.get_sentence_vector_tfidf(sentence, tfidf_weights)
            else:
                sentence_vec = self.neural_model.get_sentence_vector(sentence, method='average')
            
            sentence_vectors.append(sentence_vec)
        
        document_vector = np.mean(sentence_vectors, axis=0)
        return document_vector

