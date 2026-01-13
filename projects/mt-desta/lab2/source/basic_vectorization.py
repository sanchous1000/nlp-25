"""
Task 2 (Optional): Basic vectorization methods.
"""

import numpy as np
from typing import List, Dict, Tuple
from scipy.sparse import csr_matrix
try:
    from .token_dictionary import TokenDictionary
except ImportError:
    from token_dictionary import TokenDictionary


class BasicVectorizer:
    """
    Basic vectorization methods for text.
    """
    
    def __init__(self, token_dictionary: TokenDictionary):
        """
        Initialize basic vectorizer.
        
        Args:
            token_dictionary: TokenDictionary instance with vocabulary
        """
        self.token_dict = token_dictionary
        self.vocab_size = token_dictionary.get_vocab_size()
        self.token_to_index = token_dictionary.token_to_index
    
    def frequency_vector(self, text_tokens: List[str]) -> np.ndarray:
        """
        Convert text to frequency vector using token dictionary.
        
        Args:
            text_tokens: List of tokens in the text
            
        Returns:
            Frequency vector of size vocab_size
        """
        vector = np.zeros(self.vocab_size, dtype=np.float32)
        
        for token in text_tokens:
            token_lower = token.lower().strip()
            if token_lower in self.token_to_index:
                idx = self.token_to_index[token_lower]
                vector[idx] += 1.0
        
        return vector
    
    def one_hot_matrix(self, text_tokens: List[str]) -> np.ndarray:
        """
        Convert text to one-hot encoding matrix.
        Each row corresponds to one token, each column to one vocabulary word.
        
        Args:
            text_tokens: List of tokens in the text
            
        Returns:
            One-hot matrix of shape (num_tokens, vocab_size)
        """
        num_tokens = len(text_tokens)
        matrix = np.zeros((num_tokens, self.vocab_size), dtype=np.float32)
        
        for i, token in enumerate(text_tokens):
            token_lower = token.lower().strip()
            if token_lower in self.token_to_index:
                idx = self.token_to_index[token_lower]
                matrix[i, idx] = 1.0
        
        return matrix
    
    def one_hot_vector(self, text_tokens: List[str]) -> np.ndarray:
        """
        Convert one-hot matrix to vector by averaging columns.
        
        Args:
            text_tokens: List of tokens in the text
            
        Returns:
            Vector of size vocab_size
        """
        matrix = self.one_hot_matrix(text_tokens)
        return np.mean(matrix, axis=0)
    
    def frequency_matrix(self, text_tokens: List[str]) -> np.ndarray:
        """
        Convert text to frequency matrix.
        Each row corresponds to one token, each column to one vocabulary word.
        
        Args:
            text_tokens: List of tokens in the text
            
        Returns:
            Frequency matrix of shape (num_tokens, vocab_size)
        """
        num_tokens = len(text_tokens)
        matrix = np.zeros((num_tokens, self.vocab_size), dtype=np.float32)
        
        for i, token in enumerate(text_tokens):
            token_lower = token.lower().strip()
            if token_lower in self.token_to_index:
                idx = self.token_to_index[token_lower]
                matrix[i, idx] += 1.0
        
        return matrix
    
    def frequency_matrix_to_vector(self, text_tokens: List[str]) -> np.ndarray:
        """
        Convert frequency matrix to vector by averaging columns.
        
        Args:
            text_tokens: List of tokens in the text
            
        Returns:
            Vector of size vocab_size
        """
        matrix = self.frequency_matrix(text_tokens)
        return np.mean(matrix, axis=0)
    
    def tf_idf_vector(self, text_tokens: List[str], term_doc_matrix) -> np.ndarray:
        """
        Convert text to TF-IDF vector.
        
        Args:
            text_tokens: List of tokens in the text
            term_doc_matrix: Term-document matrix from TokenDictionary
            
        Returns:
            TF-IDF vector of size vocab_size
        """
        # Calculate TF (term frequency)
        tf_vector = self.frequency_vector(text_tokens)
        
        # Normalize TF by document length
        doc_length = len(text_tokens)
        if doc_length > 0:
            tf_vector = tf_vector / doc_length
        
        # Calculate IDF (inverse document frequency)
        num_docs = term_doc_matrix.shape[1]
        # Count documents containing each term
        doc_counts = np.array((term_doc_matrix > 0).sum(axis=1)).flatten()
        
        # Avoid division by zero
        idf_vector = np.zeros(self.vocab_size, dtype=np.float32)
        for idx in range(self.vocab_size):
            if doc_counts[idx] > 0:
                idf_vector[idx] = np.log(num_docs / doc_counts[idx])
        
        # Calculate TF-IDF
        tf_idf_vector = tf_vector * idf_vector
        
        return tf_idf_vector
    
    def sentence_tf_idf_matrices(self, sentences: List[List[str]], term_doc_matrix) -> Tuple[np.ndarray, np.ndarray]:
        """
        Convert sentences to frequency and TF-IDF matrices.
        Each row corresponds to one sentence, each column to one vocabulary word.
        
        Args:
            sentences: List of sentences, each sentence is a list of tokens
            term_doc_matrix: Term-document matrix from TokenDictionary
            
        Returns:
            Tuple of (frequency_matrix, tf_idf_matrix)
        """
        num_sentences = len(sentences)
        max_sentence_length = max(len(sent) for sent in sentences) if sentences else 0
        
        # Frequency matrix: (num_sentences, vocab_size)
        freq_matrix = np.zeros((num_sentences, self.vocab_size), dtype=np.float32)
        
        # Calculate IDF
        num_docs = term_doc_matrix.shape[1]
        doc_counts = np.array((term_doc_matrix > 0).sum(axis=1)).flatten()
        idf_vector = np.zeros(self.vocab_size, dtype=np.float32)
        for idx in range(self.vocab_size):
            if doc_counts[idx] > 0:
                idf_vector[idx] = np.log(num_docs / doc_counts[idx])
        
        # Build frequency matrix
        for sent_idx, sentence in enumerate(sentences):
            sent_freq = np.zeros(self.vocab_size, dtype=np.float32)
            sent_length = len(sentence)
            
            for token in sentence:
                token_lower = token.lower().strip()
                if token_lower in self.token_to_index:
                    idx = self.token_to_index[token_lower]
                    sent_freq[idx] += 1.0
            
            # Normalize by sentence length
            if sent_length > 0:
                sent_freq = sent_freq / sent_length
            
            freq_matrix[sent_idx] = sent_freq
        
        # Calculate TF-IDF matrix
        tf_idf_matrix = freq_matrix * idf_vector[np.newaxis, :]
        
        return freq_matrix, tf_idf_matrix
    
    def sentence_matrices_to_vector(self, sentences: List[List[str]], term_doc_matrix, use_tfidf: bool = True) -> np.ndarray:
        """
        Convert sentence matrices to document vector by averaging.
        
        Args:
            sentences: List of sentences, each sentence is a list of tokens
            term_doc_matrix: Term-document matrix
            use_tfidf: If True, use TF-IDF; if False, use frequency
            
        Returns:
            Document vector of size vocab_size
        """
        freq_matrix, tf_idf_matrix = self.sentence_tf_idf_matrices(sentences, term_doc_matrix)
        
        if use_tfidf:
            matrix = tf_idf_matrix
        else:
            matrix = freq_matrix
        
        # Average over sentences
        return np.mean(matrix, axis=0)

