"""
Task 1 (Optional): Build token dictionary with frequencies and term-document matrix.
Implemented from scratch without using standard library implementations.
"""

import os
import json
from collections import defaultdict
from typing import Dict, List, Tuple
import numpy as np
from scipy.sparse import csr_matrix
import pickle


class TokenDictionary:
    """
    Build token dictionary with frequencies and term-document matrix.
    """
    
    def __init__(self, min_frequency: int = 2, remove_stopwords: bool = True, remove_punctuation: bool = True):
        """
        Initialize token dictionary builder.
        
        Args:
            min_frequency: Minimum token frequency to include in dictionary
            remove_stopwords: Whether to remove stop words
            remove_punctuation: Whether to remove punctuation
        """
        self.min_frequency = min_frequency
        self.remove_stopwords = remove_stopwords
        self.remove_punctuation = remove_punctuation
        
        # Common English stop words
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
            'have', 'had', 'what', 'said', 'each', 'which', 'their', 'time',
            'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
            'her', 'would', 'make', 'like', 'into', 'him', 'has', 'two',
            'more', 'very', 'after', 'words', 'long', 'than', 'first', 'been',
            'call', 'who', 'oil', 'sit', 'now', 'find', 'down', 'day', 'did',
            'get', 'come', 'made', 'may', 'part'
        }
        
        self.token_frequencies = defaultdict(int)
        self.token_to_index = {}
        self.index_to_token = {}
        self.term_document_matrix = None
        self.num_documents = 0
    
    def _is_valid_token(self, token: str) -> bool:
        """Check if token should be included in dictionary."""
        if not token:
            return False
        
        if self.remove_punctuation and not token[0].isalnum():
            return False
        
        if self.remove_stopwords and token.lower() in self.stopwords:
            return False
        
        return True
    
    def build_dictionary(self, documents: List[List[str]]):
        """
        Build token dictionary from documents.
        
        Args:
            documents: List of documents, each document is a list of tokens
        """
        # Count token frequencies
        for doc in documents:
            for token in doc:
                token_lower = token.lower().strip()
                if self._is_valid_token(token_lower):
                    self.token_frequencies[token_lower] += 1
        
        # Filter by minimum frequency
        filtered_tokens = {
            token: freq for token, freq in self.token_frequencies.items()
            if freq >= self.min_frequency
        }
        
        # Create token to index mapping
        sorted_tokens = sorted(filtered_tokens.items(), key=lambda x: (-x[1], x[0]))
        
        for idx, (token, freq) in enumerate(sorted_tokens):
            self.token_to_index[token] = idx
            self.index_to_token[idx] = token
        
        print(f"Built dictionary with {len(self.token_to_index)} tokens")
        print(f"Total tokens processed: {sum(self.token_frequencies.values())}")
        print(f"Tokens filtered out (frequency < {self.min_frequency}): {len(self.token_frequencies) - len(self.token_to_index)}")
    
    def build_term_document_matrix(self, documents: List[List[str]]):
        """
        Build term-document matrix (sparse representation).
        
        Args:
            documents: List of documents, each document is a list of tokens
        """
        self.num_documents = len(documents)
        vocab_size = len(self.token_to_index)
        
        # Build sparse matrix: rows = terms, columns = documents
        row_indices = []
        col_indices = []
        data = []
        
        for doc_idx, doc in enumerate(documents):
            doc_token_counts = defaultdict(int)
            
            for token in doc:
                token_lower = token.lower().strip()
                if token_lower in self.token_to_index:
                    term_idx = self.token_to_index[token_lower]
                    doc_token_counts[term_idx] += 1
            
            # Add to sparse matrix
            for term_idx, count in doc_token_counts.items():
                row_indices.append(term_idx)
                col_indices.append(doc_idx)
                data.append(count)
        
        # Create sparse matrix
        self.term_document_matrix = csr_matrix(
            (data, (row_indices, col_indices)),
            shape=(vocab_size, self.num_documents),
            dtype=np.int32
        )
        
        print(f"Built term-document matrix: {self.term_document_matrix.shape}")
        print(f"Matrix density: {self.term_document_matrix.nnz / (vocab_size * self.num_documents):.4f}")
    
    def save_dictionary(self, filepath: str):
        """Save token dictionary to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        dictionary_data = {
            'token_frequencies': dict(self.token_frequencies),
            'token_to_index': self.token_to_index,
            'index_to_token': self.index_to_token,
            'vocab_size': len(self.token_to_index),
            'num_documents': self.num_documents,
            'min_frequency': self.min_frequency
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(dictionary_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved dictionary to {filepath}")
    
    def save_term_document_matrix(self, filepath: str):
        """Save term-document matrix using pickle (efficient for sparse matrices)."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.term_document_matrix, f)
        
        print(f"Saved term-document matrix to {filepath}")
    
    def load_dictionary(self, filepath: str):
        """Load token dictionary from JSON file."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.token_frequencies = defaultdict(int, data['token_frequencies'])
        self.token_to_index = data['token_to_index']
        self.index_to_token = {int(k): v for k, v in data['index_to_token'].items()}
        self.num_documents = data.get('num_documents', 0)
        self.min_frequency = data.get('min_frequency', 2)
    
    def load_term_document_matrix(self, filepath: str):
        """Load term-document matrix from pickle file."""
        with open(filepath, 'rb') as f:
            self.term_document_matrix = pickle.load(f)
    
    def get_vocab_size(self) -> int:
        """Get vocabulary size."""
        return len(self.token_to_index)
    
    def get_token_frequency(self, token: str) -> int:
        """Get frequency of a token."""
        return self.token_frequencies.get(token.lower(), 0)

