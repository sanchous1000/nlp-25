"""
Data loader module for loading embeddings from lab2.
"""

import os
import numpy as np
from typing import Tuple, List
from tqdm import tqdm


def load_embeddings(embeddings_file: str) -> Tuple[np.ndarray, List[str]]:
    """
    Load document embeddings from TSV file generated in lab2.
    
    Args:
        embeddings_file: Path to TSV file with embeddings
        
    Returns:
        Tuple of (embeddings, doc_ids) where:
        - embeddings: numpy array of shape (n_documents, embedding_dim)
        - doc_ids: List of document IDs
    """
    embeddings = []
    doc_ids = []
    
    with open(embeddings_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc="Loading embeddings"):
            line = line.strip()
            if not line:
                continue
            
            parts = line.split('\t')
            if len(parts) < 2:
                continue
            
            doc_id = parts[0].strip()
            embedding_values = [float(x) for x in parts[1:]]
            
            if len(embedding_values) == 0:
                continue
            
            doc_ids.append(doc_id)
            embeddings.append(embedding_values)
    
    if len(embeddings) == 0:
        raise ValueError(f"No valid embeddings found in {embeddings_file}")
    
    embeddings = np.array(embeddings, dtype=np.float32)
    print(f"Loaded {len(embeddings)} embeddings with dimension {embeddings.shape[1]}")
    print(f"Sample doc_ids (first 5): {doc_ids[:5]}")
    return embeddings, doc_ids


def load_labels_from_corpus(data_dir: str, split: str = 'test', doc_ids: List[str] = None) -> np.ndarray:
    """
    Load labels for documents from lab1 corpus structure.
    
    Args:
        data_dir: Base directory containing train/test folders
        split: 'train' or 'test'
        doc_ids: List of document IDs to match
        
    Returns:
        Array of labels corresponding to doc_ids
    """
    split_dir = os.path.join(data_dir, split)
    
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Directory {split_dir} not found. Please run lab1 first.")
    
    # Create mapping from doc_id to label
    doc_id_to_label = {}
    
    for label_dir in sorted(os.listdir(split_dir)):
        label_path = os.path.join(split_dir, label_dir)
        
        if not os.path.isdir(label_path):
            continue
            
        try:
            label = int(label_dir)
        except ValueError:
            continue
        
        # Get all TSV files in this label directory
        for filename in os.listdir(label_path):
            if filename.endswith('.tsv'):
                doc_id = filename[:-4] if filename.endswith('.tsv') else filename
                doc_id_to_label[doc_id] = label
    
    # Map doc_ids to labels
    if doc_ids is None:
        # Return all labels in order
        labels = []
        for label_dir in sorted(os.listdir(split_dir)):
            label_path = os.path.join(split_dir, label_dir)
            if not os.path.isdir(label_path):
                continue
            try:
                label = int(label_dir)
            except ValueError:
                continue
            for filename in sorted(os.listdir(label_path)):
                if filename.endswith('.tsv'):
                    labels.append(label)
        return np.array(labels)
    else:
        labels = []
        for doc_id in doc_ids:
            label = doc_id_to_label.get(doc_id, -1)
            if label == -1:
                # Try to find label by checking if doc_id matches any file
                for label_dir in sorted(os.listdir(split_dir)):
                    label_path = os.path.join(split_dir, label_dir)
                    if not os.path.isdir(label_path):
                        continue
                    try:
                        label_val = int(label_dir)
                    except ValueError:
                        continue
                    for filename in os.listdir(label_path):
                        if filename.endswith('.tsv'):
                            file_doc_id = filename[:-4] if filename.endswith('.tsv') else filename
                            if file_doc_id == doc_id:
                                label = label_val
                                break
            labels.append(label)
        return np.array(labels)


def load_train_data(lab1_corpus_dir: str, lab2_embeddings_file: str = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load training data: embeddings and labels.
    Note: For now, we use test embeddings. In practice, you should generate train embeddings in lab2.
    
    Args:
        lab1_corpus_dir: Directory with lab1 corpus
        lab2_embeddings_file: Path to embeddings file from lab2 (optional)
        
    Returns:
        Tuple of (X_train, y_train)
    """
    if lab2_embeddings_file and os.path.exists(lab2_embeddings_file):
        # Load pre-computed embeddings
        embeddings, doc_ids = load_embeddings(lab2_embeddings_file)
        # Try to load train labels, fallback to test if train embeddings not available
        try:
            labels = load_labels_from_corpus(lab1_corpus_dir, split='train', doc_ids=doc_ids)
        except:
            labels = load_labels_from_corpus(lab1_corpus_dir, split='test', doc_ids=doc_ids)
    else:
        # Need to generate embeddings - this would require running lab2
        raise FileNotFoundError("Embeddings file not found. Please run lab2 first to generate embeddings.")
    
    return embeddings, labels


def load_test_data(lab1_corpus_dir: str, lab2_embeddings_file: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load test data: embeddings and labels.
    
    Args:
        lab1_corpus_dir: Directory with lab1 corpus
        lab2_embeddings_file: Path to embeddings file from lab2
        
    Returns:
        Tuple of (X_test, y_test)
    """
    embeddings, doc_ids = load_embeddings(lab2_embeddings_file)
    labels = load_labels_from_corpus(lab1_corpus_dir, split='test', doc_ids=doc_ids)
    
    # Debug: Check label matching
    invalid_count = np.sum(labels == -1)
    if invalid_count > 0:
        print(f"Warning: {invalid_count} out of {len(labels)} documents have invalid labels")
        print(f"Sample unmatched doc_ids: {[doc_ids[i] for i in range(min(5, len(doc_ids))) if labels[i] == -1]}")
    
    # Filter out invalid labels
    valid_mask = labels != -1
    if np.sum(valid_mask) < len(labels):
        print(f"Filtering out {np.sum(~valid_mask)} documents with invalid labels")
        embeddings = embeddings[valid_mask]
        labels = labels[valid_mask]
    
    print(f"Final dataset: {len(embeddings)} documents with valid labels")
    print(f"Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")
    
    return embeddings, labels

