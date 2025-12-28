"""
Data loader module for loading preprocessed TSV files from lab1.
"""

import os
from typing import List, Tuple, Dict
from tqdm import tqdm
from pathlib import Path


def load_tsv_file(file_path: str) -> List[Tuple[str, str, str]]:
    """
    Load a TSV file and extract tokens, stems, and lemmas.
    Optimized for faster reading.
    
    Args:
        file_path: Path to the TSV file
        
    Returns:
        List of tuples (token, stem, lemma) for each line
    """
    data = []
    current_sentence = []
    
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line:  # Non-empty line
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        token = parts[0].strip()
                        stem = parts[1].strip()
                        lemma = parts[2].strip()
                        current_sentence.append((token, stem, lemma))
                else:  # Empty line indicates sentence boundary
                    if current_sentence:
                        data.append(current_sentence)
                        current_sentence = []
        
        # Add last sentence if file doesn't end with newline
        if current_sentence:
            data.append(current_sentence)
    except Exception as e:
        # Skip corrupted files
        print(f"Warning: Error loading {file_path}: {e}")
        return []
    
    return data


def load_corpus(data_dir: str, split: str = 'train', use_progress_bar: bool = True) -> Tuple[List[List[List[Tuple[str, str, str]]]], List[int], List[str]]:
    """
    Load all TSV files from the annotated corpus directory.
    Optimized with progress bar and efficient file listing.
    
    Args:
        data_dir: Base directory containing train/test folders
        split: 'train' or 'test'
        use_progress_bar: Whether to show progress bar
        
    Returns:
        Tuple of (documents, labels, doc_ids) where:
        - documents: List of documents, each document is a list of sentences,
          each sentence is a list of (token, stem, lemma) tuples
        - labels: List of labels for each document
        - doc_ids: List of document IDs (filenames without extension)
    """
    split_dir = os.path.join(data_dir, split)
    
    if not os.path.exists(split_dir):
        raise FileNotFoundError(f"Directory {split_dir} not found. Please run lab1 first.")
    
    # First, collect all file paths with their labels
    file_paths = []
    file_labels = []
    file_ids = []
    
    # Iterate through label directories
    for label_dir in sorted(os.listdir(split_dir)):
        label_path = os.path.join(split_dir, label_dir)
        
        if not os.path.isdir(label_path):
            continue
            
        try:
            label = int(label_dir)
        except ValueError:
            continue
        
        # Collect all TSV files in this label directory
        for filename in os.listdir(label_path):
            if filename.endswith('.tsv'):
                file_path = os.path.join(label_path, filename)
                file_paths.append(file_path)
                file_labels.append(label)
                # Extract doc_id from filename (remove .tsv extension)
                doc_id = filename[:-4] if filename.endswith('.tsv') else filename
                file_ids.append(doc_id)
    
    # Load files with progress bar
    documents = []
    labels = []
    doc_ids = []
    
    if use_progress_bar:
        iterator = tqdm(file_paths, desc=f"Loading {split} data", unit="files")
    else:
        iterator = file_paths
    
    for file_path, label, doc_id in zip(iterator, file_labels, file_ids):
        doc_data = load_tsv_file(file_path)
        
        if doc_data:  # Only add non-empty documents
            documents.append(doc_data)
            labels.append(label)
            doc_ids.append(doc_id)
    
    return documents, labels, doc_ids


def get_all_tokens(documents: List[List[List[Tuple[str, str, str]]]], use_lemma: bool = True) -> List[List[str]]:
    """
    Extract all tokens from documents.
    
    Args:
        documents: List of documents (each is a list of sentences)
        use_lemma: If True, use lemmas; if False, use tokens
        
    Returns:
        List of documents, each document is a list of token strings
    """
    result = []
    for doc in documents:
        doc_tokens = []
        for sentence in doc:
            for token, stem, lemma in sentence:
                doc_tokens.append(lemma if use_lemma else token)
        result.append(doc_tokens)
    return result


def get_sentences_as_tokens(documents: List[List[List[Tuple[str, str, str]]]], use_lemma: bool = True) -> List[List[List[str]]]:
    """
    Get documents as lists of sentences, each sentence as a list of tokens.
    
    Args:
        documents: List of documents (each is a list of sentences)
        use_lemma: If True, use lemmas; if False, use tokens
        
    Returns:
        List of documents, each document is a list of sentences, each sentence is a list of tokens
    """
    result = []
    for doc in documents:
        doc_sentences = []
        for sentence in doc:
            sentence_tokens = [lemma if use_lemma else token for token, stem, lemma in sentence]
            doc_sentences.append(sentence_tokens)
        result.append(doc_sentences)
    return result
