"""
Data loader for Lab 3.2 - Topic Modeling

Loads term-document matrix and vocabulary from Lab 2.
"""

import os
import json
import pickle
import re
import numpy as np
from scipy.sparse import csr_matrix
from typing import Tuple, List, Dict, Optional

try:
    import nltk
    from nltk.corpus import stopwords
    # Download stopwords if not already downloaded
    try:
        nltk.data.find('corpora/stopwords')
    except LookupError:
        nltk.download('stopwords', quiet=True)
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False


def get_stop_words() -> List[str]:
    """
    Get a refined list of stop words to filter from vocabulary.
    Uses NLTK's English stopwords plus additional domain-specific words.
    
    Returns:
        List of stop words to remove
    """
    stop_words = []
    
    # Add NLTK English stopwords if available
    if NLTK_AVAILABLE:
        stop_words.extend(stopwords.words('english'))
    
    # Add additional domain-specific stop words
    additional_stopwords = [
        # Articles and pronouns
        'the', 'a', 'an', 'this', 'that', 'these', 'those',
        'i', 'you', 'he', 'she', 'it', 'we', 'they',
        'his', 'her', 'its', 'our', 'their', 'my', 'your',
        'him', 'her', 'us', 'them',
        
        # Common verbs (very frequent, low information)
        'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'having',
        'do', 'does', 'did', 'doing', 'done',
        'say', 'says', 'said', 'saying',
        'get', 'gets', 'got', 'getting',
        'go', 'goes', 'went', 'going', 'gone',
        'make', 'makes', 'made', 'making',
        'take', 'takes', 'took', 'taking', 'taken',
        'come', 'comes', 'came', 'coming',
        'see', 'sees', 'saw', 'seeing', 'seen',
        'know', 'knows', 'knew', 'knowing', 'known',
        'think', 'thinks', 'thought', 'thinking',
        'want', 'wants', 'wanted', 'wanting',
        'use', 'uses', 'used', 'using',
        'find', 'finds', 'found', 'finding',
        'give', 'gives', 'gave', 'giving', 'given',
        'tell', 'tells', 'told', 'telling',
        'work', 'works', 'worked', 'working',
        'call', 'calls', 'called', 'calling',
        'try', 'tries', 'tried', 'trying',
        'ask', 'asks', 'asked', 'asking',
        'need', 'needs', 'needed', 'needing',
        'feel', 'feels', 'felt', 'feeling',
        'become', 'becomes', 'became', 'becoming',
        'leave', 'leaves', 'left', 'leaving',
        'put', 'puts', 'putting',
        'mean', 'means', 'meant', 'meaning',
        'keep', 'keeps', 'kept', 'keeping',
        'let', 'lets', 'letting',
        'begin', 'begins', 'began', 'beginning', 'begun',
        'seem', 'seems', 'seemed', 'seeming',
        'help', 'helps', 'helped', 'helping',
        'show', 'shows', 'showed', 'showing', 'shown',
        'hear', 'hears', 'heard', 'hearing',
        'play', 'plays', 'played', 'playing',
        'run', 'runs', 'ran', 'running',
        'move', 'moves', 'moved', 'moving',
        'live', 'lives', 'lived', 'living',
        'believe', 'believes', 'believed', 'believing',
        'bring', 'brings', 'brought', 'bringing',
        'happen', 'happens', 'happened', 'happening',
        'write', 'writes', 'wrote', 'writing', 'written',
        'sit', 'sits', 'sat', 'sitting',
        'stand', 'stands', 'stood', 'standing',
        'lose', 'loses', 'lost', 'losing',
        'pay', 'pays', 'paid', 'paying',
        'meet', 'meets', 'met', 'meeting',
        'include', 'includes', 'included', 'including',
        'continue', 'continues', 'continued', 'continuing',
        'set', 'sets', 'setting',
        'learn', 'learns', 'learned', 'learning',
        'change', 'changes', 'changed', 'changing',
        'lead', 'leads', 'led', 'leading',
        'understand', 'understands', 'understood', 'understanding',
        'watch', 'watches', 'watched', 'watching',
        'follow', 'follows', 'followed', 'following',
        'stop', 'stops', 'stopped', 'stopping',
        'create', 'creates', 'created', 'creating',
        'speak', 'speaks', 'spoke', 'speaking', 'spoken',
        'read', 'reads', 'reading',
        'allow', 'allows', 'allowed', 'allowing',
        'add', 'adds', 'added', 'adding',
        'spend', 'spends', 'spent', 'spending',
        'grow', 'grows', 'grew', 'growing', 'grown',
        'open', 'opens', 'opened', 'opening',
        'walk', 'walks', 'walked', 'walking',
        'win', 'wins', 'won', 'winning',
        'offer', 'offers', 'offered', 'offering',
        'remember', 'remembers', 'remembered', 'remembering',
        'love', 'loves', 'loved', 'loving',
        'consider', 'considers', 'considered', 'considering',
        'appear', 'appears', 'appeared', 'appearing',
        'buy', 'buys', 'bought', 'buying',
        'wait', 'waits', 'waited', 'waiting',
        'serve', 'serves', 'served', 'serving',
        'die', 'dies', 'died', 'dying',
        'send', 'sends', 'sent', 'sending',
        'build', 'builds', 'built', 'building',
        'stay', 'stays', 'stayed', 'staying',
        'fall', 'falls', 'fell', 'falling', 'fallen',
        'cut', 'cuts', 'cutting',
        'reach', 'reaches', 'reached', 'reaching',
        'kill', 'kills', 'killed', 'killing',
        'raise', 'raises', 'raised', 'raising',
        'pass', 'passes', 'passed', 'passing',
        'sell', 'sells', 'sold', 'selling',
        'decide', 'decides', 'decided', 'deciding',
        'return', 'returns', 'returned', 'returning',
        'explain', 'explains', 'explained', 'explaining',
        'develop', 'develops', 'developed', 'developing',
        'carry', 'carries', 'carried', 'carrying',
        'break', 'breaks', 'broke', 'breaking', 'broken',
        'receive', 'receives', 'received', 'receiving',
        'agree', 'agrees', 'agreed', 'agreeing',
        'support', 'supports', 'supported', 'supporting',
        'hit', 'hits', 'hitting',
        'produce', 'produces', 'produced', 'producing',
        'eat', 'eats', 'ate', 'eating', 'eaten',
        'cover', 'covers', 'covered', 'covering',
        'catch', 'catches', 'caught', 'catching',
        'draw', 'draws', 'drew', 'drawing', 'drawn',
        'choose', 'chooses', 'chose', 'choosing', 'chosen',
        
        # Prepositions and conjunctions
        'of', 'in', 'on', 'at', 'to', 'for', 'with', 'by',
        'from', 'up', 'about', 'into', 'through', 'during',
        'including', 'against', 'among', 'throughout', 'despite',
        'towards', 'upon', 'concerning', 'to', 'of', 'in',
        'and', 'or', 'but', 'if', 'than', 'because', 'while',
        
        # Adverbs
        'very', 'just', 'now', 'then', 'here', 'there', 'where',
        'when', 'why', 'how', 'also', 'only', 'even', 'still',
        'back', 'well', 'too', 'so', 'more', 'most', 'much',
        'many', 'some', 'any', 'all', 'both', 'each', 'few',
        'other', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'can', 'will',
        'just', 'should', 'now',
        
        # News agencies and common news terms
        'reuters', 'ap', 'associated', 'press', 'agency',
        'quot', 'quote', 'quoted', 'says', 'said',
        
        # Single characters and numbers
        's', 't', 'm', 'd', 'll', 've', 're',
        
        # Days of week (too common in news)
        'monday', 'tuesday', 'wednesday', 'thursday', 'friday',
        'saturday', 'sunday', 'mon', 'tue', 'wed', 'thu', 'fri',
        'sat', 'sun',
        
        # HTML/XML artifacts
        'lt', 'gt', 'amp', 'nbsp',
    ]
    
    # Combine and remove duplicates
    stop_words = list(set(stop_words + additional_stopwords))
    
    return stop_words


def filter_stop_words(
    term_doc_matrix: csr_matrix,
    vocabulary: List[str],
    stop_words: Optional[List[str]] = None
) -> Tuple[csr_matrix, List[str]]:
    """
    Filter stop words from term-document matrix and vocabulary.
    
    Args:
        term_doc_matrix: Term-document matrix (vocab_size, num_documents)
        vocabulary: List of vocabulary words
        stop_words: Optional list of stop words. If None, uses default list.
    
    Returns:
        Filtered term-document matrix and vocabulary
    """
    if stop_words is None:
        stop_words = get_stop_words()
    
    # Convert to set for faster lookup
    stop_words_set = set(word.lower() for word in stop_words)
    
    # Find indices to keep (words not in stop words, not digits, not single chars)
    keep_indices = []
    for i, word in enumerate(vocabulary):
        word_lower = word.lower()
        
        # Skip if in stop words
        if word_lower in stop_words_set:
            continue
        
        # Skip if contains only digits (e.g., '39', '2020', '123')
        if word.isdigit() or (word.replace('.', '').replace('-', '').isdigit()):
            continue
        
        # Skip single character tokens (except 'a' and 'i' which are already in stopwords)
        if len(word) == 1:
            continue
        
        # Skip if token is mostly punctuation or special characters (no letters)
        if not re.search(r'[a-zA-Z]', word):
            continue
        
        keep_indices.append(i)
    
    if len(keep_indices) == len(vocabulary):
        # No filtering needed
        return term_doc_matrix, vocabulary
    
    # Filter vocabulary
    filtered_vocabulary = [vocabulary[i] for i in keep_indices]
    
    # Filter matrix (keep only rows corresponding to kept vocabulary)
    filtered_matrix = term_doc_matrix[keep_indices, :]
    
    removed_count = len(vocabulary) - len(filtered_vocabulary)
    print(f"Filtering stop words, digits, and single chars: {len(vocabulary)} -> {len(filtered_vocabulary)} tokens")
    print(f"Removed {removed_count} tokens (stop words, digits, single chars)")
    
    return filtered_matrix, filtered_vocabulary


def load_vocabulary_from_lab2(
    lab2_dir: str,
    filter_stopwords: bool = True
) -> Tuple[Dict, csr_matrix, List[str]]:
    """
    Load term-document matrix and vocabulary from Lab 2.
    
    Args:
        lab2_dir: Path to lab2 directory
        filter_stopwords: Whether to filter stop words
    
    Returns:
        Tuple of (dictionary data, term-document matrix, vocabulary list)
        Matrix format: (vocab_size, num_documents)
    """
    # Paths to Lab 2 output files
    dict_file = os.path.join(lab2_dir, 'assets\dictionaries', 'token_dictionary.json')
    matrix_file = os.path.join(lab2_dir, 'assets\matrices', 'term_document_matrix.pkl')
    
    if not os.path.exists(dict_file):
        raise FileNotFoundError(
            f"Token dictionary not found at {dict_file}. "
            "Please run Lab 2 Task 1 first."
        )
    
    if not os.path.exists(matrix_file):
        raise FileNotFoundError(
            f"Term-document matrix not found at {matrix_file}. "
            "Please run Lab 2 Task 1 first."
        )
    
    # Load dictionary (JSON format)
    print(f"Loading dictionary from {dict_file}...")
    with open(dict_file, 'r', encoding='utf-8') as f:
        dict_data = json.load(f)
    
    # Load matrix (pickle format)
    print(f"Loading matrix from {matrix_file}...")
    with open(matrix_file, 'rb') as f:
        term_doc_matrix = pickle.load(f)
    
    # Matrix from lab2 is (vocab_size, num_documents)
    vocab_size, num_docs = term_doc_matrix.shape
    print(f"Matrix format: (vocab_size, num_documents) = ({vocab_size}, {num_docs})")
    print(f"Matrix shape: {term_doc_matrix.shape} (vocab_size={vocab_size}, num_docs={num_docs})")
    
    # Build vocabulary list from dictionary
    # Lab 2 dictionary structure: index_to_token maps index -> token
    index_to_token = dict_data.get('index_to_token', {})
    dict_vocab_size = dict_data.get('vocab_size', len(index_to_token))
    
    print(f"Dictionary vocab_size: {dict_vocab_size}")
    
    # Build vocabulary list by iterating through indices
    # Convert string keys to integers (JSON keys are strings)
    vocabulary = []
    for i in range(vocab_size):
        # Try both string and int keys (JSON may have string keys)
        token = index_to_token.get(str(i)) or index_to_token.get(i)
        if token:
            vocabulary.append(token)
        else:
            # Fallback if index is missing
            vocabulary.append(f"word_{i}")
    
    # Ensure vocabulary list has correct size
    if len(vocabulary) < vocab_size:
        vocabulary.extend([f"word_{i}" for i in range(len(vocabulary), vocab_size)])
    elif len(vocabulary) > vocab_size:
        vocabulary = vocabulary[:vocab_size]
    
    # Apply stop word filtering if requested
    if filter_stopwords:
        print("Filtering stop words...")
        term_doc_matrix, vocabulary = filter_stop_words(term_doc_matrix, vocabulary)
        print(f"After filtering: matrix shape={term_doc_matrix.shape}, vocabulary size={len(vocabulary)}")
    
    return dict_data, term_doc_matrix, vocabulary


def split_train_test(
    term_doc_matrix: csr_matrix,
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[csr_matrix, csr_matrix]:
    """
    Split term-document matrix into train and test sets.
    
    The matrix format is (vocab_size, num_documents), so we split by columns (documents).
    
    Args:
        term_doc_matrix: Term-document matrix (vocab_size, num_documents)
        test_size: Proportion of documents for test set
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (train_matrix, test_matrix)
        Both matrices have shape (vocab_size, num_docs_train/test)
    """
    vocab_size, num_docs = term_doc_matrix.shape
    print(f"Splitting matrix: shape={term_doc_matrix.shape} (vocab_size={vocab_size}, num_docs={num_docs})")
    
    # Set random seed
    np.random.seed(random_state)
    
    # Generate random permutation of document indices
    doc_indices = np.random.permutation(num_docs)
    
    # Calculate split point
    split_idx = int(num_docs * (1 - test_size))
    
    # Split indices
    train_indices = doc_indices[:split_idx]
    test_indices = doc_indices[split_idx:]
    
    # Split matrix by columns (documents)
    train_matrix = term_doc_matrix[:, train_indices]
    test_matrix = term_doc_matrix[:, test_indices]
    
    print(f"Train matrix: {train_matrix.shape} (vocab_size={train_matrix.shape[0]}, docs={train_matrix.shape[1]})")
    print(f"Test matrix: {test_matrix.shape} (vocab_size={test_matrix.shape[0]}, docs={test_matrix.shape[1]})")
    
    return train_matrix, test_matrix

