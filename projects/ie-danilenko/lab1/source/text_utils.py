import re
import nltk
from nltk.tokenize import sent_tokenize
from lemmatization import process_word
import warnings
warnings.filterwarnings('ignore')

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    nltk.download('punkt_tab')

def segment_sentences(text):
    sentences = sent_tokenize(text)
    return sentences

def tokenize_text(text):
    patterns = [
        # Email адреса
        r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        # Телефонные номера
        r'\+?[1-9]\d{1,14}',  # Международные номера
        r'\+?7[- ]?\d{3}[- ]?\d{3}[- ]?\d{2}[- ]?\d{2}',  # +7-901-000-00-00
        r'8\(?\d{3}\)?\d{7}',  # 8(918)3213412
        r'\+1[- ]?\d{3}[- ]?\d{3}[- ]?\d{4}',  # +1-555-123-4567
        r'\(\d{3}\)\s?\d{3}[- ]?\d{4}',  # (555) 987-6543
        r'\d{3}[- ]?\d{3}[- ]?\d{4}',  # 555-123-4567
        # Адреса
        r'\d+\s+[A-Za-z\s]+(?:St|Ave|Rd|Blvd|Dr|Ln|Way|Pl|Ct)\.?\s*,\s*[^,]+,\s*[A-Z]{2}\s+\d{5}(?:,\s*[A-Z]{2,3})?',
        r'\d+\s+[A-Za-z\s]+(?:St|Ave|Rd|Blvd|Dr|Ln|Way|Pl|Ct)\.?\s*,\s*[^,]+,\s*[A-Z]{2}\s+\d{5}',
        # Эмотиконы
        r'[;:][-]?[)(DPp]',
        r'[)(DPp][-]?[;:]',
        # Математические выражения
        r'[a-zA-Z]\s*[=<>]\s*[a-zA-Z0-9]',
        r'[a-zA-Z0-9]+\s*[+\-*/]\s*[a-zA-Z0-9]+',
        r'\([a-zA-Z0-9+\-*/^]+\d+\)',
        # Сокращения
        r'\b[A-Za-z]+\.\s*[A-Za-z]+',
        r'\b(?:Dr|Mr|Mrs|Ms|Prof|Inc|Ltd|Corp)\.',
        # URL
        r'https?://[^\s]+',
        # Обычные слова и знаки препинания
        r'\b\w+\b',
        r'[^\w\s]'
    ]
    
    combined_pattern = '|'.join(patterns)
    
    tokens = re.findall(combined_pattern, text, re.IGNORECASE)
    
    tokens = [token.strip() for token in tokens if token.strip()]
    
    return tokens

def process_text(text):
    sentences = segment_sentences(text)
    
    result = []
    for sentence in sentences:
        tokens = tokenize_text(sentence)
        
        for token in tokens:
            stem, lemma = process_word(token)
            
            result.append({
                'token': token,
                'stem': stem,
                'lemma': lemma
            })
    
    return result
