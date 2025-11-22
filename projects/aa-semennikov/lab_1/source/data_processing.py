import re
import nltk
from nltk import pos_tag
from nltk.corpus import wordnet, stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer


nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


COMPLEX_PATTERNS = [
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # email
    r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3,4}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # phone
    r'\b(?:Dr|Mr|Mrs|Ms|Prof)\.\s+[A-Z][a-zA-Z]*',  # honorifics
    r'https?://[^\s<>"{}|\\^`\[\]]+',  # url
]


def tokenize(text):
    stop_word = set(stopwords.words('english'))
    # Создаём словарь для сопоставления плейсхолдеров с оригинальными значениями
    placeholder_map = {}
    modified_text = text

    for pattern in COMPLEX_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            matched_str = match.group(0)
            if matched_str not in placeholder_map:
                placeholder = f"__TOKEN_{len(placeholder_map)}__"
                placeholder_map[placeholder] = matched_str
                modified_text = modified_text.replace(matched_str, placeholder)

    sentences = [s.strip() for s in re.split(r'[.!?]+', modified_text) if s.strip()]

    res = []
    for sent in sentences:
        # Извлекаем токены: либо плейсхолдеры (__TOKEN_N__), либо обычные токены, либо токены с апострофами
        tokens = re.findall(r"__TOKEN_\d+__|\w+(?:'\w+)?", sent)
        filtered_tokens = []
        for tok in tokens:
            original_tok = placeholder_map.get(tok, tok)
            # Сохраняем токен, если это плейсхолдер или не стоп-слово
            if tok in placeholder_map or original_tok.lower() not in stop_word:
                filtered_tokens.append(original_tok)
        res.append(filtered_tokens)

    # Возвращаем предложения: [[список токенов], [список токенов], ...]
    return res


def process_tokens(sentences_list, stemmer, lemmatizer):
    res = []
    for token_list in sentences_list:
        if not token_list:
            continue
        pos_tags = pos_tag(token_list, lang='eng')
        sentence_result = []
        for word, pos in pos_tags:
            if not word.isalpha():
                stem = word
                lemma = word
            else:
                stem = stemmer.stem(word.lower())
                wn_pos = get_wordnet_pos(pos)
                lemma = lemmatizer.lemmatize(word.lower(), wn_pos)
            sentence_result.append((word, stem, lemma))
        res.append(sentence_result)
    return res