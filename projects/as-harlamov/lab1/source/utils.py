import re
import ssl

import nltk
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import SnowballStemmer, WordNetLemmatizer


def _setup_nltk():
    """Настройка SSL и загрузка необходимых NLTK-ресурсов."""
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    required_resources = [
        'punkt',
        'averaged_perceptron_tagger_eng',
        'wordnet',
        'omw-1.4',
    ]
    for resource in required_resources:
        nltk.download(resource, quiet=True)


def get_wordnet_pos(treebank_tag: str) -> str:
    """Преобразует POS-тег из Penn Treebank в WordNet-тег."""
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


# Предопределённые паттерны для сложных токенов
COMPLEX_TOKEN_PATTERNS = [
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # email
    r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3,4}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # phone
    r'\b(?:Dr|Mr|Mrs|Ms|Prof)\.\s+[A-Z][a-zA-Z]*',  # titles
]


def tokenize(text: str) -> list[list[str]]:
    """
    Токенизирует текст на предложения и токены, сохраняя сложные сущности как единые токены.
    Возвращает список предложений, где каждое предложение — список токенов.
    """
    # Создаём уникальные плейсхолдеры для сложных токенов
    placeholder_map: dict[str, str] = {}
    modified_text = text

    for pattern in COMPLEX_TOKEN_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            matched_str = match.group(0)
            if matched_str not in placeholder_map:
                placeholder = f"__TOKEN_{len(placeholder_map)}__"
                placeholder_map[placeholder] = matched_str
                modified_text = modified_text.replace(matched_str, placeholder)

    # Сегментация на предложения
    sentences = [s.strip() for s in re.split(r'[.!?]+', modified_text) if s.strip()]

    # Базовая токенизация (сохраняем плейсхолдеры как токены)
    result = []
    for sent in sentences:
        # Извлекаем слова, апострофы и плейсхолдеры
        tokens = re.findall(r"__TOKEN_\d+__|\w+(?:'\w+)?", sent)
        # Восстанавливаем оригинальные значения
        result.append([placeholder_map.get(tok, tok) for tok in tokens])

    return result


def process_tokens(
    sentences_list: list[list[str]],
    stemmer: SnowballStemmer,
    lemmatizer: WordNetLemmatizer,
) -> list[list[tuple[str, str, str]]]:
    """
    Для каждого токена вычисляет стемму и лемму.
    Возвращает структуру: [[(token, stem, lemma), ...], ...]
    """
    result = []
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
        result.append(sentence_result)
    return result
