import re

from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.stem import SnowballStemmer, WordNetLemmatizer

# Предопределённые паттерны для сложных токенов
COMPLEX_TOKEN_PATTERNS = [
    r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # email
    r'(?:\+?\d{1,3}[-.\s]?)?\(?\d{3,4}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # phone
    r'\b(?:Dr|Mr|Mrs|Ms|Prof)\.\s+[A-Z][a-zA-Z]*',  # titles
]


def tokenize(text: str) -> list[list[str]]:
    """
    Мапа для хранения сложных токенов, пример:
    "__TOKEN_0__": "Dr. Johnson",
    "__TOKEN_1__": "john.doe@company.com",
    "__TOKEN_2__": "555-123-4567"
    """
    placeholder_map: dict[str, str] = {}
    modified_text = text

    for pattern in COMPLEX_TOKEN_PATTERNS:
        for match in re.finditer(pattern, text, flags=re.IGNORECASE):
            matched_str = match.group(0)
            if matched_str not in placeholder_map:
                placeholder = f"__TOKEN_{len(placeholder_map)}__"
                placeholder_map[placeholder] = matched_str
                modified_text = modified_text.replace(matched_str, placeholder)

    sentences = [s.strip() for s in re.split(r'[.!?]+', modified_text) if s.strip()]

    TOKEN_REGEX = r"__TOKEN_\d+__"
    WORD_REGEX = r"\w+(?:'\w+)?"

    result = []
    for sent in sentences:
        tokens = re.findall(rf"{TOKEN_REGEX}|{WORD_REGEX}", sent)
        result.append([placeholder_map.get(tok, tok) for tok in tokens])

    return result


def process_tokens(
    sentences: list[list[str]], # Список предложений, где каждое предложение - список слов
    stemmer: SnowballStemmer, # Алгоритм стемминга (например, Porter или Snowball)
    lemmatizer: WordNetLemmatizer, # Лемматизатор из WordNet
) -> list[list[tuple[str, str, str]]]:
    """
    Для каждого токена вычисляет стемму и лемму.
    Возвращает структуру: [[(token, stem, lemma), ...], ...]
    """
    result = []
    for token_list in sentences:
        if not token_list:
            continue # Скипаем пустые предложения
        pos_tags = pos_tag(token_list, lang='eng') # определение части речи
        sentence_result = []
        for word, pos in pos_tags:
            if not word.isalpha():
                # Если наше слово "странное" (содержит цифры, пунктуацию и т.д.), то пропускаем без обрабтки
                stem = word
                lemma = word
            else:
                stem = stemmer.stem(word.lower())
                wn_pos = treebank_to_wordnet_tag(pos) # преобразовываем теги в формат, знакомый WordNetLemmatizer
                lemma = lemmatizer.lemmatize(word.lower(), wn_pos)
            sentence_result.append((word, stem, lemma))
        result.append(sentence_result)
    return result


def treebank_to_wordnet_tag(treebank_tag: str) -> str:
    if treebank_tag.startswith('JJ'):  # Прилагательные (JJ, JJR, JJS)
        return wordnet.ADJ
    elif treebank_tag.startswith('VB'):  # Глаголы (VB, VBD, VBG, VBN, VBP, VBZ)
        return wordnet.VERB
    elif treebank_tag.startswith('NN'):  # Существительные (NN, NNS, NNP, NNPS)
        return wordnet.NOUN
    elif treebank_tag.startswith('RB'):  # Наречия (RB, RBR, RBS)
        return wordnet.ADV
    else:  # По умолчанию считаем существительным
        return wordnet.NOUN