import re
import ssl
from typing import List

import nltk
from nltk import pos_tag
from nltk.corpus import wordnet, stopwords
from nltk.stem import WordNetLemmatizer


def _setup_nltk():
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
        'stopwords',
    ]
    for resource in required_resources:
        try:
            nltk.download(resource, quiet=True)
        except:
            pass


def get_wordnet_pos(treebank_tag: str) -> str:
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


def tokenize_and_lemmatize(text: str) -> List[str]:
    _setup_nltk()
    
    sentences = nltk.sent_tokenize(text)
    tokens = []
    for sent in sentences:
        words = nltk.word_tokenize(sent)
        tokens.extend(words)
    
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    lemmatized = []
    pos_tags = pos_tag(tokens, lang='eng')
    for word, pos in pos_tags:
        if word.isalpha() and word.lower() not in stop_words:
            wn_pos = get_wordnet_pos(pos)
            lemma = lemmatizer.lemmatize(word.lower(), wn_pos)
            lemmatized.append(lemma)
    
    return lemmatized

