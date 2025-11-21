import numpy as np
from gensim.models import Word2Vec
from nltk import pos_tag
from nltk.stem import SnowballStemmer, WordNetLemmatizer

from utils import tokenize
from utils import get_wordnet_pos


stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()


def document_to_vector(text: str, w2v_model: Word2Vec, vector_size: int) -> np.ndarray:
    tokenized_sents = tokenize(text)

    pos_tagged_sents = []
    for sent in tokenized_sents:
        if not sent:
            continue
        pos_tags = pos_tag(sent, lang='eng')
        lemmatized = []
        for word, pos in pos_tags:
            if word.isalpha():
                wn_pos = get_wordnet_pos(pos)
                lemma = lemmatizer.lemmatize(word.lower(), wn_pos)
                lemmatized.append(lemma)
        pos_tagged_sents.append(lemmatized)

    sent_vectors = []
    for sent in pos_tagged_sents:
        word_vectors = []
        for word in sent:
            if word in w2v_model.wv:
                word_vectors.append(w2v_model.wv[word])
        if word_vectors:
            sent_vec = np.mean(word_vectors, axis=0)
            sent_vectors.append(sent_vec)

    # 4. Вектор документа
    if sent_vectors:
        doc_vec = np.mean(sent_vectors, axis=0)
    else:
        doc_vec = np.zeros(vector_size)

    return doc_vec
