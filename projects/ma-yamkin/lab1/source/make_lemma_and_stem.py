from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from nltk.corpus import wordnet
import nltk

# Загрузка необходимых ресурсов
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger_eng')
nltk.download('wordnet')
nltk.download('omw-1.4')


stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()

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


def process_tokens(sentences_list):
    lemma_stem_list = []
    for token_list in sentences_list:
        pos_tags = pos_tag(token_list)
        result = []
        for word, pos in pos_tags:
            if not word.isalpha():
                stem = word
                lemma = word
            else:
                stem = stemmer.stem(word.lower())
                wn_pos = get_wordnet_pos(pos)
                lemma = lemmatizer.lemmatize(word.lower(), wn_pos)
            result.append((word, stem, lemma))
        lemma_stem_list.append(result)
    return lemma_stem_list