import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
from stemming import stem_word
import warnings
warnings.filterwarnings('ignore')

try:
    nltk.data.find('corpora/wordnet')
except LookupError:
    nltk.download('wordnet')

try:
    nltk.data.find('corpora/omw-1.4')
except LookupError:
    nltk.download('omw-1.4')

try:
    nltk.data.find('taggers/averaged_perceptron_tagger')
except LookupError:
    nltk.download('averaged_perceptron_tagger')

lemmatizer = WordNetLemmatizer()

def get_wordnet_pos(word):
    tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"J": wordnet.ADJ,
                "N": wordnet.NOUN,
                "V": wordnet.VERB,
                "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def lemmatize_word(word):
    try:
        pos = get_wordnet_pos(word)
        return lemmatizer.lemmatize(word.lower(), pos)
    except:
        return word.lower()

def process_word(word):
    stem = stem_word(word)
    lemma = lemmatize_word(word)
    return stem, lemma
