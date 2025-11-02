from nltk.stem import SnowballStemmer
import warnings
warnings.filterwarnings('ignore')

stemmer = SnowballStemmer('english')

def stem_word(word):
    try:
        return stemmer.stem(word.lower())
    except:
        return word.lower()

def stem_text(text):
    words = text.split()
    stemmed_words = [stem_word(word) for word in words]
    return ' '.join(stemmed_words)
