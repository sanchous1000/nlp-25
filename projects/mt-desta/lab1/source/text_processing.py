import re
import nltk
from nltk import pos_tag

nltk.download('punkt_tab', quiet=True)      
nltk.download('wordnet', quiet=True)    
nltk.download('omw-1.4', quiet=True) 
nltk.download('averaged_perceptron_tagger_eng', quiet=True)

from nltk.stem import WordNetLemmatizer, SnowballStemmer

class TextProcessing:

    def __init__(self, text):
        self.text = text
        self.patterns = {
            "abbrevation": r"\b(?:[A-Za-z]\.){2,}",
            "abbrevation2":r"\b(?:Mr|Mrs|Ms|Miss|Dr|Prof|Sir|Madam|Mx)\.\s+[A-Z][a-z]+\b",
            "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
            "url": r"https?://[^\s/$.?#].[^\s]*",
            "telephone": r"\+?\d{1,3}[\s-]?\(?\d{1,4}\)?[\s-]?\d{1,4}[\s-]?\d{1,9}",
        }
            
    def tokenize(self):

        placeholders = {}
        counter = 0
        
        for name, pattern in self.patterns.items():
            matches = re.findall(pattern, self.text, flags=re.IGNORECASE)
            for match in set(matches):
                placeholder = f"__{name}_{counter}__"
                self.text = self.text.replace(match, placeholder)
                placeholders[placeholder] = match
                counter += 1


        sentences = re.split(r'[.!?]+', self.text)

        sentences = [sentence.strip() for sentence in sentences if sentence and sentence.strip()]

        p = re.compile(r"\w+(?:'\w+)?|[^\w\s]")

        tokens_all = [
            [
            placeholders.get(token, token) for token in p.findall(sentence)
            ] 
            for sentence in sentences
        ]

        self.tokens_all = tokens_all

    def get_wordnet_pos(self,tag):
            if tag.startswith('J'):  
                return 'a'
            elif tag.startswith('V'):  
                return 'v'
            elif tag.startswith('N'):  
                return 'n'
            elif tag.startswith('R'):  
                return 'r'
            else:
                return 'n' 

    def lemmatize(self, tokens):
        lemmatizer = WordNetLemmatizer()

        tagged_tokens = pos_tag(tokens)
 
        lemmatized_sentence = []

        for word, tag in tagged_tokens:
            if word.lower() == 'are' or word.lower() in ['is', 'am']:
                lemmatized_sentence.append(word)  
            else:
                lemmatized_sentence.append(lemmatizer.lemmatize(word, self.get_wordnet_pos(tag)))

        return lemmatized_sentence

    def stemme(self, tokens):
        stemmer = SnowballStemmer('english')

        tagged_tokens = pos_tag(tokens)

        stemmed_sentence = []

        for word, pos in tagged_tokens:
            if not word.isalpha():
                stemmed_sentence.append(word)
            else:
                stemmed_sentence.append(stemmer.stem(word.lower()))
        return stemmed_sentence