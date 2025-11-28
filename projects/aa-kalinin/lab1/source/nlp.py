import re
from typing import List, Tuple

import nltk
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet


class Tokenizer:

    def __init__(self):
        self.email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
        self.url_pattern = r'(?:https?://)?(?:www\.)?[a-zA-Z0-9][a-zA-Z0-9-]*\.[a-zA-Z]{2,}(?:/[^\s]*)?'
        self.phone_patterns = [
            r'\+?1?[-.\s]?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}',  # (xxx) xxx-xxxx
            r'\d{3}[-.\s]\d{4}',  # xxx-xxxx (local)
            r'1-800-[A-Z0-9]{3,}(?:-[A-Z0-9]+)*',  # 1-800 numbers
        ]
        self.abbreviations = {
            'Dr', 'Mr', 'Mrs', 'Ms', 'Prof', 'Jr', 'Sr', 'Inc', 'Corp', 'Ltd',
            'Co', 'etc', 'vs', 'St', 'Ave', 'Blvd', 'Rd', 'Apt', 'No', 'Vol',
            'pp', 'ed', 'al', 'Jan', 'Feb', 'Mar', 'Apr', 'Jun', 'Jul', 'Aug',
            'Sep', 'Oct', 'Nov', 'Dec', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri',
            'Sat', 'Sun', 'U', 'S', 'A', 'N', 'E', 'W', 'Gen', 'Gov', 'Rep',
            'Sen', 'Rev', 'Capt', 'Lt', 'Col', 'Sgt', 'Fig', 'cf', 'e', 'i',
            'a', 'p', 'm'
        }
        self.emoticon_pattern = r'(?:[:;=8][-\'^]?[)(\[\]DPpOo3><|/\\]+|[)(\[\]DPpOo3><]+[-\'^]?[:;=8]|<3|</3|\^_*\^|o_o|O_O|-_-|>_<|T_T|;_;|:\'[)(])'
        self.currency_pattern = r'[$€£¥]\s*\d+(?:,\d{3})*(?:\.\d+)?(?:\s*(?:million|billion|trillion|m|b|k|M|B|K))?|\d+(?:,\d{3})*(?:\.\d+)?\s*(?:dollars?|euros?|pounds?|yen)'
        self.date_patterns = [
            r'\d{1,2}/\d{1,2}/\d{2,4}',  # MM/DD/YYYY
            r'\d{1,2}-\d{1,2}-\d{2,4}',  # MM-DD-YYYY
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
        ]
        self.time_pattern = r'\d{1,2}:\d{2}(?::\d{2})?(?:\s*[AaPp][Mm])?'
        self._build_tokenizer_regex()

    def _build_tokenizer_regex(self):
        """Build the master regex pattern for tokenization."""
        patterns = [
            f'({self.email_pattern})',
            f'({self.emoticon_pattern})',
            f'({self.currency_pattern})',
            f'({self.time_pattern})',
        ]

        for phone_pat in self.phone_patterns:
            patterns.append(f'({phone_pat})')

        for date_pat in self.date_patterns:
            patterns.append(f'({date_pat})')

        abbrev_pattern = '|'.join(re.escape(a) for a in self.abbreviations)
        patterns.append(f'((?:{abbrev_pattern})\\.)')
        patterns.append(r"(\w+(?:n't|'ll|'re|'ve|'m|'d|'s))")
        patterns.append(r'([#@]\w+)')
        patterns.append(r'(\d+(?:\.\d+)?%)')
        patterns.append(r'(\d+(?:,\d{3})*(?:\.\d+)?)')
        patterns.append(r'(\w+)')
        patterns.append(r'([^\s\w])')

        self.tokenize_pattern = re.compile('|'.join(patterns), re.IGNORECASE)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text while preserving complex patterns."""
        if not text:
            return []

        # Normalize encoded characters
        text = text.replace('#36;', '$')
        text = text.replace('&lt;', '<')
        text = text.replace('&gt;', '>')
        text = text.replace('&amp;', '&')
        text = text.replace('\\n', ' ')
        text = text.replace('\\', '')

        tokens = []
        for match in self.tokenize_pattern.finditer(text):
            token = match.group().strip()
            if token:
                tokens.append(token)

        return tokens


class SentenceSegmenter:
    """Sentence segmentation using regular expressions."""

    def __init__(self):
        self.abbreviations = {
            'Dr', 'Mr', 'Mrs', 'Ms', 'Prof', 'Jr', 'Sr', 'Inc', 'Corp', 'Ltd',
            'Co', 'etc', 'vs', 'St', 'Ave', 'Blvd', 'Rd', 'Apt', 'No', 'Vol',
            'pp', 'ed', 'al', 'Jan', 'Feb', 'Mar', 'Apr', 'Jun', 'Jul', 'Aug',
            'Sep', 'Oct', 'Nov', 'Dec', 'U', 'S', 'A', 'e', 'i', 'g'
        }

        abbrev_pattern = '|'.join(re.escape(a) for a in self.abbreviations)
        self.abbrev_regex = re.compile(f'^({abbrev_pattern})$', re.IGNORECASE)

    def segment(self, text: str) -> List[str]:
        """Segment text into sentences."""
        if not text:
            return []

        text = re.sub(r'\s+', ' ', text.strip())
        sentences = []
        current = []
        words = text.split()

        for i, word in enumerate(words):
            current.append(word)
            if self._is_sentence_end(word, words, i):
                sentence = ' '.join(current).strip()
                if sentence:
                    sentences.append(sentence)
                current = []

        if current:
            sentence = ' '.join(current).strip()
            if sentence:
                sentences.append(sentence)

        return sentences

    def _is_sentence_end(self, word: str, words: List[str], index: int) -> bool:
        """Check if a word ends a sentence."""
        if not word:
            return False

        if not any(word.endswith(p) for p in ['.', '!', '?', '."', '?"', '!"', ".'", "?'", "!'"]):
            return False

        base_word = re.sub(r'[.!?"\'\s]+$', '', word)

        if self.abbrev_regex.match(base_word):
            return False

        if len(base_word) == 1 and base_word.isupper():
            return False

        if index + 1 < len(words):
            next_word = words[index + 1]
            if next_word and next_word[0].isupper():
                return True
            if next_word and next_word[0].islower():
                return False

        return True


class TextProcessor:
    """Main text processing pipeline combining tokenization, stemming, and lemmatization."""

    def __init__(self):
        self.tokenizer = Tokenizer()
        self.segmenter = SentenceSegmenter()
        self.stemmer = SnowballStemmer('english')
        self.lemmatizer = WordNetLemmatizer()

    def _get_wordnet_pos(self, word: str) -> str:
        """Get WordNet POS tag for a word using heuristics."""
        word_lower = word.lower()

        if word_lower.endswith(('ing', 'ed', 'ize', 'ise', 'ify', 'ate')):
            return wordnet.VERB
        if word_lower.endswith(('able', 'ible', 'ful', 'less', 'ous', 'ive', 'al', 'ish')):
            return wordnet.ADJ
        if word_lower.endswith('ly'):
            return wordnet.ADV
        if word_lower.endswith(('tion', 'ness', 'ment', 'ity', 'ty', 'ism', 'er', 'or')):
            return wordnet.NOUN

        return wordnet.NOUN

    def stem(self, token: str) -> str:
        """Apply stemming to a token."""
        if not token or not token.isalpha():
            return token.lower() if token else token
        return self.stemmer.stem(token)

    def lemmatize(self, token: str, pos: str = None) -> str:
        """Apply lemmatization to a token."""
        if not token or not token.isalpha():
            return token.lower() if token else token

        if pos is None:
            pos = self._get_wordnet_pos(token)

        return self.lemmatizer.lemmatize(token.lower(), pos)

    def process_sentence(self, sentence: str) -> List[Tuple[str, str, str]]:
        """Process a sentence and return list of (token, stem, lemma) tuples."""
        tokens = self.tokenizer.tokenize(sentence)
        results = []

        try:
            pos_tags = nltk.pos_tag(tokens)
        except Exception:
            pos_tags = [(t, 'NN') for t in tokens]

        for token, pos_tag in pos_tags:
            if pos_tag.startswith('V'):
                wn_pos = wordnet.VERB
            elif pos_tag.startswith('J'):
                wn_pos = wordnet.ADJ
            elif pos_tag.startswith('R'):
                wn_pos = wordnet.ADV
            else:
                wn_pos = wordnet.NOUN

            stem = self.stem(token)
            lemma = self.lemmatize(token, wn_pos)
            results.append((token, stem, lemma))

        return results

    def process_text(self, text: str) -> List[List[Tuple[str, str, str]]]:
        """Process entire text and return sentences with annotations."""
        sentences = self.segmenter.segment(text)
        return [self.process_sentence(s) for s in sentences]

