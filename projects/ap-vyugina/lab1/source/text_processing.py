import re
from typing import List

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer, WordNetLemmatizer


class TextProcessor:
    """Класс для обработки текста: сегментация, токенизация, стемминг, лемматизация"""
    
    def __init__(self):
        self.stemmer = SnowballStemmer('english')
        self.lemmatizer = WordNetLemmatizer()

        self.patterns = {
            # Электронная почта
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            
            # Телефонные номера (различные форматы)
            'phone': r'(?:\+?[1-9]\d{0,3}[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{2,4}(?:[-.\s]?[0-9]+)?',
            
            # URL и веб-адреса
            'url': r'https?://[^\s<>"{}|\\^`\[\]]+',
            
            # Валютные суммы
            'currency': r'\$[\d,]+(?:\.\d{2})?|\d+\.\d{2}\$',
            
            # Сокращения (Dr., Mr., St., Inc., etc.)
            'abbreviation': r'\b(?:Dr|Mr|Mrs|Ms|Prof|St|Inc|Corp|Ltd|Co|etc|vs|e\.g|i\.e)\.',
            
            # Сокращения с последующим словом (St. Mary's, Dr. Smith)
            'abbreviation_with_word': r'\b(?:Dr|Mr|Mrs|Ms|Prof|St|Inc|Corp|Ltd|Co)\.\s+[A-Z][a-zA-Z]*(?:\'[a-z]+)?',
            
        }
        
        self.compiled_patterns = {
            name: re.compile(pattern, re.IGNORECASE)
            for name, pattern in self.patterns.items()
        }
        
        # Порядок обработки паттернов (от более сложных к простым)
        self.pattern_order = [
            'abbreviation_with_word',  # Сначала сокращения с словами
            'abbreviation',  # Затем простые сокращения
            'email',
            'phone', 
            'url',
            'currency',
        ]

        self.stop_words = set(stopwords.words('english'))
    
    def segment_sentences(self, text: str) -> List[str]:
        text = re.sub(r'\s+', ' ', text.strip())
        
        abbreviations = ['Dr', 'Mr', 'Mrs', 'Ms', 'Prof', 'St', 'Inc', 'Corp', 'Ltd', 'Co', 'etc', 'vs', 'e.g', 'i.e']
        
        # Заменяем точки после сокращений на специальный маркер
        for abbrev in abbreviations:
            pattern = r'\b' + re.escape(abbrev) + r'\.'
            text = re.sub(pattern, abbrev + '<ABBREV>', text)
        
        # Разделяем по концам предложений (точка, восклицательный знак, вопросительный знак)
        sentence_endings = r'(?<=[.!?])\s+(?=[A-Z])'
        sentences = re.split(sentence_endings, text)
        
        # Восстанавливаем точки после сокращений
        sentences = [re.sub(r'<ABBREV>', '.', sent.strip()) for sent in sentences]
        sentences = [sent for sent in sentences if sent.strip()]
        
        return sentences
    
    def tokenize_text(self, text: str) -> List[str]:

        tokens = []
        remaining_text = text
        
        # Обрабатываем сложные случаи в порядке приоритета
        while remaining_text.strip():
            found_match = False
            
            # Проверяем паттерны в заданном порядке
            for pattern_name in self.pattern_order:
                if pattern_name in self.compiled_patterns:
                    pattern = self.compiled_patterns[pattern_name]
                    match = pattern.search(remaining_text)
                    if match:
                        # Добавляем текст до совпадения как обычные токены
                        before_match = remaining_text[:match.start()].strip()
                        if before_match:
                            tokens.extend(self._tokenize_simple(before_match))
                        
                        # Добавляем найденный токен (сложные случаи сохраняем)
                        tokens.append(match.group())
                        
                        # Обновляем оставшийся текст
                        remaining_text = remaining_text[match.end():]
                        found_match = True
                        break
            
            if not found_match:
                # Если не найдено совпадений, токенизируем оставшийся текст
                tokens.extend(self._tokenize_simple(remaining_text))
                break
        
        # Фильтруем только пустые токены (пунктуация уже удалена в _tokenize_simple)
        filtered_tokens = [token for token in tokens if token.strip()]
        
        return filtered_tokens
    
    def _tokenize_simple(self, text: str) -> List[str]:
        # Разделяем по пробелам
        tokens = re.findall(r'\S+', text)
        
        # Фильтруем токены, удаляя пунктуацию
        result = []
        for token in tokens:
            # Удаляем спецсимволы в начале и конце слова
            cleaned_token = re.sub(r'^[^\w]+|[^\w]+$', '', token)
            
            if cleaned_token and cleaned_token not in self.stop_words:
                result.append(cleaned_token)
        
        return result
    
    
    def stem_tokens(self, tokens: List[str]) -> List[str]:
        stemmed_tokens = []
        for token in tokens:
            # Стемминг только для слов (не для чисел, пунктуации и специальных символов)
            if re.match(r'^[a-zA-Z]+$', token):
                stemmed_tokens.append(self.stemmer.stem(token))
            else:
                stemmed_tokens.append(token)
        
        return stemmed_tokens
    
    def lemmatize_tokens(self, tokens: List[str]) -> List[str]:
        lemmatized_tokens = []
        for token in tokens:
            # Лемматизация только для слов
            if re.match(r'^[a-zA-Z]+$', token):
                lemmatized_tokens.append(self.lemmatizer.lemmatize(token))
            else:
                lemmatized_tokens.append(token)
        
        return lemmatized_tokens
    
    def process_text(self, text: str) -> dict:
        sentences = self.segment_sentences(text)
        tokenized_sentences = [self.tokenize_text(sent) for sent in sentences]
                
        tokens = []
        for sentence in tokenized_sentences:
            tokens.extend(sentence)

        stemmed_tokens = self.stem_tokens(tokens)
        lemmatized_tokens = self.lemmatize_tokens(tokens)
        
        return {
            'sentences': sentences,
            'tokens': tokens,
            'stemmed_tokens': stemmed_tokens,
            'lemmatized_tokens': lemmatized_tokens,
            'tokenized_sentences': tokenized_sentences
        }


def load_dataset(file_path: str) -> pd.DataFrame:
    """
    Загрузка датасета из CSV файла
    
    Args:
        file_path: Путь к CSV файлу
        
    Returns:
        DataFrame с данными
    """
    return pd.read_csv(file_path, header=None, names=['id', 'category', 'title', 'text'])




if __name__ == "__main__":
    processor = TextProcessor()
    
    test_cases = [
        ("Email", "Contact us at john.doe@example.com for more information"),
        ("Phone", "Call us at +7-901-000-00-00 or 8(918)3213412"),
        ("Abbreviation", "Dr. Smith works at St. Mary's Hospital Inc."),
        ("URL", "Visit https://www.example.com for details"),
        ("Complex", "Dr. Smith (john@example.com) called at +7-901-000-00-00, from 10.00 to 18.30, price $100"),
        ("Homonymy 1", "My car needs new tires. I'm completely tired after working."),
        ("Homonymy 2", "The carrier carried a huge bag."),
        ("Different forms", "I flew to Japan for 10 hours, the flights were delayed. I hate flying!"),

    ]
    
    print("=== Тестирование токенизации ===\n")
    
    for description, text in test_cases:
        result = processor.process_text(text)
        print(f"{description}:")
        print(f"  Текст: {text}")
        print(f"  Токены: {result['tokens']}")
        print(f"  Стемминг: {result['stemmed_tokens']}")
        print(f"  Лемматизация: {result['lemmatized_tokens']}")
    
    print("\n=== Тестирование полной обработки ===\n")
    
    sample_text = "Dr. Smith works at St. Mary's Hospital. Contact him at john@example.com or call +7-901-000-00-00. Pay $100 and come from 10.00 to 18:30."
    
    result = processor.process_text(sample_text)
    
    print(f"Исходный текст: {sample_text}")
    print(f"Предложения: {result['sentences']}")
    print(f"Токены: {result['tokens']}")
    print(f"Стемминг: {result['stemmed_tokens']}")
    print(f"Лемматизация: {result['lemmatized_tokens']}")
