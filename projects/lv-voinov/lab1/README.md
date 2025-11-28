## Отчет по 1 лабораторной

Использовались SnowballStemmer, WordNetLemmatizer из библиотеки nltk.

Полный код находится в файле lab1.ipynb.

Код токенизации:

```
def split_sentences_regex(text):
    pattern = r'\s*[!?.]+\s+'
    sentences = re.split(pattern, text)
    return [s.strip() for s in sentences if s.strip()]

def tokenize_with_entities(text):
    tokens = []
    
    patterns = {
        'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        
        'phone': r'\b(?:\+?\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b',
        
        'address': (
            r'\b\d+\s+[A-Z][a-z]+\s+(?:Street|St|Avenue|Ave|Road|Rd|Boulevard|Blvd|Lane|Ln|Drive|Dr)\b|'
            r'(?:Apt|Apartment|Suite|Ste)\.?\s*\d+[A-Z]?\b'
        )
    }
    
    combined_pattern = re.compile(
        f'(?P<email>{patterns["email"]})|(?P<phone>{patterns["phone"]})|(?P<address>{patterns["address"]})',
        re.IGNORECASE
    )
    
    last_end = 0
    
    for match in combined_pattern.finditer(text):
        if match.start() > last_end:
            interim_text = text[last_end:match.start()]
            words = re.findall(r'\b\w+\b', interim_text.lower())
            for word in words:
                tokens.append({'token': word, 'type': 'word'})
        
        entity_type = 'word'
        for group_name in ['email', 'phone', 'address']:
            if match.group(group_name):
                entity_type = group_name
                break
        
        tokens.append({
            'token': match.group().lower() if entity_type in ['word', 'email', 'phone'] else match.group(),
            'type': entity_type
        })
        last_end = match.end()
    
    if last_end < len(text):
        interim_text = text[last_end:]
        words = re.findall(r'\b\w+\b', interim_text.lower())
        for word in words:
            tokens.append({'token': word, 'type': 'word'})
    
    return tokens
```

Пример токенизации особых случаев:

```
text = '''Alternative phone: 8(918)3213412. Write to abc@abc.com.
NY office: 123 Fifth Avenue, Apt 45. Call +1 (555) 123-4567!
'''

[{'token': 'alternative', 'type': 'word'}, {'token': 'phone', 'type': 'word'}, {'token': '8(918)3213412', 'type': 'phone'}]
[{'token': 'write', 'type': 'word'}, {'token': 'to', 'type': 'word'}, {'token': 'abc@abc.com', 'type': 'email'}]
[{'token': 'ny', 'type': 'word'}, {'token': 'office', 'type': 'word'}, {'token': '123 Fifth Avenue', 'type': 'address'}, {'token': 'Apt 45', 'type': 'address'}]
[{'token': 'call', 'type': 'word'}, {'token': '1 (555) 123-4567', 'type': 'phone'}]
```

Код стемминга и лемматизации:

```
def stemming(token_objects):
    return [
        stemmer.stem(item['token']) if item['type'] == 'word' else item['token']
        for item in token_objects
    ]

def lemmatization(token_objects):
    return [
        lemmatizer.lemmatize(item['token']) if item['type'] == 'word' else item['token']
        for item in token_objects
    ]
```

Тестирование лемматизации для омонимов. Случаи когда лемматизация как глагол не совпадает с лемматизацией как существительное:

```
The leaves leave the tree.
leaves -> leaf
Как глагол: leave, как существительное: leaf

He wound the bandage around the wound.
wound -> wound
Как глагол: wind, как существительное: wound

He wound the bandage around the wound.
wound -> wound
Как глагол: wind, как существительное: wound

She is running fast. The running water is cold.
running -> running
Как глагол: run, как существительное: running

She is running fast. The running water is cold.
running -> running
Как глагол: run, как существительное: running

He draws a drawing every day.
drawing -> drawing
Как глагол: draw, как существительное: drawing

The mice mouse around quietly.
mice -> mouse
Как глагол: mice, как существительное: mouse
```

Было сгенерировано 120000 tsv-файлов аннотации в необходимом формате для train датасета.
