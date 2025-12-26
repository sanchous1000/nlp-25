import re


def advanced_tokenize(text):
    # Шаблоны для сложных случаев
    patterns = [
        (r'\b[a-zA-Z]+@[a-zA-Z]+\.[a-zA-Z]+\b', 'EMAIL'),
        (r'(?:\+7-\d{3}-\d{3}-\d{2}-\d{2}|\b8\(\d{3}\)\d{7}\b)', 'PHONE'),
        (r'\b(?:Dr|Mr|Mrs|Ms|Prof)\.\s+[A-Z][a-zA-Z]*', 'APPEAL'),
        (r'^[a-zA-Z_]\w*\s*=\s*[\w\s()+\-*/^]+(?:\s*=\s*[\w\s()+\-*/^]+)*$', 'MATH')
    ]

    placeholders = {} # abs@abs.ru
    counter = 0

    for pattern, tag in patterns:
        matches = re.findall(pattern, text, flags=re.IGNORECASE)
        for match in set(matches):
            placeholder = f"__{tag}_{counter}__"
            text = text.replace(match, placeholder)
            placeholders[placeholder] = match
            counter += 1

    # Разбиение на предложения (учитывая, что точки в сокращениях не конец)
    sentences = re.split(r'[.!?]+', text)

    for sen in sentences:
        if sen == '':
            sentences.remove(sen)

    tokens_all = []
    for sent in sentences:
        # Токенизация простого текста
        tokens = re.findall(r"\w+(?:'\w+)?|[^\w\s]", sent)
        # Восстановление сложных токенов
        restored = []
        for token in tokens:
            if token in placeholders:
                restored.append(placeholders[token])
            else:
                restored.append(token)
        tokens_all.append(restored)

    return tokens_all