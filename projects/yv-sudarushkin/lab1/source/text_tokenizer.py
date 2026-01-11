import re


class TextTokenizer:
    def __init__(self, debug=False):
        # Паттерны для токенов, которые должны быть защищены (считаются одним токеном)
        self.protected_patterns = {
            # Эмотиконы и смайлы
            'emoticons': r'(?:[:;=]-?[)\(DPO])|(?:<3)|(?:\u2639|\u263a|\u2764)',

            # URL и веб-адреса
            'url': r'https?://[^\s<>"{}|\\^`\[\]]+[^\s<>"{}|\\^`\[\].]',
            'www': r'www\.[^\s<>"{}|\\^`\[\]]+\.[a-z]{2,}[^\s<>"{}|\\^`\[\]]*',

            # Электронные почты
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',

            # Телефоны (разные форматы)
            'phone': r'\b(?:\+\d{1,3}[-.\s]?)?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b',

            # Хештеги и упоминания
            'hashtag': r'#\w+',
            'mention': r'@\w+',

            # Числа и денежные суммы
            'currency': r'\$(?:\d+[\s\.,]?\d+)+(?:\.\d{2})?|(?:\d+[\s\.,]?\d+)+(?:\.\d{2})?\$',
            'percent': r'\d+(?:\.\d+)?%',
            'number': r'\b\d+(?:\.,\d+)?\b',
            'scientific': r'\b\d+(?:\.\d+)?[eE][+-]?\d+\b',

            # Сокращения (английские)
            'abbreviations': r'\b(?:[A-Za-z]\.){2,}s?|\b(?:Dr|Mr|Mrs|Ms|Prof|Rev|Hon|Capt|Lt|Col|Gen|Sgt|Cpl|Pvt|Esq|etc|approx|appt|apt|dept|est|min|max|temp|vol|fig|p|pp|ch|sec|ft|lb|oz|kg|mm|cm|m|km|mph)\.',

            # Время и даты
            'time': r'\b\d{1,2}:\d{2}(?::\d{2})?\s*(?:[ap]\.?m\.?)?\b',
            'date': r'\b\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4}\b',

            # Версии и коды
            'version': r'\b[vV]\d+(?:\.\d+)*\b',
            'isbn': r'\b\d{1,5}[- ]?\d{1,7}[- ]?\d{1,6}[- ]?[xX\d]\b',

            # Слова с апострофами (don't, it's)
            'contractions': r"\b\w+(?:'[a-z]{1,3})?\b",

            # Дефисные слова (state-of-the-art)
            'hyphenated_words': r'\b\w+(?:-\w+)+\b',
            # Токен подобные слова
            'encode_token': r'__TOKEN_\d+__',
        }

        # Компилируем защитный паттерн для поиска специальных токенов
        self.protection_pattern = re.compile(
            '|'.join(f'(?P<{name}>{pattern})' for name, pattern in self.protected_patterns.items()),
            re.IGNORECASE | re.UNICODE
        )

        # Основной паттерн для токенизации
        self.token_pattern = r'(?:{protected_patterns})|\w+(?:-\w+)*|\.\.\.|[^\w\s]'

        self._debug = debug

    def _debug_protection(self, text):
        """Отладочная функция: показывает, какие паттерны срабатывают"""
        if not self._debug:
            return

        print("=== DEBUG TOKEN PROTECTION ===")
        for match in self.protection_pattern.finditer(text):
            for name, value in match.groupdict().items():
                if value is not None and value.strip():
                    start, end = match.start(), match.end()
                    context_start = max(0, start - 5)
                    context_end = min(len(text), end + 5)
                    context = text[context_start:context_end]
                    print(f"Pattern '{name}' matched: '{value}'")
                    print(f"  Position: {start}-{end}")
                    print(f"  Context: ...{context}...")
                    print()
        print()

    def _protect_special_tokens(self, text):
        """Заменяет специальные токены на временные метки с сохранением информации для восстановления"""
        protected_text = text
        replacements = {}
        token_counter = 0

        # Собираем все совпадения с их позициями
        matches = []
        for match in self.protection_pattern.finditer(text):
            matches.append(match)

        # Сортируем совпадения по убыванию позиции начала (чтобы заменять с конца и не сбивать индексы)
        matches.sort(key=lambda x: x.start(), reverse=True)

        # Выполняем замены с конца текста к началу
        for match in matches:
            original = match.group()
            start, end = match.start(), match.end()

            # Пропускаем, если этот участок уже был заменен
            if protected_text[start:end] != original:
                continue

            # Создаем уникальный маркер
            token_id = f"__TOKEN_{token_counter}__"
            replacements[token_id] = original

            # Заменяем в тексте
            protected_text = protected_text[:start] + token_id + protected_text[end:]
            token_counter += 1

        if self._debug:
            print(f"Protected text: {protected_text}")
            print(f"Replacements: {replacements}")

        return protected_text, replacements

    def _restore_protected_tokens(self, tokens, replacements):
        """Восстанавливает защищенные токены в списке токенов"""
        restored_tokens = []

        for token in tokens:
            if token in replacements:
                # Если токен является маркером, восстанавливаем оригинал
                restored_tokens.append(replacements[token])
            elif any(marker in token for marker in replacements.keys()):
                # Если токен содержит маркер (может случиться при неправильной сегментации)
                restored_token = token
                for marker, original in replacements.items():
                    if marker in restored_token:
                        restored_token = restored_token.replace(marker, original)
                restored_tokens.append(restored_token)
            else:
                # Обычный токен
                restored_tokens.append(token)

        return restored_tokens

    def tokenize(self, text):
        """Основной метод токенизации текста"""
        if not text.strip():
            return []

        if self._debug:
            print(f"Original text: {text}")
            self._debug_protection(text)

        # Защищаем специальные токены
        protected_text, replacements = self._protect_special_tokens(text)

        if self._debug:
            print(f"After protection: {protected_text}")

        # Создаем динамический паттерн для токенизации
        protected_patterns_str = '|'.join(f'(?:{pattern})' for pattern in self.protected_patterns.values())
        dynamic_token_pattern = self.token_pattern.replace('{protected_patterns}', protected_patterns_str)

        # Используем findall для поиска всех токенов (включая знаки препинания)
        token_pattern = re.compile(dynamic_token_pattern, re.IGNORECASE | re.UNICODE)
        raw_tokens = token_pattern.findall(protected_text)

        if self._debug:
            print(f"After token pattern matching: {raw_tokens}")

        # Восстанавливаем защищенные токены
        restored_tokens = self._restore_protected_tokens(raw_tokens, replacements)

        # Фильтруем пустые токены
        final_tokens = [token for token in restored_tokens if token.strip()]

        if self._debug:
            print("=== TOKENIZATION RESULT ===")
            for i, token in enumerate(final_tokens, 1):
                print(f"Token {i}: '{token}'")
            print()

        return final_tokens

    def tokenize_sentences(self, sentences):
        """Токенизация списка предложений (совместимость с вашим сегментатором)"""
        tokenized_sentences = []
        for sentence in sentences:
            tokens = self.tokenize(sentence)
            tokenized_sentences.append(tokens)
        return tokenized_sentences


# Функция для комплексного тестирования
def test_complete_tokenizer():
    tokenizer = TextTokenizer(debug=False)

    # Тестовые случаи, охватывающие все аспекты
    test_cases = [
        # Базовые случаи со знаками препинания
        "Hello world!",
        "What is this? I don't know...",
        "Price: $19.99! Really??",

        # Специальные токены
        "Email: test@example.com, URL: https://example.com, Phone: +1 (555) 123-4567",
        "Dr. Smith works at ABC Inc. with Mr. Jones.",

        # Эмотиконы и социальные элементы
        "I'm happy :) but sometimes sad :(. Check out #NLP and @john_doe!",

        # Числа и форматы
        "The constant is 6.02e23. Temperature: 25.5°C. Discount: 15%.",
        "Version v2.1.3 released on 2023-12-01 at 2:30 p.m.",

        # Сложные смешанные случаи
        "Dr. Smith (email: dr.smith@hospital.com) called re: patient #123-45. Status: stable! Cost: $150.00.",
        "Visit https://example.com/path?query=1 for details!!! Or email info@site.co.uk.",

        # Крайние случаи
        "Multiple... punctuation!!! And symbols: @user, #tag1, #tag2.",
        "Test1@domain.com and test2@domain.org should both be preserved!"
    ]

    print("=== COMPREHENSIVE TOKENIZER TESTING ===")
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n--- TEST CASE {i} ---")
        print(f"Input: {test_case}")

        tokens = tokenizer.tokenize(test_case)
        print(f"Tokens ({len(tokens)}): {tokens}")

        # Анализ результатов
        words = [t for t in tokens if t.isalpha()]
        numbers = [t for t in tokens if any(c.isdigit() for c in t) and not any(c.isalpha() for c in t)]
        punctuation = [t for t in tokens if re.match(r'^[^\w\s]+$', t)]
        special = [t for t in tokens if any(char in t for char in ['@', '#', '$', '%', ':', '/'])]

        print(
            f"Analysis: {len(words)} words, {len(numbers)} numbers, {len(punctuation)} punctuation, {len(special)} special tokens")

        # Проверяем целостность специальных токенов
        email_preserved = any('@' in token for token in tokens)
        url_preserved = any('http' in token for token in tokens)
        currency_preserved = any('$' in token for token in tokens)

        print(
            f"Special tokens preserved - Email: {email_preserved}, URL: {url_preserved}, Currency: {currency_preserved}")


# Интеграционный тест с сегментатором
def test_integration_with_segmenter():
    print("\n=== INTEGRATION TEST WITH SENTENCE SEGMENTER ===")
    from sentence_segmenter import SentenceSegmenter
    # Используем ваш сегментатор
    segmenter = SentenceSegmenter(debug=False)
    tokenizer = TextTokenizer(debug=False)

    complex_text = """
    Dr. Smith works at ABC Inc. Her email is john.smith@company.com. 
    She said: "The meeting is at 2:30 p.m. in room 3.14! Don't be late." 
    The project costs $15,000.00. Visit https://example.com for details!!!
    """

    print(f"Original text: {complex_text}")

    # Сегментация на предложения
    sentences = segmenter.segment_text(complex_text)
    print(f"Segmented into {len(sentences)} sentences:")
    for i, sent in enumerate(sentences, 1):
        print(f"  {i}. {sent}")

    # Токенизация каждого предложения
    tokenized_sentences = tokenizer.tokenize_sentences(sentences)
    print(f"\nTokenized sentences:")
    for i, tokens in enumerate(tokenized_sentences, 1):
        print(f"  {i}. {tokens}")


if __name__ == "__main__":
    test_complete_tokenizer()
    test_integration_with_segmenter()