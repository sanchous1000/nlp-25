import re


class SentenceSegmenter:
    def __init__(self, debug=False):
        # Паттерны для случаев, которые НЕ должны разрывать предложения
        self.non_breaking_patterns = {
            'abbreviations': r'\b(?:[A-Z]\.\s*){2,}[A-Z]?',  # Инициалы: A. B. Smith, Ph.D.
            'common_abbr': r'\b(?:Dr|Mr|Mrs|Ms|Prof|Rev|Hon|Capt|Lt|Col|Gen|Sgt|Cpl|Pvt|Esq)\.',
            'time': r'\b\d{1,2}:\d{2}(?::\d{2})?\s*[ap]\.?m\.?',  # 10:30 a.m., 2:00 p.m.
            'dates': r'\b\d{1,2}\.\d{1,2}\.\d{2,4}',  # 12.03.2023
            'version': r'\b[vV]\d+\.\d+(?:\.\d+)*',  # v1.2.3
            'decimal': r'\b\d+\.\d+',  # 3.14, 2.5
            'email': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            'url': r'https?://[^\s]+[^\s.]'
        }

        # Основной паттерн для разделения предложений
        self.sentence_endings = r'[.!?]+'

        # Комбинированный паттерн для защиты не-разрывающих случаев
        self.protection_pattern = re.compile(
            '|'.join(f'(?P<{name}>{pattern})' for name, pattern in self.non_breaking_patterns.items()),
            re.IGNORECASE | re.UNICODE
        )
        self._debug = debug

    def _debug_protection(self, text):
        """Отладочная функция: показывает, какие паттерны срабатывают и где"""
        print("=== DEBUG PROTECTION PATTERNS ===")
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
                    print(f"  Dots in match: {value.count('.')}")
                    print()

        # Также покажем все точки в тексте и их контекст
        print("=== ALL DOTS IN TEXT ===")
        for i, char in enumerate(text):
            if char == '.':
                context_start = max(0, i - 10)
                context_end = min(len(text), i + 10)
                context = text[context_start:context_end]
                print(f"Dot at position {i}: ...{context}...")
        print()

    def _protect_non_breaking(self, text, debug=False):
        """Заменяет точки в не-разрывающих случаях на временные метки с отладкой"""
        if self._debug:
            self._debug_protection(text)

        protected_text = text
        replacements = {}
        pattern_matches = {}

        # Находим все случаи, которые нужно защитить
        for match in self.protection_pattern.finditer(text):
            for name, value in match.groupdict().items():
                if value is not None and value.strip():
                    original = value
                    # Записываем, какой паттерн сработал
                    pattern_matches[(match.start(), match.end())] = name
                    # Заменяем точки на временный маркер
                    protected = original.replace('.', '‹DOT›')
                    replacements[original] = protected
                    protected_text = protected_text.replace(original, protected)

        if self._debug:
            print(f"""=== AFTER PROTECTION ===
Protected text: {protected_text}
Replacements: {replacements}
""")

        return protected_text, replacements, pattern_matches

    def _should_split_sentence(self, protected_text, current_pos, debug=False):
        """Определяет, стоит ли разрывать предложение в текущей позиции"""
        if current_pos >= len(protected_text):
            return False

        char = protected_text[current_pos]

        # Проверяем, является ли текущий символ концом предложения
        if not re.match(self.sentence_endings, char):
            return False

        # Проверяем следующий символ (если есть)
        if current_pos + 1 < len(protected_text):
            next_char = protected_text[current_pos + 1]

            # Если следующий символ - пробел, а за ним - заглавная буква или конец текста
            if next_char.isspace():
                # Ищем следующий не-пробельный символ
                next_non_space = current_pos + 2
                while next_non_space < len(protected_text) and protected_text[next_non_space].isspace():
                    next_non_space += 1

                if next_non_space >= len(protected_text):
                    return True  # Конец текста
                elif protected_text[next_non_space].isupper():
                    return True  # Заглавная буква после пробела
            elif next_char in ['"', "'", "”", "»", "<"]:
                # Кавычка после точки - тоже конец предложения
                return True
        else:
            # Конец текста
            return True

        return False

    def segment_text(self, text, debug=False):
        """Основной метод токенизации предложений с отладкой"""
        if not text.strip():
            return []

        if self._debug:
            print(f"Original text: {text}")
            print()

        # Защищаем случаи, которые не должны разрывать предложения
        protected_text, replacements, pattern_matches = self._protect_non_breaking(text, debug)

        if self._debug:
            print("=== SENTENCE SPLITTING ===")

        # Разделяем на предложения
        sentences = []
        current_sentence = ""
        i = 0

        while i < len(protected_text):
            char = protected_text[i]
            current_sentence += char

            # Проверяем, стоит ли разрывать предложение в этой позиции
            should_split = self._should_split_sentence(protected_text, i, debug)

            if self._debug and char in '.!?' and not protected_text[i:].startswith('‹DOT›'):
                context_start = max(0, i - 5)
                context_end = min(len(protected_text), i + 5)
                context = protected_text[context_start:context_end]
                print(f"Position {i}: '{char}' in context '{context}'")
                print(f"  Should split: {should_split}")
                # Проверяем, была ли эта точка защищена
                for (start, end), pattern_name in pattern_matches.items():
                    if start <= i < end:
                        print(f"  Protected by pattern: {pattern_name}")
                        break
                print()

            if should_split:
                sentences.append(current_sentence)
                current_sentence = ""
                # Пропускаем пробел после точки, если есть
                if i + 1 < len(protected_text) and protected_text[i + 1].isspace():
                    i += 1

            i += 1

        # Добавляем последнее предложение, если оно есть
        if current_sentence.strip():
            sentences.append(current_sentence)

        # Восстанавливаем защищенные случаи
        restored_sentences = []
        for sentence in sentences:
            restored = sentence
            for original, protected in replacements.items():
                # Восстанавливаем оригинальные точки
                restored = restored.replace(protected, original)
            restored_sentences.append(restored.strip())

        result = [s for s in restored_sentences if s]

        if self._debug:
            print("=== FINAL RESULT ===")
            for i, sent in enumerate(result, 1):
                print(f"Sentence {i}: {sent}")
            print()

        return result


# Функция для тестирования проблемных случаев
def test_problem_cases():
    tokenizer = SentenceSegmenter(debug=True)

    problem_cases = [
        "Hello Dr. Smith! How are you? I'm fine.",
        "The meeting is at 2:30 p.m. in room 3.14. Don't be late!",
        "She works at ABC Inc. Her email is test@example.com.",
        "Visit https://example.com. It's a great website!",
        "I like apples, oranges, etc. But I don't like bananas.",
        "The price is $19.99. It's a good deal!",
        "Check version v2.1.3. It fixes many bugs.",
    ]

    for i, test_case in enumerate(problem_cases, 1):
        print(f"=== TEST CASE {i} ===")
        print(f"Input: {test_case}")
        sentences = tokenizer.segment_text(test_case)
        print(f"Output: {sentences}")
        print("\n" + "="*50 + "\n")


if __name__ == "__main__":
    test_problem_cases()
