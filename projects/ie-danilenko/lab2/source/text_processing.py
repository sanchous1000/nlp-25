"""
Общие функции и константы для обработки текста.
Используется в различных скриптах проекта для единообразия обработки текста.
"""

# Английские стоп-слова
STOP_WORDS = {
    'a', 'an', 'and', 'are', 'as', 'at', 'be', 'been', 'by', 'for', 'from',
    'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to',
    'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they', 'have',
    'had', 'what', 'said', 'each', 'which', 'their', 'time', 'will', 'about',
    'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some', 'her',
    'would', 'make', 'like', 'into', 'him', 'has', 'two', 'more', 'very',
    'after', 'words', 'long', 'than', 'first', 'been', 'call', 'who', 'oil',
    'its', 'now', 'find', 'down', 'day', 'did', 'get', 'come', 'made', 'may',
    'part', 'over', 'new', 'sound', 'take', 'only', 'little', 'work', 'know',
    'place', 'year', 'live', 'me', 'back', 'give', 'most', 'very', 'after',
    'thing', 'our', 'just', 'name', 'good', 'sentence', 'man', 'think', 'say'
}

# Пунктуация для фильтрации
PUNCTUATION = set('.,!?;:"()[]{}\'\"-/—–')


def is_punctuation(token):
    """
    Проверка, является ли токен пунктуацией.
    
    Args:
        token: Токен для проверки
        
    Returns:
        True, если токен состоит только из пунктуации или пуст после удаления пробелов
    """
    return all(char in PUNCTUATION for char in token) or not token.strip()


def is_stop_word(token):
    """
    Проверка, является ли токен стоп-словом.
    
    Args:
        token: Токен для проверки (должен быть в нижнем регистре)
        
    Returns:
        True, если токен является стоп-словом
    """
    return token.lower() in STOP_WORDS


def should_filter_token(token):
    """
    Проверяет, нужно ли фильтровать токен (пунктуация или стоп-слово).
    
    Args:
        token: Токен для проверки
        
    Returns:
        True, если токен нужно отфильтровать
    """
    token_lower = token.lower().strip()
    return (not token_lower or 
            is_punctuation(token_lower) or 
            is_stop_word(token_lower))

