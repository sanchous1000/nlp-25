from text_utils import segment_sentences, tokenize_text, process_text
from stemming import stem_word
from lemmatization import lemmatize_word

def demo_tokenization():
    test_cases = [
        # Русские примеры
        ("Телефонные номера (RU)", "Звоните по номеру +7-901-000-00-00 или 8(918)3213412"),
        ("Email адреса (RU)", "Свяжитесь с нами: support@company.com или info@example.org"),
        ("Адреса (RU)", "Наш офис: г. Санкт-Петербург, ул. Советская, д. 1294124, кв. 1"),
        ("Эмотиконы (RU)", "Привет :) как дела? :( грустно... :D радостно! ;)"),
        ("Математические выражения (RU)", "Формула: a = b*c = (c+d)^2, также x + y = z"),
        ("Сокращения (RU)", "Dr. Иванов работает в компании Inc. вместе с Mr. Петровым"),
        ("URL (RU)", "Посетите наш сайт: https://example.com или http://test.org"),
        
        # Английские примеры
        ("Phone numbers (EN)", "Call us at +1-555-123-4567 or (555) 987-6543"),
        ("Email addresses (EN)", "Contact us: support@company.com or info@example.org"),
        ("Addresses (EN)", "Our office: 123 Main St, New York, NY 10001, USA"),
        ("Emoticons (EN)", "Hello :) how are you? :( sad... :D happy! ;)"),
        ("Math expressions (EN)", "Formula: x = y*z = (a+b)^2, also p + q = r"),
        ("Abbreviations (EN)", "Dr. Smith works at ABC Corp. with Mr. Johnson"),
        ("URLs (EN)", "Visit our website: https://example.com or http://test.org"),
        ("Mixed content (EN)", "TechCorp Inc. (tel: +1-800-555-0199, email: info@techcorp.com) :)")
    ]
    
    print("ДЕМОНСТРАЦИЯ ТОКЕНИЗАЦИИ")
    print("=" * 80)
    
    for i, (name, text) in enumerate(test_cases, 1):
        print(f"\n{i}. {name}")
        print(f"Текст: {text}")
        print("-" * 60)
        
        tokens = tokenize_text(text)
        print(f"Токенов: {len(tokens)}")
        print("=" * 80)

def demo_stemming_lemmatization():
    test_words = [
        # Русские слова
        "работа", "работать", "работающий", "работал", "работала",
        "дом", "дома", "дому", "домом", "домашний",
        "красивый", "красивая", "красивое", "красивые", "красота",
        "читать", "читаю", "читает", "читал", "читала", "читающий",
        
        # Английские слова
        "work", "working", "worked", "works", "worker",
        "house", "houses", "housing", "housed", "home",
        "beautiful", "beauty", "beautify", "beautifying",
        "read", "reading", "reads", "reader", "readable"
    ]
    
    print("\nДЕМОНСТРАЦИЯ СТЕММИНГА И ЛЕММАТИЗАЦИИ")
    print("=" * 80)
    
    print(f"{'Слово':<15} {'Стем':<15} {'Лемма':<15}")
    print("-" * 45)
    
    for word in test_words:
        stem = stem_word(word)
        lemma = lemmatize_word(word)
        print(f"{word:<15} {stem:<15} {lemma:<15}")

def demo_full_processing():
    sample_texts = [
        """
        TechCorp Inc. is hiring!
        
        Contact us:
        - Phone: +1-800-555-0199, (555) 123-4567
        - Email: hr@techcorp.com, info@techcorp.com
        - Address: 123 Main St, New York, NY 10001, USA
        
        We're looking for experienced developers :) 
        Salary: from $100,000 
        Formula: salary = base * experience^2
        
        Website: https://techcorp.com
        """
    ]
    
    print("\nДЕМОНСТРАЦИЯ ПОЛНОЙ ОБРАБОТКИ ТЕКСТА")
    print("=" * 80)
    
    for i, sample_text in enumerate(sample_texts, 1):
        language = "РУССКИЙ" if i == 1 else "АНГЛИЙСКИЙ"
        print(f"\n{i}. {language} ТЕКСТ:")
        print("-" * 60)
        print("Исходный текст:")
        print(sample_text)
        print("\n" + "=" * 80)
        
        annotations = process_text(sample_text)
        
        print(f"Обработано токенов: {len(annotations)}")
        print("=" * 80)

if __name__ == "__main__":
    demo_tokenization()
    demo_stemming_lemmatization()
    demo_full_processing()
    
    print("\n" + "=" * 80)
    print("Демонстрация завершена!")