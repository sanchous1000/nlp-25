import re
import os
import pandas as pd
import nltk
from nltk.stem import SnowballStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
from tqdm import tqdm


nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger')

stemmer = SnowballStemmer("english")
lemmatizer = WordNetLemmatizer()


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


def segment_sentences(text):
    text = re.sub(r'\s+', ' ', text).strip()
    sentences = re.split(
        r'(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<![A-Z][a-z]{2}\.)(?<=[.!?])\s+(?=[\"\']*[A-Z])',
        text
    )
    return sentences


def tokenize(text):
    pattern = r"""
        [\w.]+@[\w.]+\.[\w.]+
        |
        (?:\+?\d{1,3}[-.\s]?)?                
        \(?\d{3}\)?[-.\s]?                    
        \d{3}[-.\s]?\d{4}                     
        |
        (?:\+?\d{1,3}[-.\s]?)?               
        \(?\d{3}\)?[-.\s]?                    
        \d{3}[-.\s]?\d{2}[-.\s]?\d{2}         
        |
        [:=;xX][oO\-]?[D\)\]\(\]/\\OpP3] 
        |
        \b\w+(?:['.]\w+)*\b
    """

    tokens = re.findall(pattern, text, re.VERBOSE)
    return [t for t in tokens if t.strip()]


def process_text(text):
    sentences = segment_sentences(text)
    annotated_sentences = []

    for sent in sentences:
        tokens = tokenize(sent)
        pos_tags = nltk.pos_tag(tokens)
        annotated_tokens = []

        for token, (word, pos_tag) in zip(tokens, pos_tags):
            stem = stemmer.stem(token) if token.isalpha() else token

            if token.isalpha():
                wordnet_pos = get_wordnet_pos(pos_tag)
                lemma = lemmatizer.lemmatize(token.lower(), pos=wordnet_pos)
            else:
                lemma = token

            annotated_tokens.append((token, stem, lemma))

        annotated_sentences.append(annotated_tokens)

    return annotated_sentences


def save_annotations(annotations, output_path):
    """Сохранение аннотаций в TSV формат"""
    with open(output_path, 'w', encoding='utf-8') as f:
        for i, sent in enumerate(annotations):
            for token, stem, lemma in sent:
                f.write(f"{token}\t{stem}\t{lemma}\n")
            if i < len(annotations) - 1:
                f.write("\n")  # Двойной перенос строки между предложениями (один уже есть после последнего токена)


def check_lemmatization(text_sample):
    """
    Проверка результатов лемматизации и поиск случаев омонимии.
    Возвращает словарь с примерами омонимии.
    """
    tokens = tokenize(text_sample)
    homonymy_examples = {}
    
    for token in tokens:
        # Пропускаем не-слова
        if not re.match(r'^[A-Za-zА-Яа-яЁё]+$', token):
            continue
            
        word_lower = token.lower()
        
        # Получаем все возможные леммы для разных частей речи
        lemmas_dict = {
            'n': lemmatizer.lemmatize(word_lower, pos=wordnet.NOUN),
            'v': lemmatizer.lemmatize(word_lower, pos=wordnet.VERB),
            'a': lemmatizer.lemmatize(word_lower, pos=wordnet.ADJ),
            'r': lemmatizer.lemmatize(word_lower, pos=wordnet.ADV)
        }
        
        # Находим уникальные леммы (омонимия)
        unique_lemmas = set(lemmas_dict.values())
        if len(unique_lemmas) > 1:
            if word_lower not in homonymy_examples:
                homonymy_examples[word_lower] = {
                    'word': token,
                    'lemmas': unique_lemmas,
                    'pos_mapping': lemmas_dict
                }
    
    return homonymy_examples


def analyze_lemmatization():
    """Анализ результатов лемматизации и поиск случаев омонимии"""
    print("Проверка результатов лемматизации...")
    
    # Берем несколько примеров из датасета
    train_df = pd.read_csv("assets/raw/train.csv")
    sample_texts = train_df["text"].head(10).tolist()
    
    all_homonymy = {}
    for text in sample_texts:
        homonymy = check_lemmatization(text)
        all_homonymy.update(homonymy)
    
    print(f"\nНайдено случаев омонимии: {len(all_homonymy)}")
    print("\nПримеры омонимии (словоформа -> возможные леммы):")
    print("-" * 60)
    
    # Показываем первые 10 примеров
    for i, (word, info) in enumerate(list(all_homonymy.items())[:10]):
        print(f"\n{i+1}. Слово: '{info['word']}'")
        print(f"   Возможные леммы: {', '.join(sorted(info['lemmas']))}")
        print(f"   Части речи: {info['pos_mapping']}")
    
    if len(all_homonymy) > 10:
        print(f"\n... и еще {len(all_homonymy) - 10} случаев")
    
    return all_homonymy


def write_split(df, split):
    """Обработка датасета и сохранение аннотаций в структуру директорий"""
    base = f"assets/annotated-corpus/{split}"
    
    print(f"Обработка {split} датасета ({len(df)} документов)...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Processing {split}"):
        label = str(row["label"])
        text = str(row["text"])

        # Используем индекс как doc_id, если нет колонки id
        doc_id = str(row["id"]) if "id" in row else str(idx)

        # Создаем директорию для класса
        dir_path = os.path.join(base, label)
        os.makedirs(dir_path, exist_ok=True)

        # Путь к выходному файлу
        out_path = os.path.join(dir_path, f"{doc_id}.tsv")

        # Обрабатываем документ
        annotations = process_text(text)
        
        # Сохраняем аннотации
        save_annotations(annotations, out_path)


if __name__ == "__main__":
    # Проверка лемматизации перед обработкой
    homonymy_examples = analyze_lemmatization()

    # Обработка датасетов
    train_df = pd.read_csv("assets/raw/train.csv")
    test_df = pd.read_csv("assets/raw/test.csv")

    print("\nОбработка датасетов...")
    write_split(train_df, "train")
    write_split(test_df, "test")
    print("Готово!")