import sys
from collections import defaultdict, Counter
from pathlib import Path

def analyze_tsv_files(corpus_dir):
    corpus_path = Path(corpus_dir)
    
    if not corpus_path.exists():
        print(f"Директория {corpus_dir} не найдена!")
        return {}, []
    
    lemmatization_stats = defaultdict(list)
    tsv_files = list(corpus_path.rglob("*.tsv"))
    
    if not tsv_files:
        print(f"TSV файлы не найдены в {corpus_dir}")
        return {}, []
    
    
    for tsv_file in tsv_files:
        try:
            with open(tsv_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    parts = line.split('\t')
                    if len(parts) >= 3:
                        token, stem, lemma = parts[0], parts[1], parts[2]
                        lemmatization_stats[token.lower()].append({
                            'stem': stem,
                            'lemma': lemma,
                            'file': tsv_file.name
                        })
        except Exception as e:
            print(f"Ошибка при обработке файла {tsv_file}: {e}")
    
    homonymy_cases = analyze_homonymy(lemmatization_stats)
    print_statistics(lemmatization_stats, homonymy_cases)
    
    return lemmatization_stats, homonymy_cases

def analyze_homonymy(lemmatization_stats):
    print("\n" + "="*60)
    print("АНАЛИЗ СЛУЧАЕВ ОМОНИМИИ")
    print("="*60)
    
    homonymy_cases = []
    
    for word, entries in lemmatization_stats.items():
        unique_lemmas = set(entry['lemma'] for entry in entries)
        
        if len(unique_lemmas) > 1:
            homonymy_cases.append({
                'word': word,
                'lemmas': list(unique_lemmas),
                'count': len(entries),
                'examples': entries[:3]
            })
    
    homonymy_cases.sort(key=lambda x: x['count'], reverse=True)
    
    print(f"Найдено {len(homonymy_cases)} случаев омонимии")
    print("\nТоп-20 случаев омонимии:")
    print("-" * 60)
    
    for i, case in enumerate(homonymy_cases[:20], 1):
        print(f"{i:2d}. Слово: '{case['word']}'")
        print(f"    Леммы: {', '.join(case['lemmas'])}")
        print(f"    Количество: {case['count']}")
        print()
    
    return homonymy_cases

def print_statistics(lemmatization_stats, homonymy_cases):
    print("\n" + "="*60)
    print("ОБЩАЯ СТАТИСТИКА")
    print("="*60)
    
    total_words = sum(len(entries) for entries in lemmatization_stats.values())
    unique_words = len(lemmatization_stats)
    
    print(f"Общее количество токенов: {total_words}")
    print(f"Уникальных слов: {unique_words}")
    print(f"Случаев омонимии: {len(homonymy_cases)}")
    
    all_lemmas = []
    for entries in lemmatization_stats.values():
        all_lemmas.extend([entry['lemma'] for entry in entries])
    
    lemma_counts = Counter(all_lemmas)
    print(f"Уникальных лемм: {len(lemma_counts)}")
    
    print("\nТоп-10 самых частых лемм:")
    for lemma, count in lemma_counts.most_common(10):
        print(f"  {lemma}: {count}")

if __name__ == "__main__":
    corpus_dir = sys.argv[1] if len(sys.argv) > 1 else 'assets/annotated-corpus/test'
    print("Анализ результатов лемматизации")
    print("="*60)
    
    lemmatization_stats, homonymy_cases = analyze_tsv_files(corpus_dir)
    print("\nАнализ завершен!")