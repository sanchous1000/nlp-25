import sys
from text_utils import process_text
from annotation import create_annotated_corpus
from time import time

def test_processing():
    test_texts = [
        "Hello! How are you? My email: john@company.com",
        "Call us at +1-555-123-4567 or (555) 987-6543",
        "Address: 123 Main St, New York, NY 10001, USA",
        "Math: x = y*z = (a+b)^2",
        "Dr. Smith works at ABC Corp.",
        "Emoji: :) and :( emotions",
        "Visit our website: https://example.com",
        "Mr. Johnson from Ltd. company"
    ]
    
    print("Тестирование обработки текста:")
    print("=" * 50)
    
    for i, text in enumerate(test_texts, 1):
        print(f"\nТест {i}: {text}")
        annotations = process_text(text)
        
        for ann in annotations:
            print(f"  {ann['token']} -> {ann['stem']} -> {ann['lemma']}")
        
        print(f"Всего токенов: {len(annotations)}")

if __name__ == "__main__":
    
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else 'assets/annotated-corpus'
        
        test_processing()
        start = time()
        create_annotated_corpus(dataset_path, output_dir)
        print(time() - start)
    else:
        print("Использование: python main.py <путь_к_датасету> [путь_к_выходной_папке]")
        print("Пример: python main.py dataset/train.csv")