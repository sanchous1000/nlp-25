import sys
from pathlib import Path
from text_utils import process_text
from dataset_utils import read_csv_file, get_columns, process_dataframe_row

def save_annotation_to_tsv(annotations, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for ann in annotations:
            f.write(f"{ann['token']}\t{ann['stem']}\t{ann['lemma']}\n")

def create_annotated_corpus(dataset_path, output_dir='assets/annotated-corpus'):
    dataset_name = dataset_path.split('/')[-1].replace('.csv', '')
    dataset_data = read_csv_file(dataset_path)
    
    if dataset_data is None:
        return
    
    output_path = Path(output_dir + f'/{dataset_name}')
    output_path.mkdir(parents=True, exist_ok=True)
    
    columns, label_col, text_col = get_columns(dataset_data)
    if columns is None:
        return
    
    for idx, row in dataset_data.iterrows():
        data = process_dataframe_row(row, label_col, text_col)
        annotations = process_text(data['text'])
        
        class_dir = output_path / data['label']
        class_dir.mkdir(exist_ok=True)
        
        output_file = class_dir / f"{idx}.tsv"
        save_annotation_to_tsv(annotations, output_file)
    
    print(f"Аннотированный корпус создан: {len(dataset_data)} документов")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        dataset_path = sys.argv[1]
        output_dir = sys.argv[2] if len(sys.argv) > 2 else 'assets/annotated-corpus'
        create_annotated_corpus(dataset_path, output_dir)
    else:
        print("Использование: python annotation.py <путь_к_датасету> [путь_к_выходной_папке]")