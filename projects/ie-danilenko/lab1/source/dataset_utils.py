import pandas as pd
from pathlib import Path

def read_csv_file(file_path):
    file_path = Path(file_path)
    
    if file_path.exists():
        try:
            data = pd.read_csv(file_path)
            return data
        except Exception as e:
            print(f"Ошибка при чтении файла {file_path}: {e}")
            return None
    else:
        print(f"Файл {file_path} не найден!")
        return None

def get_columns(dataframe):
    if dataframe is None:
        return None, None, None
    
    columns = list(dataframe.columns)
    label_col = columns[0]  # Первая колонка - label
    text_col = columns[2]    # Третья колонка - text
    return columns, label_col, text_col

def process_dataframe_row(row, label_col, text_col):
    return {
        'label': str(row[label_col]),
        'text': str(row[text_col])
    }