import os
from pathlib import Path

output_dir_train = "assets/annotated_corpus/train"
output_dir_test = "assets/annotated_corpus/test"

os.makedirs(output_dir_train, exist_ok=True)
os.makedirs(output_dir_test, exist_ok=True)


def sanitize_filename(filename: str) -> str:
    """Sanitize a string to be used as a valid filename."""
    import re
    # Remove invalid characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    # Replace spaces with underscores
    filename = re.sub(r'\s+', '_', filename)
    # Limit length
    return filename[:255]

def save(doc_id,pack,label,split='train'):
    filename = f"{sanitize_filename(str(doc_id))}.tsv"
    folder_path = sanitize_filename(str(label))

    output_dir = output_dir_train if split == 'train' else output_dir_test
    
    os.makedirs(os.path.join(output_dir, folder_path), exist_ok=True)

    with open(os.path.join(output_dir, folder_path, filename), 'w', encoding='utf-8') as f:
        for tokens, stemmes, lemmas in pack:
            line = f"\t{tokens}\t{stemmes}\t{lemmas}"
            f.write(line + '\n')
        f.write('\n')  # Separate sentences by a newline
