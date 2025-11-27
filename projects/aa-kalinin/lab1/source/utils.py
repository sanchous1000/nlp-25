import csv
import os
from pathlib import Path
from typing import List, Dict, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count

import nltk
from tqdm import tqdm
from nltk.stem import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from nlp import TextProcessor


def download_nltk_data():
    packages = ['wordnet', 'averaged_perceptron_tagger', 'punkt', 'omw-1.4', 'averaged_perceptron_tagger_eng']
    for package in packages:
        try:
            nltk.download(package, quiet=True)
        except Exception as e:
            print(f"Warning: Could not download {package}: {e}")


# Global processor for worker processes (initialized once per worker)
_worker_processor = None


def _init_worker():
    """Initialize worker process with its own TextProcessor."""
    global _worker_processor
    _worker_processor = TextProcessor()


def _process_document(args: Tuple[str, str, str, str, str]) -> str:
    """
    Worker function to process a single document.

    Args:
        args: Tuple of (doc_id, label, text, split, output_dir)

    Returns:
        doc_id on success
    """
    global _worker_processor
    doc_id, label, text, split, output_dir = args

    # Initialize processor if not already done
    if _worker_processor is None:
        _init_worker()

    output_path = Path(output_dir) / split / label
    output_path.mkdir(parents=True, exist_ok=True)

    sentences = _worker_processor.process_text(text)

    tsv_path = output_path / f"{doc_id}.tsv"
    with open(tsv_path, 'w', encoding='utf-8') as f:
        for i, sentence in enumerate(sentences):
            for token, stem, lemma in sentence:
                f.write(f"{token}\t{stem}\t{lemma}\n")
            if i < len(sentences) - 1:
                f.write("\n")

    return doc_id


class AnnotationGenerator:
    """Generate TSV annotation files from processed documents."""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.processor = TextProcessor()

    def generate_tsv(self, doc_id: str, label: str, text: str, split: str = 'train') -> Path:
        """Generate a TSV annotation file for a document."""
        output_path = self.output_dir / split / label
        output_path.mkdir(parents=True, exist_ok=True)

        sentences = self.processor.process_text(text)

        tsv_path = output_path / f"{doc_id}.tsv"
        with open(tsv_path, 'w', encoding='utf-8') as f:
            for i, sentence in enumerate(sentences):
                for token, stem, lemma in sentence:
                    f.write(f"{token}\t{stem}\t{lemma}\n")
                if i < len(sentences) - 1:
                    f.write("\n")

        return tsv_path


def load_dataset(csv_path: str) -> List[Dict]:
    """Load dataset from CSV file."""
    documents = []
    with open(csv_path, 'r', encoding='utf-8', errors='replace') as f:
        reader = csv.reader(f)
        for idx, row in enumerate(reader):
            if len(row) >= 3:
                label = row[0].strip().strip('"')
                title = row[1].strip().strip('"')
                text = row[2].strip().strip('"')
                full_text = f"{title}. {text}" if title else text

                documents.append({
                    'doc_id': str(idx + 1).zfill(6),
                    'label': label,
                    'text': full_text
                })

    return documents


def process_dataset(dataset_dir: str, output_dir: str, verbose: bool = True, n_workers: int = None):
    """
    Process entire dataset and generate annotations using multiprocessing.

    Args:
        dataset_dir: Path to dataset directory containing train.csv and test.csv
        output_dir: Path to output directory for annotations
        verbose: Print progress information
        n_workers: Number of worker processes (default: cpu_count - 1)
    """
    dataset_path = Path(dataset_dir)

    # Determine number of workers
    if n_workers is None:
        n_workers = max(1, cpu_count() - 1)

    if verbose:
        print(f"Using {n_workers} worker processes")

    for split in ['train', 'test']:
        csv_path = dataset_path / f"{split}.csv"
        if not csv_path.exists():
            if verbose:
                print(f"Warning: {csv_path} not found, skipping...")
            continue

        documents = load_dataset(str(csv_path))

        # Prepare arguments for workers
        work_items = [
            (doc['doc_id'], doc['label'], doc['text'], split, output_dir)
            for doc in documents
        ]

        # Process with multiprocessing
        with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker) as executor:
            futures = {executor.submit(_process_document, item): item[0] for item in work_items}

            with tqdm(total=len(futures), desc=f"Processing {split}", disable=not verbose) as pbar:
                for future in as_completed(futures):
                    try:
                        future.result()
                    except Exception as e:
                        doc_id = futures[future]
                        print(f"Error processing {doc_id}: {e}")
                    pbar.update(1)


def process_dataset_single(dataset_dir: str, output_dir: str, verbose: bool = True):
    """Process dataset in single-threaded mode (for debugging or small datasets)."""
    dataset_path = Path(dataset_dir)
    generator = AnnotationGenerator(output_dir)

    for split in ['train', 'test']:
        csv_path = dataset_path / f"{split}.csv"
        if not csv_path.exists():
            if verbose:
                print(f"Warning: {csv_path} not found, skipping...")
            continue

        documents = load_dataset(str(csv_path))

        for doc in tqdm(documents, desc=f"Processing {split}", disable=not verbose):
            generator.generate_tsv(
                doc_id=doc['doc_id'],
                label=doc['label'],
                text=doc['text'],
                split=split
            )


def demonstrate_lemmatization_ambiguity() -> str:
    """Demonstrate cases of lemmatization ambiguity/homonymy."""
    processor = TextProcessor()
    lemmatizer = WordNetLemmatizer()

    report = """
=============================================================================
                    LEMMATIZATION EVALUATION REPORT
=============================================================================

1. UNDERSTANDING LEMMATIZATION AMBIGUITY/HOMONYMY

Lemmatization attempts to reduce words to their dictionary form (lemma).
However, some words can have multiple valid lemmas depending on their
part of speech (POS) or semantic context. This is called lemmatization
ambiguity or homonymy.

-----------------------------------------------------------------------------

2. EXAMPLES OF AMBIGUITY

Example 1: "saw"
- As a noun: "saw" → "saw" (a cutting tool)
- As a verb (past tense of "see"): "saw" → "see"

"""
    saw_noun = lemmatizer.lemmatize('saw', wordnet.NOUN)
    saw_verb = lemmatizer.lemmatize('saw', wordnet.VERB)
    report += f"   lemmatize('saw', NOUN) = '{saw_noun}'\n"
    report += f"   lemmatize('saw', VERB) = '{saw_verb}'\n\n"

    report += """
Example 2: "better"
- As an adjective: "better" → "better" (comparative form is kept)
- As a verb: "better" → "better" (to improve)

"""
    better_adj = lemmatizer.lemmatize('better', wordnet.ADJ)
    better_verb = lemmatizer.lemmatize('better', wordnet.VERB)
    report += f"   lemmatize('better', ADJ)  = '{better_adj}'\n"
    report += f"   lemmatize('better', VERB) = '{better_verb}'\n\n"

    report += """
Example 3: "leaves"
- As a noun (plural): "leaves" → "leaf"
- As a verb (3rd person singular): "leaves" → "leave"

"""
    leaves_noun = lemmatizer.lemmatize('leaves', wordnet.NOUN)
    leaves_verb = lemmatizer.lemmatize('leaves', wordnet.VERB)
    report += f"   lemmatize('leaves', NOUN) = '{leaves_noun}'\n"
    report += f"   lemmatize('leaves', VERB) = '{leaves_verb}'\n\n"

    report += """
Example 4: "meeting"
- As a noun: "meeting" → "meeting" (a gathering)
- As a verb: "meeting" → "meet" (present participle)

"""
    meeting_noun = lemmatizer.lemmatize('meeting', wordnet.NOUN)
    meeting_verb = lemmatizer.lemmatize('meeting', wordnet.VERB)
    report += f"   lemmatize('meeting', NOUN) = '{meeting_noun}'\n"
    report += f"   lemmatize('meeting', VERB) = '{meeting_verb}'\n\n"

    report += """
Example 5: "banks"
- As a noun (plural): "banks" → "bank" (financial institution OR river side)
- As a verb: "banks" → "bank" (to tilt)

"""
    banks_noun = lemmatizer.lemmatize('banks', wordnet.NOUN)
    banks_verb = lemmatizer.lemmatize('banks', wordnet.VERB)
    report += f"   lemmatize('banks', NOUN) = '{banks_noun}'\n"
    report += f"   lemmatize('banks', VERB) = '{banks_verb}'\n\n"

    report += """
Example 6: "lying"
- As a verb (from "lie" - to recline): "lying" → "lie"
- As a verb (from "lie" - to deceive): "lying" → "lie"
- These are HOMOGRAPHS - same spelling, different meanings!

"""
    lying_verb = lemmatizer.lemmatize('lying', wordnet.VERB)
    report += f"   lemmatize('lying', VERB) = '{lying_verb}' (ambiguous source!)\n\n"

    report += """
-----------------------------------------------------------------------------

3. HOW THE LEMMATIZER RESOLVES AMBIGUITY

WordNetLemmatizer resolves ambiguity through:

a) POS Tag Input:
   - The lemmatizer accepts a part-of-speech parameter
   - Correct POS tagging is crucial for accurate lemmatization
   - Our pipeline uses NLTK's POS tagger to determine the likely POS

b) Default Behavior:
   - If no POS is specified, WordNetLemmatizer defaults to NOUN
   - This can lead to incorrect lemmas for verbs and adjectives

c) Context Ignorance:
   - WordNetLemmatizer does NOT consider sentence context
   - Semantic disambiguation (like distinguishing "bank" as financial
     institution vs. river bank) is NOT performed
   - This is a known limitation of rule-based lemmatizers

-----------------------------------------------------------------------------

4. STEMMING VS LEMMATIZATION COMPARISON

"""
    words = ['running', 'ran', 'runs', 'easily', 'fairly', 'better', 'best', 'studies', 'studying']
    stemmer = SnowballStemmer('english')

    report += f"{'Word':<15} {'Stem':<15} {'Lemma (default)':<15}\n"
    report += "-" * 45 + "\n"
    for word in words:
        stem = stemmer.stem(word)
        lemma = lemmatizer.lemmatize(word)
        report += f"{word:<15} {stem:<15} {lemma:<15}\n"

    report += """

Key Observations:
- Stemming is more aggressive (e.g., 'easily' → 'easili')
- Lemmatization preserves valid dictionary words
- Stemming doesn't consider POS; lemmatization can

-----------------------------------------------------------------------------

5. REAL-WORLD DATASET EXAMPLES

Processing sample sentences from our news dataset:

"""
    sample_sentences = [
        "The company saw record profits this quarter.",
        "I saw a saw in the hardware store.",
        "The leaves fall when autumn leaves arrive.",
        "He was meeting at the meeting room.",
        "She banks at the banks of the river.",
    ]

    for sentence in sample_sentences:
        report += f"\nSentence: \"{sentence}\"\n"
        results = processor.process_sentence(sentence)
        report += f"{'Token':<15} {'Stem':<15} {'Lemma':<15}\n"
        report += "-" * 45 + "\n"
        for token, stem, lemma in results:
            if token.isalpha():
                report += f"{token:<15} {stem:<15} {lemma:<15}\n"

    report += """
=============================================================================
                              END OF REPORT
=============================================================================
"""
    return report
