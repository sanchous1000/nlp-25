# Lab Work #1 — English Text Preprocessing Pipeline

## Overview
This repository contains a preprocessing pipeline implemented for English text. The pipeline performs sentence segmentation, tokenization (with special handling for complex tokens), stemming, lemmatization, and generation of an annotated corpus in TSV format. The source dataset is the English Basic News Dataset (documents labeled by category).

Each processed document is saved as a TSV file with three tab-separated columns:

<token> <stem> <lemma>

Blank lines separate sentences in the output to preserve sentence boundaries.

## Pipeline components
- Sentence segmentation
- Tokenization with special handling for: emails, phone numbers, titles (Dr., Prof.), emails, hyperlinks and other formal constructions
- Stemming using SnowballStemmer
- Lemmatization using NLTK's WordNetLemmatizer (POS-aware)
- Export: an annotated TSV per document placed into a hierarchical directory structure mirroring class labels (see `assets/annotated_corpus`)

## Technologies and tools
- Dataset: English Basic News Dataset
- Libraries: datasets from huggingface, nltk
- Models/algorithms: SnowballStemmer, WordNetLemmatizer (NLTK)

## Handling complex cases
The tokenizer was designed to recognize and preserve complex tokens as single units. Examples:

- Email addresses: `john.doe@example.com` → single token
- Phone numbers: `+1-555-123-4567`, `555 987 6543`, `+1 (202) 555-0198` → not split across punctuation or brackets
- Titles with names: `Dr. Smith`, `Prof. Johnson` → treated as a single lexical unit when appropriate

These rules reduce incorrect fragmentation and improve downstream stemming/lemmatization quality.

## Example annotation fragment
Here is a short excerpt from the generated TSV for clarity (columns: token, stem, lemma):

| Token     | Stem     | Lemma     |
|-----------|----------|-----------|
| Sister    | sister   | Sister    |
| of        | of       | of        |
| man       | man      | man       |
| who       | who      | who       |
| died      | die      | die       |
| in        | in       | in        |
| Vancouver | vancouv  | Vancouver |
| police    | polic    | police    |
| custody   | custodi  | custody   |
| slams     | slam     | slam      |
| chief     | chief    | chief     |

## Performance (observed)
- Training set (120,000 lines): ~13 minutes
- Test set (7,600 lines): ~1 minute

Timings depend on hardware and environment. These numbers were measured during development on the provided dataset and are intended as rough guidance.

## Known issues and limitations
- Homonymy/context-dependent lemmatization: Some word forms map to different lemmas depending on POS or context. Example: "leaves" can be the verb `leave` or the noun `leaf` depending on context. The current pipeline uses POS tags to improve lemmatization, but ambiguous cases remain a challenge.
- Irregular verbs: Some irregular forms (e.g., `saw`) may be incorrectly resolved in some contexts. This is common with rule-based lemmatizers and can be mitigated with additional context or alternative lemmatization models.

## Project structure
- `source/` — preprocessing scripts: `annotate.py`, `main.py`, `text_processing.py`
- `assets/annotated_corpus/` — output TSV files organized by class label and split (train/test)
- `requirements.txt` — Python dependencies
- `run.ipynb` — notebook with runnable examples

## How to run (basic)
1. Create and activate a Python environment (recommended). Example PowerShell commands:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Conclusions
The lab work produced a robust preprocessing pipeline that handles many real-world tokenization challenges and produces a structured annotated corpus suitable for downstream tasks. The system is extensible — future work can improve context-aware lemmatization and add more coverage for specialized token types.
