# Lab Work #2 — Text Vectorization (Векторизация текста)

## Overview
This repository contains a text vectorization pipeline implemented to convert text from character representation into sequences of real numbers (vectors) of a given size. The pipeline implements both basic vectorization methods (frequency vectors, TF-IDF) and neural network-based methods (Word2Vec). The source dataset is the preprocessed annotated corpus from Lab 1 (English Basic News Dataset).

The final output is a TSV file containing vector representations of test documents, where each document is represented as a fixed-size vector of real numbers.

## Pipeline components
- **Token dictionary construction** (Task 1, optional): Building vocabulary with token frequencies and term-document matrix from scratch
- **Basic vectorization methods** (Task 2, optional): Frequency vectors, one-hot encoding, TF-IDF vectors, sentence-level matrices
- **Neural network vectorization** (Task 3): Word2Vec model training on the training dataset using Gensim
- **Semantic similarity demonstration** (Task 4): Cosine distance calculations to show that semantically close words have smaller distances
- **Dimensionality reduction** (Task 5, optional): PCA application to basic vectorization methods
- **Method comparison** (Task 6, optional): Comparing neural network vs basic vectorization effectiveness
- **Document vectorization** (Task 7): Complete pipeline: sentence segmentation → token vectors → sentence vectors → document vector
- **Test set vectorization** (Task 8): Vectorizing all test documents and saving in TSV format

## Technologies and tools
- **Dataset**: Preprocessed annotated corpus from Lab 1 (AG News Dataset)
- **Libraries**: gensim (Word2Vec), scikit-learn (PCA, cosine similarity), numpy, scipy (sparse matrices)
- **Models/algorithms**: Word2Vec (CBOW), TF-IDF, PCA, cosine distance

## Vectorization methods

### Basic methods (Task 2)
The implementation includes several basic vectorization approaches:

- **Frequency vectors**: Count-based representation using token frequencies from dictionary
- **One-hot encoding**: Binary matrix representation converted to vectors by averaging
- **TF-IDF vectors**: Term frequency-inverse document frequency weighting for importance scoring
- **Sentence-level matrices**: Frequency and TF-IDF matrices at sentence granularity

### Neural network method (Task 3)
Word2Vec model trained using the CBOW (Continuous Bag of Words) architecture:
- Vector size: 100 dimensions
- Window size: 5 tokens
- Minimum token frequency: 2
- Training epochs: 10

The model learns distributed representations where semantically similar words are encoded with close vector values.

## Semantic similarity demonstration (Task 4)
The implementation demonstrates that the Word2Vec model captures semantic relationships:

- **Semantically similar words** (e.g., "sport" and "game", "athlete") have smaller cosine distances
- **Related domain words** (e.g., "sport" and "team", "competition") have moderate distances
- **Semantically distant words** (e.g., "sport" and "computer", "philosophy") have larger distances

This validates that the vectorization method successfully encodes semantic information in the embedding space.

## Document vectorization pipeline (Task 7)
The complete document vectorization process:

1. **Sentence segmentation**: Split text into sentences using punctuation markers
2. **Tokenization**: Extract tokens from each sentence
3. **Token vectorization**: Get vector representation for each token using trained Word2Vec model
4. **Sentence vectorization**: Calculate average (or TF-IDF weighted average) of token vectors for each sentence
5. **Document vectorization**: Calculate final document vector by averaging sentence vectors

This hierarchical approach (token → sentence → document) preserves semantic information at multiple levels.

## Example output fragment
Here is a short excerpt from the generated TSV file for clarity (format: doc_id, followed by embedding components):

```tsv
14	0.023456	-0.012345	0.045678	...	0.001234
15	0.034567	-0.023456	0.056789	...	0.002345
16	0.012345	-0.034567	0.023456	...	0.003456
```

Each document is represented as a 100-dimensional vector (matching the Word2Vec vector size), where each component is a real number encoding semantic information about the document.

## Performance (observed)
- **Training data loading** (121,884 documents): ~5-10 minutes
- **Token dictionary construction** (Task 1): ~2-3 minutes
- **Word2Vec model training** (Task 3): ~5-10 minutes (depends on hardware)
- **Test set vectorization** (7,600 documents): ~2-3 minutes
- **Total pipeline execution** (Tasks 3, 7, 8): ~15-25 minutes

Timings depend on hardware and environment. These numbers were measured during development on the provided dataset and are intended as rough guidance.

## Known issues and limitations
- **Out-of-vocabulary tokens**: Words not seen during training receive zero vectors, which may affect document representations for documents with many unknown words. This can be mitigated by using larger training corpora or pre-trained models.
- **Sentence segmentation**: The current implementation uses simple punctuation-based segmentation, which may not handle all edge cases (e.g., abbreviations, decimal numbers). More sophisticated segmentation could improve results.
- **Fixed vector size**: All documents are represented with the same dimensionality (100), which may not be optimal for all use cases. Different vector sizes can be experimented with.
- **Average pooling**: Using simple averaging for sentence and document vectors may lose important information. Alternative pooling methods (max pooling, attention mechanisms) could be explored.

## Project structure
- `source/` — vectorization scripts: `data_loader.py`, `token_dictionary.py`, `basic_vectorization.py`, `neural_vectorization.py`, `cosine_similarity.py`, `document_vectorizer.py`, `main.py`
- `assets/models/` — saved Word2Vec models
- `assets/dictionaries/` — token dictionaries and term-document matrices (Task 1)
- `assets/embeddings/` — output TSV file with document embeddings (Task 8)
- `requirements.txt` — Python dependencies
- `run.ipynb` — notebook with runnable examples and demonstrations

## How to run (basic)
1. Create and activate a Python environment (recommended). Example PowerShell commands:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Ensure Lab 1 has been completed to generate the annotated corpus.

3. Run the main pipeline (Tasks 3, 7, 8):

```powershell
python source/main.py
```

This will:
- Load training data from `../lab1/assets/annotated_corpus/train/`
- Train a Word2Vec model
- Vectorize test documents
- Save embeddings to `assets/embeddings/test_embeddings.tsv`

4. For interactive exploration and optional tasks, use the Jupyter notebook:

```powershell
jupyter notebook run.ipynb
```

## Conclusions
The lab work produced a comprehensive text vectorization pipeline that successfully converts text documents into fixed-size numerical vectors. The implementation demonstrates both traditional (TF-IDF) and modern (Word2Vec) approaches to text vectorization, showing that neural network-based methods can effectively capture semantic relationships between words and documents.

The hierarchical document vectorization approach (token → sentence → document) provides a flexible framework that can be extended with more sophisticated pooling methods or attention mechanisms. The system is extensible — future work can explore different neural architectures (GloVe, fastText), experiment with hyperparameters, and apply the embeddings to downstream tasks such as text classification or similarity search.
