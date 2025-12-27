# Lab Work #3.2 — Topic Modeling (Тематическое моделирование)

## Overview
This repository contains a topic modeling pipeline implemented using Latent Dirichlet Allocation (LDA). The pipeline performs topic modeling on the term-document matrix generated in Lab 2, identifies optimal number of topics, analyzes perplexity, and generates document-topic probability distributions.

The pipeline uses the term-document matrix from Lab 2 (Task 1) and applies LDA to discover latent topics in the document collection. It includes stop word filtering to improve topic quality and interpretability.

## Pipeline components
- **Data Loading**: Loads term-document matrix and vocabulary from Lab 2
- **Stop Word Filtering**: Removes uninformative tokens (articles, pronouns, common verbs, etc.)
- **Train/Test Split**: Splits documents for training and evaluation
- **LDA Topic Modeling**: Implements Latent Dirichlet Allocation using scikit-learn
- **Perplexity Analysis**: Evaluates model quality using perplexity metric
- **Polynomial Approximation**: Fits polynomial curves to perplexity data using R-squared
- **Optimal Topic Selection**: Finds optimal number of topics using elbow method and minimum perplexity
- **Document-Topic Distributions**: Generates probability distributions for each document-topic pair
- **Iteration Experiments**: Tests different training iteration counts

## Technologies and tools
- **Dataset**: Term-document matrix from Lab 2 (AG News dataset)
- **Libraries**: scikit-learn, numpy, scipy, pandas, matplotlib, seaborn
- **Models/Algorithms**: 
  - Latent Dirichlet Allocation (LDA) from scikit-learn
  - Polynomial regression for perplexity approximation
  - Elbow method for optimal topic selection

## Topic Modeling Methods

### LDA (Latent Dirichlet Allocation)
LDA is a generative probabilistic model that assumes documents are mixtures of topics, and topics are distributions over words. The model discovers:
- **Topic-word distributions**: Probability of each word in each topic
- **Document-topic distributions**: Probability of each topic in each document

### Stop Word Filtering
To improve topic quality, the pipeline filters:
- Articles and pronouns (the, a, this, that, etc.)
- Common verbs (say, get, go, make, etc.)
- Prepositions and conjunctions (of, in, on, and, or, etc.)
- News agency names (reuters, ap, etc.)
- Single characters and numbers
- HTML/XML artifacts

This filtering removes ~2.5% of vocabulary but dramatically improves topic interpretability.

## Tasks

### Task 1: LDA Experiments with Different Numbers of Topics
- Tests topic counts: 2, 4, 5, 10, 20, 40
- Calculates perplexity for each configuration
- Extracts top 10 words for each topic
- Generates document-topic probability distributions
- Identifies top documents for each topic

### Task 2: Perplexity Analysis with Polynomial Approximation
- Plots perplexity vs number of topics
- Fits polynomial curve using R-squared metric
- Finds best polynomial degree for approximation

### Task 3: Finding Optimal Number of Topics
- Uses elbow method to find optimal topic count
- Compares with minimum perplexity method
- Provides recommendations based on perplexity and interpretability

### Optional: Iteration Count Experiments
- Tests different training iteration counts (5, 10, 20)
- Analyzes perplexity vs training time trade-off
- Finds optimal iteration count

## Example Output

### Top Words for 10 Topics (After Stop Word Filtering)
- **Topic 0 (Business/Technology)**: company, new, million, plan, service, phone, software, business, mobile, deal
- **Topic 1 (Space/Science)**: space, new, year, nasa, flight, scientist, next, week, launch, season
- **Topic 2 (Sports)**: test, olympic, gold, win, athens, united, world, england, team, league
- **Topic 3 (Technology)**: microsoft, new, internet, red, web, search, security, computer, windows, software
- **Topic 4 (Economics)**: price, us, percent, high, year, ...

### Perplexity Results
| Topics | Perplexity | Training Time |
|--------|------------|---------------|
| 2      | 5784.11    | 108.92s       |
| 4      | 5908.99    | 81.35s        |
| 10     | 6492.86    | 63.95s        |
| 20     | 7790.34    | 62.94s        |
| 40     | 10588.52   | 62.04s        |

## Performance (observed)
- **Vocabulary size**: 34,413 tokens (after filtering, originally 35,311)
- **Documents**: 121,884 total (97,508 train, 24,376 test)
- **Stop words removed**: 898 tokens (2.5% reduction)
- **Training time**: 60-110 seconds depending on number of topics
- **Optimal topics**: 10 topics (recommended) or 2 topics (minimum perplexity)
- **Optimal iterations**: 20 iterations (best quality) or 10 iterations (good balance)

## Known issues and limitations
- Perplexity increases with stop word filtering (expected trade-off for quality)
- Training time varies with number of topics (counterintuitively decreases with more topics)
- Topic quality is evaluated qualitatively (no automatic coherence metric)
- Matrix format conversion required (lab2 uses (vocab_size, num_docs), sklearn expects (num_docs, vocab_size))

## Project structure
```
lab3.2/
├── source/
│   ├── __init__.py
│   ├── data_loader.py          # Loads term-document matrix from lab2
│   ├── topic_modeling.py        # LDA implementation
│   ├── experiments.py           # Experiment framework
│   ├── analysis.py              # Analysis functions (plotting, optimal topics)
│   └── main.py                  # Main execution script
├── assets/
│   └── results/
│       ├── lda_experiments.json              # Experiment results
│       ├── iteration_experiments.json        # Iteration experiment results
│       ├── perplexity_vs_topics.png          # Perplexity plot
│       └── distributions/
│           └── doc_topic_dist_n_topics_*.tsv  # Document-topic distributions
├── requirements.txt
├── README.md
└── run.ipynb                    # Jupyter notebook for interactive analysis
```

## Installation

1. **Install dependencies**:
```bash
pip install -r requirements.txt
```

2. **Ensure Lab 2 is completed**: The pipeline requires the term-document matrix from Lab 2 Task 1:
   - `lab2/assets/token_dictionary.pkl`
   - `lab2/assets/term_document_matrix.npz`

## How to run (basic)

### Option 1: Using the main script
```bash
cd lab3.2
python source/main.py
```

This will:
1. Load term-document matrix from `../lab2`
2. Run LDA experiments with different topic counts
3. Generate perplexity plots and polynomial approximations
4. Find optimal number of topics
5. Save all results to `assets/results/`

### Option 2: Using Jupyter Notebook
```bash
cd lab3.2
jupyter notebook run.ipynb
```

The notebook provides interactive analysis with step-by-step execution and visualizations.

## Configuration

### Stop Word Filtering
Stop word filtering is **enabled by default**. To disable, modify `source/data_loader.py`:
```python
dict_data, term_doc_matrix, vocabulary = load_vocabulary_from_lab2(
    lab2_dir,
    filter_stopwords=False  # Disable filtering
)
```

### Number of Topics
Modify `n_topics_list` in `source/main.py` or `run.ipynb`:
```python
n_topics_list = [2, 4, 5, 10, 20, 40]  # Customize as needed
```

### Training Iterations
Modify `n_iter` parameter:
```python
results = run_lda_experiments(
    train_matrix, test_matrix, vocabulary,
    n_topics_list, n_iter=20  # Increase for better quality
)
```

## Output Format

### Document-Topic Distribution (TSV)
Each document-topic distribution file contains:
- One row per document
- First column: document ID
- Remaining columns: probability of each topic (tab-separated)

Example:
```tsv
0	0.123456	0.234567	0.345678	...
1	0.234567	0.123456	0.456789	...
```

### Experiment Results (JSON)
Contains:
- `n_topics`: Number of topics
- `n_iter`: Number of iterations
- `perplexity`: Perplexity score
- `training_time`: Training time in seconds
- `top_words`: Top words for each topic
- `top_documents`: Top documents for each topic

## Conclusions

1. **Stop word filtering is highly recommended**: Dramatically improves topic quality and interpretability, with only a slight increase in perplexity.

2. **Optimal configuration**: 
   - **10 topics** provides good balance between perplexity and interpretability
   - **10-20 iterations** for training (10 for speed, 20 for quality)

3. **Topic quality**: After filtering, topics are clear, semantically coherent, and easily interpretable.

4. **Perplexity trade-off**: Higher perplexity with filtering is expected and acceptable for better topic quality.

5. **Practical applications**: The 10-topic model can be used for document categorization, content recommendation, trend analysis, and feature extraction.

