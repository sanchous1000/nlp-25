# Lab Work #3.1 — Text Classification

## Overview
This repository contains a text classification pipeline implemented using Support Vector Machines (SVM) and other machine learning models. The pipeline uses vector representations of documents obtained from Lab 2 (text vectorization) to perform multiclass classification on the English Basic News Dataset. The implementation includes experiments with different kernel functions, iteration counts, and vector transformations to optimize classification performance.

The classification task assigns one of four categories to each news article:
- Class 0: World news
- Class 1: Sports news
- Class 2: Business news
- Class 3: Science/Technology news

## Pipeline components
- **Data loading**: Loading document embeddings from Lab 2 and corresponding labels from Lab 1
- **Custom metrics implementation**: Precision, recall, F1-score, and accuracy calculated from scratch (without library functions)
- **SVM classification**: Support Vector Machine with multiple kernel functions (linear, RBF, polynomial)
- **Model comparison**: Comparison between SVM and MLP (Multi-Layer Perceptron) classifiers
- **Hyperparameter experiments**: Systematic experiments with varying numbers of training iterations
- **Vector transformations**: Experiments with dropping dimensions, dimensionality reduction (PCA), and adding mathematical features
- **Performance evaluation**: Comprehensive metrics tracking including training time

## Technologies and tools
- **Dataset**: Preprocessed document embeddings from Lab 2 (AG News Dataset)
- **Libraries**: scikit-learn (SVM, MLP), numpy, pandas
- **Models/algorithms**: SVM (One-vs-Rest), MLP, PCA
- **Kernel functions**: Linear, RBF (Radial Basis Function), Polynomial

## Classification methods

### Support Vector Machine (SVM)
SVM finds optimal decision boundaries between classes in the feature space. The implementation uses:
- **One-vs-Rest (OvR)**: For multiclass classification, trains N binary classifiers (one per class)
- **Multiple kernels**: 
  - **Linear**: No transformation, works well for linearly separable data
  - **RBF**: Radial basis function kernel for non-linear decision boundaries
  - **Polynomial**: Polynomial kernel for capturing polynomial relationships

### Multi-Layer Perceptron (MLP)
Neural network classifier used for comparison:
- Hidden layer architecture: (100,) - single hidden layer with 100 neurons
- Activation: ReLU
- Solver: Adam optimizer

## Custom metrics implementation
All classification metrics are implemented from scratch without using library functions:

- **Accuracy**: Proportion of correctly classified documents
- **Precision** (macro-averaged): Average precision across all classes
- **Recall** (macro-averaged): Average recall across all classes
- **F1-score** (macro-averaged): Harmonic mean of precision and recall
- **Confusion matrix**: Per-class classification performance

The implementation includes both macro-averaged (equal weight per class) and micro-averaged (equal weight per sample) metrics.

## Experiment framework

### Task 1: Kernel and iteration experiments
Systematic experiments varying:
- **Kernel functions**: Linear, RBF, Polynomial
- **Max iterations**: 100, 500, 1000, 2000 epochs
- **Metrics tracked**: Accuracy, precision, recall, F1-score, training time

### Task 2: Optimal parameter selection
Analysis of experimental results to determine:
- Optimal kernel function
- Optimal number of training iterations
- Best performing model configuration

### Task 3: Vector transformation experiments
Additional experiments with modified feature vectors:

1. **Drop random dimensions**: Remove randomly selected features to study robustness
2. **Dimensionality reduction**: Apply PCA to reduce feature space dimensionality
3. **Add mathematical features**: Extend vectors with log, cos, sin, sqrt, and squared features

Each transformation is evaluated to understand its impact on classification performance.

## Example results structure
The experiments generate detailed results including:

```json
{
  "model_type": "SVM",
  "kernel": "linear",
  "max_iter": 1000,
  "accuracy": 0.8523,
  "precision": 0.8512,
  "recall": 0.8501,
  "f1_score": 0.8506,
  "training_time": 12.45
}
```

## Performance (observed)
Typical performance metrics on the AG News dataset:
- **SVM (Linear kernel)**: ~85-90% accuracy, ~2-15 seconds training time
- **SVM (RBF kernel)**: ~88-92% accuracy, ~5-20 seconds training time
- **SVM (Polynomial kernel)**: ~86-91% accuracy, ~8-25 seconds training time
- **MLP**: ~87-91% accuracy, ~10-30 seconds training time

Actual results depend on:
- Number of training iterations
- Kernel function and hyperparameters
- Feature vector dimensionality
- Hardware performance

## Known issues and limitations
- **Train/test split**: Currently uses test embeddings split for demonstration. In practice, separate train embeddings should be generated in Lab 2.
- **Limited hyperparameter tuning**: Only max_iter is varied systematically. Other hyperparameters (C, gamma) use default values.
- **Simple vector transformations**: The drop dimensions approach uses random selection rather than importance-based selection.
- **Computational complexity**: RBF and polynomial kernels can be computationally expensive for large datasets.
- **Class imbalance**: The current implementation doesn't explicitly handle potential class imbalances, though macro-averaging helps.

## Project structure
- `source/` — classification scripts: `data_loader.py`, `metrics.py`, `classifier.py`, `vector_transformations.py`, `experiments.py`, `main.py`
- `assets/results/` — JSON files with experiment results
- `requirements.txt` — Python dependencies
- `run.ipynb` — notebook with interactive experiments and visualizations

## How to run (basic)
1. Ensure Lab 1 and Lab 2 have been completed to generate the annotated corpus and embeddings.

2. Create and activate a Python environment (recommended). Example PowerShell commands:

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

3. Run the main classification pipeline:

```powershell
python source/main.py
```

This will:
- Load embeddings from `../lab2/assets/embeddings/test_embeddings.tsv`
- Load labels from `../lab1/assets/annotated_corpus/`
- Run SVM experiments with different kernels and iterations
- Perform vector transformation experiments
- Save results to `assets/results/`

4. For interactive exploration, use the Jupyter notebook:

```powershell
jupyter notebook run.ipynb
```

## Conclusions
The lab work produced a comprehensive text classification system that successfully classifies news articles into four categories using document embeddings. The experiments demonstrate that:

- **Kernel selection matters**: Different kernels (linear, RBF, polynomial) show varying performance characteristics
- **Iteration count optimization**: Finding the optimal number of training iterations balances performance and computational cost
- **Feature engineering impact**: Vector transformations (dimensionality reduction, feature addition) can significantly affect classification accuracy
- **SVM effectiveness**: SVM with appropriate kernel selection achieves strong performance on this text classification task

The system is extensible — future work can explore more sophisticated feature engineering, advanced neural architectures (BERT, ELMo), ensemble methods, and more comprehensive hyperparameter optimization. The custom metrics implementation provides transparency and understanding of the evaluation process.

