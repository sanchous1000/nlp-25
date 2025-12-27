"""
Main script for text classification experiments.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Add source to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from source.data_loader import load_test_data
from source.experiments import run_svm_experiments, run_model_comparison, run_vector_transformation_experiments
from source.metrics import classification_report


def main():
    """
    Main function to run classification experiments.
    """
    print("=" * 60)
    print("Lab 3.1 - Text Classification")
    print("=" * 60)
    
    # Configuration
    lab1_corpus_dir = "../lab1/assets/annotated_corpus"
    lab2_embeddings_file = "../lab2/assets/embeddings/test_embeddings.tsv"
    output_dir = "assets/results"
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Check if embeddings file exists
    if not os.path.exists(lab2_embeddings_file):
        print(f"Error: Embeddings file not found: {lab2_embeddings_file}")
        print("Please run lab2 first to generate embeddings.")
        return
    
    # Load test data (we'll use test set for experiments)
    # Note: For proper train/test split, we'd need train embeddings too
    print("\n[Step 1] Loading embeddings and labels...")
    X_test, y_test = load_test_data(lab1_corpus_dir, lab2_embeddings_file)
    
    print(f"Loaded {len(X_test)} test documents")
    print(f"Embedding dimension: {X_test.shape[1]}")
    print(f"Number of classes: {len(np.unique(y_test))}")
    print(f"Class distribution: {dict(zip(*np.unique(y_test, return_counts=True)))}")
    
    # Check for invalid labels
    invalid_mask = y_test == -1
    if np.any(invalid_mask):
        print(f"Warning: {np.sum(invalid_mask)} documents have invalid labels (-1)")
        # Remove invalid labels
        valid_mask = ~invalid_mask
        X_test = X_test[valid_mask]
        y_test = y_test[valid_mask]
        print(f"After filtering: {len(X_test)} documents")
    
    # Check if we have valid labels
    if len(np.unique(y_test)) < 2:
        print("Error: Need at least 2 classes for classification")
        print(f"Unique labels found: {np.unique(y_test)}")
        return
    
    # For demonstration, we'll split test data into train/test
    # Use stratified split to maintain class distribution
    from sklearn.model_selection import train_test_split
    
    X_train, X_test_split, y_train, y_test_split = train_test_split(
        X_test, y_test,
        test_size=0.5,
        stratify=y_test,  # Maintain class distribution
        random_state=42
    )
    
    print(f"\nUsing stratified split: {len(X_train)} train, {len(X_test_split)} test")
    print(f"Train class distribution: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"Test class distribution: {dict(zip(*np.unique(y_test_split, return_counts=True)))}")
    
    # Check data statistics
    print(f"\nEmbedding statistics:")
    print(f"  Mean: {np.mean(X_train):.6f}, Std: {np.std(X_train):.6f}")
    print(f"  Min: {np.min(X_train):.6f}, Max: {np.max(X_train):.6f}")
    
    # Task 1: Run SVM experiments with different kernels and iterations
    print("\n" + "=" * 60)
    print("Task 1: SVM Experiments with Different Kernels and Iterations")
    print("=" * 60)
    
    kernels = ['linear', 'rbf', 'poly']
    max_iters = [100, 500, 1000, 2000]
    
    svm_results = run_svm_experiments(
        X_train, y_train,
        X_test_split, y_test_split,
        kernels=kernels,
        max_iters=max_iters
    )
    
    # Save results
    results_file = os.path.join(output_dir, "svm_experiments.json")
    with open(results_file, 'w') as f:
        json.dump(svm_results, f, indent=2)
    print(f"\nSaved results to {results_file}")
    
    # Task 2: Find optimal parameters
    print("\n" + "=" * 60)
    print("Task 2: Analysis of Results")
    print("=" * 60)
    
    # Find best result
    best_result = max(svm_results, key=lambda x: x['f1_score'])
    print(f"\nBest result:")
    print(f"  Model: {best_result['model_type']}")
    print(f"  Kernel: {best_result['kernel']}")
    print(f"  Max iterations: {best_result['max_iter']}")
    print(f"  Accuracy: {best_result['accuracy']:.4f}")
    print(f"  F1-score: {best_result['f1_score']:.4f}")
    print(f"  Training time: {best_result['training_time']:.2f}s")
    
    optimal_kernel = best_result['kernel']
    optimal_max_iter = best_result['max_iter']
    
    # Task 3: Vector transformation experiments
    print("\n" + "=" * 60)
    print("Task 3: Vector Transformation Experiments")
    print("=" * 60)
    
    # Experiment 1: Drop random dimensions
    print("\n[Experiment 1] Dropping random dimensions...")
    drop_results = run_vector_transformation_experiments(
        X_train, y_train,
        X_test_split, y_test_split,
        model_type='SVM',
        kernel=optimal_kernel,
        optimal_max_iter=optimal_max_iter,
        transformation_type='drop_dimensions'
    )
    
    # Experiment 2: Reduce dimensionality with PCA
    print("\n[Experiment 2] Reducing dimensionality with PCA...")
    reduce_results = run_vector_transformation_experiments(
        X_train, y_train,
        X_test_split, y_test_split,
        model_type='SVM',
        kernel=optimal_kernel,
        optimal_max_iter=optimal_max_iter,
        transformation_type='reduce_dim'
    )
    
    # Experiment 3: Add mathematical features
    print("\n[Experiment 3] Adding mathematical features...")
    add_features_results = run_vector_transformation_experiments(
        X_train, y_train,
        X_test_split, y_test_split,
        model_type='SVM',
        kernel=optimal_kernel,
        optimal_max_iter=optimal_max_iter,
        transformation_type='add_features'
    )
    
    # Save transformation results
    transform_results = {
        'drop_dimensions': drop_results,
        'reduce_dim': reduce_results,
        'add_features': add_features_results
    }
    
    transform_file = os.path.join(output_dir, "transformation_experiments.json")
    with open(transform_file, 'w') as f:
        json.dump(transform_results, f, indent=2)
    print(f"\nSaved transformation results to {transform_file}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"Optimal kernel: {optimal_kernel}")
    print(f"Optimal max_iter: {optimal_max_iter}")
    print(f"Best F1-score: {best_result['f1_score']:.4f}")
    
    print("\nTransformation experiments completed!")
    print("Check results files for detailed metrics.")


if __name__ == "__main__":
    main()

