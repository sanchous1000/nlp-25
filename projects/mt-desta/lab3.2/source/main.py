"""
Main script for Lab 3.2 - Topic Modeling

Runs all experiments and saves results.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path

# Add source to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from source.data_loader import load_vocabulary_from_lab2, split_train_test
from source.topic_modeling import TopicModeler
from source.experiments import run_lda_experiments, run_iteration_experiments
from source.analysis import plot_perplexity_vs_topics, find_optimal_topics


def main():
    """Main execution function."""
    # Configuration
    lab2_dir = "../lab2"
    output_dir = "assets/results"
    distributions_dir = os.path.join(output_dir, "distributions")
    
    # Create output directories
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(distributions_dir, exist_ok=True)
    
    # Load data
    print("=" * 60)
    print("Lab 3.2 - Topic Modeling")
    print("=" * 60)
    print("\nStep 1: Loading term-document matrix and vocabulary from lab2...")
    
    dict_data, term_doc_matrix, vocabulary = load_vocabulary_from_lab2(
        lab2_dir,
        filter_stopwords=True  # Enable stop word filtering
    )
    
    print(f"\nLoaded:")
    print(f"  Vocabulary size: {len(vocabulary)}")
    print(f"  Number of documents: {term_doc_matrix.shape[1]}")
    print(f"  Matrix shape: {term_doc_matrix.shape}")
    
    # Split into train and test
    print("\nStep 2: Splitting into train and test sets...")
    train_matrix, test_matrix = split_train_test(
        term_doc_matrix,
        test_size=0.2,
        random_state=42
    )
    
    print(f"  Train: {train_matrix.shape[1]} documents")
    print(f"  Test: {test_matrix.shape[1]} documents")
    
    # Task 1: LDA experiments with different numbers of topics
    print("\n" + "=" * 60)
    print("Task 1: LDA Experiments with Different Numbers of Topics")
    print("=" * 60)
    
    # Number of classes in dataset (AG News has 4 classes)
    num_classes = 4
    n_topics_list = [2, 5, 10, 20, 40]
    if num_classes not in n_topics_list:
        n_topics_list.append(num_classes)
    n_topics_list = sorted(n_topics_list)
    
    print(f"\nTesting number of topics: {n_topics_list}")
    
    results = run_lda_experiments(
        train_matrix,
        test_matrix,
        vocabulary,
        n_topics_list,
        n_iter=10
    )
    
    print(f"\nCompleted {len(results)} experiments")
    
    # Save results
    results_file = os.path.join(output_dir, "lda_experiments.json")
    # Convert numpy arrays to lists for JSON serialization
    results_json = []
    for r in results:
        r_json = r.copy()
        r_json['doc_topic_distribution'] = np.array(r['doc_topic_distribution']).tolist()
        # Convert top_words to serializable format
        r_json['top_words'] = {
            str(k): [(w, float(p)) for w, p in v]
            for k, v in r['top_words'].items()
        }
        # Convert top_documents to serializable format
        r_json['top_documents'] = {
            str(k): [(int(d), float(p)) for d, p in v]
            for k, v in r['top_documents'].items()
        }
        results_json.append(r_json)
    
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(results_json, f, indent=2)
    
    print(f"Saved experiment results to {results_file}")
    
    # Save document-topic distributions
    print("\nSaving document-topic distributions...")
    for result in results:
        n_topics = result['n_topics']
        doc_topic_dist = np.array(result['doc_topic_distribution'])
        
        output_file = os.path.join(
            distributions_dir,
            f"doc_topic_dist_n_topics_{n_topics}.tsv"
        )
        
        with open(output_file, 'w', encoding='utf-8') as f:
            for doc_idx, probs in enumerate(doc_topic_dist):
                doc_id = str(doc_idx)
                prob_str = '\t'.join([f"{p:.6f}" for p in probs])
                f.write(f"{doc_id}\t{prob_str}\n")
        
        print(f"  Saved {n_topics} topics distribution to {output_file}")
    
    # Task 2: Perplexity analysis with polynomial approximation
    print("\n" + "=" * 60)
    print("Task 2: Perplexity Analysis with Polynomial Approximation")
    print("=" * 60)
    
    coefs, r2, degree, best_reg, best_poly_features = plot_perplexity_vs_topics(
        results,
        save_path=os.path.join(output_dir, "perplexity_vs_topics.png")
    )
    
    print(f"\nPolynomial Approximation:")
    print(f"  Best degree: {degree}")
    print(f"  R-squared: {r2:.4f}")
    print(f"  Coefficients: {coefs}")
    
    # Task 3: Find optimal number of topics
    print("\n" + "=" * 60)
    print("Task 3: Finding Optimal Number of Topics")
    print("=" * 60)
    
    optimal_topics_elbow = find_optimal_topics(results, method='elbow')
    optimal_topics_min = find_optimal_topics(results, method='min_perplexity')
    
    print(f"\nOptimal Number of Topics:")
    print(f"  Elbow method: {optimal_topics_elbow} topics")
    print(f"  Minimum perplexity: {optimal_topics_min} topics")
    
    print("\nPerplexity by number of topics:")
    for r in sorted(results, key=lambda x: x['n_topics']):
        print(f"  {r['n_topics']:2d} topics: {r['perplexity']:8.2f}")
    
    # Optional: Iteration count experiments
    print("\n" + "=" * 60)
    print("Optional: Iteration Count Experiments")
    print("=" * 60)
    
    base_iter = 10
    n_iters_list = [base_iter // 2, base_iter, base_iter * 2]
    test_n_topics = optimal_topics_elbow
    
    print(f"\nTesting iteration counts: {n_iters_list}")
    print(f"Using {test_n_topics} topics (optimal from previous experiments)")
    
    iter_results = run_iteration_experiments(
        train_matrix,
        test_matrix,
        vocabulary,
        test_n_topics,
        n_iters_list
    )
    
    # Save iteration results
    iter_results_file = os.path.join(output_dir, "iteration_experiments.json")
    with open(iter_results_file, 'w', encoding='utf-8') as f:
        json.dump(iter_results, f, indent=2)
    
    print(f"\nSaved iteration experiment results to {iter_results_file}")
    
    # Find optimal iterations
    best_iter_result = min(iter_results, key=lambda x: x['perplexity'])
    print(f"\nOptimal number of iterations: {best_iter_result['n_iter']}")
    print(f"  Perplexity: {best_iter_result['perplexity']:.2f}")
    print(f"  Training time: {best_iter_result['training_time']:.2f}s")
    
    print("\n" + "=" * 60)
    print("All experiments completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()

