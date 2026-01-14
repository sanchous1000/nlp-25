"""
Experiment framework for Lab 3.2 - Topic Modeling
"""

import numpy as np
from scipy.sparse import csr_matrix
from typing import List, Dict, Any
from .topic_modeling import TopicModeler


def run_lda_experiments(
    train_matrix: csr_matrix,
    test_matrix: csr_matrix,
    vocabulary: List[str],
    n_topics_list: List[int],
    n_iter: int = 10
) -> List[Dict[str, Any]]:
    """
    Run LDA experiments with different numbers of topics.
    
    Args:
        train_matrix: Training term-document matrix (vocab_size, num_docs)
        test_matrix: Test term-document matrix (vocab_size, num_docs)
        vocabulary: List of vocabulary words
        n_topics_list: List of topic counts to test
        n_iter: Number of training iterations
    
    Returns:
        List of experiment results dictionaries
    """
    results = []
    
    for n_topics in n_topics_list:
        print(f"\nExperiment: n_topics={n_topics}, n_iter={n_iter}")
        
        # Initialize and train model
        modeler = TopicModeler(n_topics=n_topics, n_iter=n_iter)
        training_time = modeler.train(train_matrix)
        
        # Get top words
        top_words = modeler.get_top_words(vocabulary, n_words=10)
        
        # Calculate perplexity on test set
        perplexity = modeler.get_perplexity(test_matrix)
        
        # Get document-topic distribution
        doc_topic_dist = modeler.get_document_topic_distribution(train_matrix)
        
        # Get top documents per topic
        top_docs = modeler.get_top_documents_per_topic(doc_topic_dist, n_docs=10)
        
        # Store results
        result = {
            'n_topics': n_topics,
            'n_iter': n_iter,
            'perplexity': perplexity,
            'training_time': training_time,
            'top_words': top_words,
            'doc_topic_distribution': doc_topic_dist.tolist(),
            'top_documents': top_docs
        }
        
        results.append(result)
        
        # Print summary
        print(f"  Perplexity: {perplexity:.2f}, Time: {training_time:.2f}s")
        print(f"  Top words for topic 0: {[w[0] for w in top_words[0][:5]]}")
    
    return results


def run_iteration_experiments(
    train_matrix: csr_matrix,
    test_matrix: csr_matrix,
    vocabulary: List[str],
    n_topics: int,
    n_iters_list: List[int]
) -> List[Dict[str, Any]]:
    """
    Run LDA experiments with different iteration counts.
    
    Args:
        train_matrix: Training term-document matrix (vocab_size, num_docs)
        test_matrix: Test term-document matrix (vocab_size, num_docs)
        vocabulary: List of vocabulary words
        n_topics: Number of topics (fixed)
        n_iters_list: List of iteration counts to test
    
    Returns:
        List of experiment results dictionaries
    """
    results = []
    
    for n_iter in n_iters_list:
        print(f"\nIteration experiment: n_topics={n_topics}, n_iter={n_iter}")
        
        # Initialize and train model
        modeler = TopicModeler(n_topics=n_topics, n_iter=n_iter)
        training_time = modeler.train(train_matrix)
        
        # Calculate perplexity on test set
        perplexity = modeler.get_perplexity(test_matrix)
        
        # Store results
        result = {
            'n_topics': n_topics,
            'n_iter': n_iter,
            'perplexity': perplexity,
            'training_time': training_time
        }
        
        results.append(result)
        
        # Print summary
        print(f"  Perplexity: {perplexity:.2f}, Time: {training_time:.2f}s")
    
    return results

