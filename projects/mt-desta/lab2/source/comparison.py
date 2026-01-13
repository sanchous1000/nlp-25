"""
Task 6 (Optional): Compare neural network vectorization vs basic methods with dimensionality reduction.
"""

import numpy as np
from typing import List, Dict
try:
    from .cosine_similarity import cosine_distance
    from .dimensionality_reduction import DimensionalityReducer
except ImportError:
    from cosine_similarity import cosine_distance
    from dimensionality_reduction import DimensionalityReducer


def compare_vectorization_methods(neural_vectors: np.ndarray, basic_vectors: np.ndarray,
                                 test_pairs: List[Tuple[int, int, str]]) -> Dict:
    """
    Compare effectiveness of neural network vs basic vectorization methods.
    
    Args:
        neural_vectors: Document vectors from neural network method
        basic_vectors: Document vectors from basic method (with dimensionality reduction)
        test_pairs: List of (doc_idx1, doc_idx2, expected_relationship) tuples
                   where relationship is 'similar' or 'distant'
    
    Returns:
        Dictionary with comparison results
    """
    results = {
        'neural': {'similar': [], 'distant': []},
        'basic': {'similar': [], 'distant': []}
    }
    
    for doc_idx1, doc_idx2, relationship in test_pairs:
        # Neural network vectors
        vec1_neural = neural_vectors[doc_idx1]
        vec2_neural = neural_vectors[doc_idx2]
        dist_neural = cosine_distance(vec1_neural, vec2_neural)
        results['neural'][relationship].append(dist_neural)
        
        # Basic method vectors
        vec1_basic = basic_vectors[doc_idx1]
        vec2_basic = basic_vectors[doc_idx2]
        dist_basic = cosine_distance(vec1_basic, vec2_basic)
        results['basic'][relationship].append(dist_basic)
    
    # Calculate statistics
    stats = {}
    for method in ['neural', 'basic']:
        stats[method] = {
            'similar_mean': np.mean(results[method]['similar']),
            'similar_std': np.std(results[method]['similar']),
            'distant_mean': np.mean(results[method]['distant']),
            'distant_std': np.std(results[method]['distant']),
            'separation': np.mean(results[method]['distant']) - np.mean(results[method]['similar'])
        }
    
    return {
        'results': results,
        'statistics': stats
    }


def print_comparison(comparison_results: Dict):
    """Print comparison results in a readable format."""
    stats = comparison_results['statistics']
    
    print("\n" + "=" * 60)
    print("Vectorization Method Comparison")
    print("=" * 60)
    
    for method in ['neural', 'basic']:
        method_name = 'Neural Network (Word2Vec)' if method == 'neural' else 'Basic Method (TF-IDF + PCA)'
        print(f"\n{method_name}:")
        print(f"  Similar documents - Mean distance: {stats[method]['similar_mean']:.4f} "
              f"(std: {stats[method]['similar_std']:.4f})")
        print(f"  Distant documents - Mean distance: {stats[method]['distant_mean']:.4f} "
              f"(std: {stats[method]['distant_std']:.4f})")
        print(f"  Separation (distant - similar): {stats[method]['separation']:.4f}")
    
    # Determine which method is better
    neural_separation = stats['neural']['separation']
    basic_separation = stats['basic']['separation']
    
    print("\n" + "-" * 60)
    if neural_separation > basic_separation:
        print("Conclusion: Neural network method provides better separation between")
        print("similar and distant documents.")
    elif basic_separation > neural_separation:
        print("Conclusion: Basic method with dimensionality reduction provides better")
        print("separation between similar and distant documents.")
    else:
        print("Conclusion: Both methods provide similar separation.")
    print("-" * 60)

