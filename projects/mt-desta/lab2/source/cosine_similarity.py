"""
Task 4: Cosine similarity/distance demonstrations.
"""

import numpy as np
from typing import List, Tuple, Dict
from sklearn.metrics.pairwise import cosine_similarity, cosine_distances


def cosine_distance(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine distance between two vectors.
    Cosine distance = 1 - cosine similarity
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine distance (0 = identical, 1 = orthogonal, 2 = opposite)
    """
    # Normalize vectors
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 1.0  # Maximum distance for zero vectors
    
    # Cosine similarity
    similarity = np.dot(vec1, vec2) / (norm1 * norm2)
    
    # Cosine distance
    distance = 1.0 - similarity
    
    return distance


def cosine_similarity_custom(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate cosine similarity between two vectors (custom implementation).
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Cosine similarity (-1 to 1)
    """
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return np.dot(vec1, vec2) / (norm1 * norm2)


def demonstrate_semantic_similarity(model, test_words: Dict[str, Dict[str, List[str]]]) -> Dict:
    """
    Demonstrate that semantically close words have smaller cosine distance.
    
    Args:
        model: NeuralVectorizer instance with trained model
        test_words: Dictionary mapping base words to groups:
            {
                'base_word': {
                    'similar': ['word1', 'word2'],
                    'related': ['word3', 'word4'],
                    'distant': ['word5', 'word6']
                }
            }
    
    Returns:
        Dictionary with results for each base word
    """
    results = {}
    
    for base_word, groups in test_words.items():
        base_vector = model.get_word_vector(base_word)
        
        if np.all(base_vector == 0):
            print(f"Warning: '{base_word}' not in vocabulary, skipping...")
            continue
        
        word_distances = {}
        
        # Calculate distances for each group
        for group_name, words in groups.items():
            group_distances = []
            for word in words:
                word_vector = model.get_word_vector(word)
                if np.any(word_vector != 0):
                    dist = cosine_distance(base_vector, word_vector)
                    group_distances.append((word, dist))
            
            # Sort by distance
            group_distances.sort(key=lambda x: x[1])
            word_distances[group_name] = group_distances
        
        results[base_word] = {
            'base_vector': base_vector,
            'distances': word_distances
        }
        
        # Print results
        print(f"\n{'='*60}")
        print(f"Base word: '{base_word}'")
        print(f"{'='*60}")
        
        for group_name in ['similar', 'related', 'distant']:
            if group_name in word_distances:
                print(f"\n{group_name.capitalize()} words:")
                for word, dist in word_distances[group_name]:
                    print(f"  {word:20s} - distance: {dist:.4f}")
    
    return results


def find_most_similar_words(model, word: str, top_n: int = 10) -> List[Tuple[str, float]]:
    """
    Find most similar words to a given word.
    
    Args:
        model: NeuralVectorizer instance
        word: Base word
        top_n: Number of similar words to return
        
    Returns:
        List of (word, similarity_score) tuples
    """
    if model.wv is None:
        raise ValueError("Model not trained")
    
    word_lower = word.lower().strip()
    if word_lower not in model.wv:
        return []
    
    similar_words = model.wv.most_similar(word_lower, topn=top_n)
    return similar_words

