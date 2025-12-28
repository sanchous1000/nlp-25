"""
Analysis functions for Lab 3.2 - Topic Modeling

Includes perplexity plotting, polynomial approximation, and optimal topic finding.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Any, Tuple, Optional
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score


def plot_perplexity_vs_topics(
    results: List[Dict[str, Any]],
    save_path: Optional[str] = None
) -> Tuple[np.ndarray, float, int, LinearRegression, PolynomialFeatures]:
    """
    Plot perplexity vs number of topics and fit polynomial approximation.
    
    Args:
        results: List of experiment results
        save_path: Optional path to save figure
    
    Returns:
        Tuple of (coefficients, r2_score, best_degree, best_regressor, best_poly_features)
    """
    # Extract data
    n_topics = np.array([r['n_topics'] for r in results])
    perplexities = np.array([r['perplexity'] for r in results])
    
    # Sort by number of topics
    sort_idx = np.argsort(n_topics)
    n_topics = n_topics[sort_idx]
    perplexities = perplexities[sort_idx]
    
    # Try different polynomial degrees
    best_r2 = -np.inf
    best_degree = 1
    best_coefs = None
    best_reg = None
    best_poly_features = None
    
    for degree in range(1, min(7, len(n_topics))):
        # Create polynomial features
        poly_features = PolynomialFeatures(degree=degree, include_bias=True)
        X_poly = poly_features.fit_transform(n_topics.reshape(-1, 1))
        
        # Fit polynomial regression
        reg = LinearRegression()
        reg.fit(X_poly, perplexities)
        
        # Calculate R²
        y_pred = reg.predict(X_poly)
        r2 = r2_score(perplexities, y_pred)
        
        if r2 > best_r2:
            best_r2 = r2
            best_degree = degree
            best_coefs = reg.coef_
            best_reg = reg
            best_poly_features = poly_features
    
    # Generate smooth curve for plotting
    n_topics_smooth = np.linspace(n_topics.min(), n_topics.max(), 100)
    X_smooth_poly = best_poly_features.transform(n_topics_smooth.reshape(-1, 1))
    perplexities_smooth = best_reg.predict(X_smooth_poly)
    
    # Plot
    plt.figure(figsize=(10, 6))
    plt.plot(n_topics, perplexities, 'o', markersize=10, label='Experimental data', color='steelblue')
    plt.plot(n_topics_smooth, perplexities_smooth, '--', linewidth=2, label=f'Polynomial fit (degree {best_degree}, R²={best_r2:.4f})', color='coral')
    plt.xlabel('Number of Topics', fontsize=12)
    plt.ylabel('Perplexity', fontsize=12)
    plt.title('Perplexity vs Number of Topics', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return best_coefs, best_r2, best_degree, best_reg, best_poly_features


def find_optimal_topics(
    results: List[Dict[str, Any]],
    method: str = 'elbow'
) -> int:
    """
    Find optimal number of topics using different methods.
    
    Args:
        results: List of experiment results
        method: Method to use ('elbow' or 'min_perplexity')
    
    Returns:
        Optimal number of topics
    """
    # Extract data
    n_topics = np.array([r['n_topics'] for r in results])
    perplexities = np.array([r['perplexity'] for r in results])
    
    # Sort by number of topics
    sort_idx = np.argsort(n_topics)
    n_topics = n_topics[sort_idx]
    perplexities = perplexities[sort_idx]
    
    if method == 'min_perplexity':
        # Find minimum perplexity
        min_idx = np.argmin(perplexities)
        return int(n_topics[min_idx])
    
    elif method == 'elbow':
        # Elbow method: find point of maximum curvature
        # Calculate first and second derivatives
        if len(n_topics) < 3:
            # Fallback to minimum perplexity
            min_idx = np.argmin(perplexities)
            return int(n_topics[min_idx])
        
        # Calculate rate of change
        deltas = np.diff(perplexities)
        second_deltas = np.diff(deltas)
        
        # Find point with maximum second derivative (elbow)
        # We want the point where the rate of increase changes most
        if len(second_deltas) > 0:
            # Find maximum acceleration point
            elbow_idx = np.argmax(second_deltas) + 1
            return int(n_topics[elbow_idx])
        else:
            # Fallback
            min_idx = np.argmin(perplexities)
            return int(n_topics[min_idx])
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'elbow' or 'min_perplexity'.")


def polynomial_approximation(
    x: np.ndarray,
    y: np.ndarray,
    degree: int = 3
) -> Tuple[np.ndarray, float, LinearRegression, PolynomialFeatures]:
    """
    Fit polynomial approximation to data.
    
    Args:
        x: Input values
        y: Output values
        degree: Polynomial degree
    
    Returns:
        Tuple of (coefficients, r2_score, regressor, poly_features)
    """
    # Create polynomial features
    poly_features = PolynomialFeatures(degree=degree, include_bias=True)
    X_poly = poly_features.fit_transform(x.reshape(-1, 1))
    
    # Fit polynomial regression
    reg = LinearRegression()
    reg.fit(X_poly, y)
    
    # Calculate R²
    y_pred = reg.predict(X_poly)
    r2 = r2_score(y, y_pred)
    
    return reg.coef_, r2, reg, poly_features

