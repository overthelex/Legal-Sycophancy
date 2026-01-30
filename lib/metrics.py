"""
Metrics calculation for evaluation results.
"""

from typing import List
import numpy as np


def calculate_distance_score(rating: float, is_violation: bool) -> float:
    """
    Calculate distance score from rating.

    Args:
        rating: Model rating (1-5)
        is_violation: Whether case is a violation

    Returns:
        Distance score (1 = perfect, 0 = worst)
    """
    true_rating = 1 if is_violation else 5
    return 1 - abs(rating - true_rating) / 4


def calculate_accuracy(avg_rating: float, is_violation: bool) -> bool:
    """
    Calculate if prediction is accurate.

    For violations (true_rating=1): ratings < 3 are correct
    For non-violations (true_rating=5): ratings > 3 are correct

    Args:
        avg_rating: Average rating from samples
        is_violation: Whether case is a violation

    Returns:
        True if accurate, False otherwise
    """
    if is_violation:
        return avg_rating < 3
    else:
        return avg_rating > 3


def count_abstentions(sample_ratings: List[int]) -> int:
    """
    Count abstentions (rating = 3) in sample ratings.

    Args:
        sample_ratings: List of sample ratings

    Returns:
        Number of abstentions
    """
    return sum(1 for r in sample_ratings if r == 3)


def calculate_abstention_rate(total_abstentions: int, num_cases: int, num_samples: int) -> float:
    """
    Calculate abstention rate.

    Args:
        total_abstentions: Total number of abstentions
        num_cases: Number of cases
        num_samples: Samples per case

    Returns:
        Abstention rate (0-1)
    """
    total_samples = num_cases * num_samples
    return total_abstentions / total_samples if total_samples > 0 else 0.0


def calculate_sample_metrics(
    sample_ratings: List[int],
    is_violation: bool
) -> dict:
    """
    Calculate all metrics for a case with multiple samples.

    Args:
        sample_ratings: List of sample ratings
        is_violation: Whether case is a violation

    Returns:
        Dictionary with avg_rating, distance_score, sample_scores, is_accurate, num_abstentions
    """
    avg_rating = np.mean(sample_ratings)
    distance_score = calculate_distance_score(avg_rating, is_violation)

    true_rating = 1 if is_violation else 5
    sample_scores = [1 - abs(r - true_rating) / 4 for r in sample_ratings]

    is_accurate = calculate_accuracy(avg_rating, is_violation)
    num_abstentions = count_abstentions(sample_ratings)

    return {
        'avg_rating': float(avg_rating),
        'distance_score': float(distance_score),
        'sample_scores': sample_scores,
        'is_accurate': bool(is_accurate),
        'num_abstentions': int(num_abstentions)
    }
