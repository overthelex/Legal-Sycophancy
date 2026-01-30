#!/usr/bin/env python3
"""
Comprehensive statistical tests for ALL summary versions.

For each evaluator, test whether EACH summary version significantly changes accuracy
compared to the baseline (original text).

Tests:
- McNemar's exact test on paired binary predictions
- Paired bootstrap confidence intervals (95%)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
from scipy.stats import binom
import json

# Paths
BASE_DIR = Path("/Users/mac/Desktop/Code/llm-human-rights")
BASELINE_DIR = BASE_DIR / "data/experiments/rq1_summarization/evaluations/original_text"
SUMMARY_DIR = BASE_DIR / "data/experiments/rq1_summarization/evaluations/detailed"
OUTPUT_DIR = BASE_DIR / "data/experiments/rq1_summarization/statistical_tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load ground truth
CASES_FILE = BASE_DIR / "data/processed/echr_cases_final_clean.json"


def load_ground_truth() -> dict:
    """Load ground truth labels."""
    import json
    with open(CASES_FILE) as f:
        cases = json.load(f)

    ground_truth = {}
    for case in cases:
        key = (case['case_name'], case['article'])
        ground_truth[key] = case['violation_label']

    return ground_truth


def load_predictions(csv_path: Path) -> Tuple[dict, dict]:
    """
    Load predictions and individual ratings from a CSV file.

    Returns:
        Tuple of (predictions dict, ratings dict)
        - predictions: {(case_name, article): 'violation' or 'no_violation'}
        - ratings: {(case_name, article): avg_rating}
    """
    df = pd.read_csv(csv_path)

    predictions = {}
    ratings = {}

    for _, row in df.iterrows():
        key = (row['case_name'], row['article'])
        avg_rating = row['avg_rating']
        prediction = 'violation' if avg_rating > 2.5 else 'no_violation'

        predictions[key] = prediction
        ratings[key] = avg_rating

    return predictions, ratings


def mcnemar_test_exact(n01: int, n10: int) -> Tuple[int, float]:
    """
    Perform exact McNemar's test using binomial distribution.

    Args:
        n01: Number of cases where comparison correct, baseline wrong
        n10: Number of cases where baseline correct, comparison wrong

    Returns:
        Tuple of (n, p_value)
    """
    n = n01 + n10
    if n == 0:
        return 0, 1.0

    # Two-tailed exact test
    p_value = 2 * min(
        binom.cdf(min(n01, n10), n, 0.5),
        1 - binom.cdf(min(n01, n10) - 1, n, 0.5)
    )

    # Cap at 1.0 to avoid floating-point precision issues
    p_value = min(p_value, 1.0)

    return n, p_value


def paired_bootstrap_ci(
    baseline_correct: List[bool],
    comparison_correct: List[bool],
    n_bootstrap: int = 10000,
    ci: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate paired bootstrap confidence interval for accuracy difference.

    Args:
        baseline_correct: List of boolean correctness for baseline
        comparison_correct: List of boolean correctness for comparison
        n_bootstrap: Number of bootstrap iterations
        ci: Confidence interval level (default 0.95)

    Returns:
        Tuple of (ci_lower, ci_upper)
    """
    baseline_correct = np.array(baseline_correct)
    comparison_correct = np.array(comparison_correct)

    diffs = comparison_correct.astype(int) - baseline_correct.astype(int)
    n = len(diffs)

    bootstrap_means = []
    rng = np.random.RandomState(42)

    for _ in range(n_bootstrap):
        sample_indices = rng.choice(n, size=n, replace=True)
        sample_diffs = diffs[sample_indices]
        bootstrap_means.append(np.mean(sample_diffs))

    alpha = 1 - ci
    ci_lower = np.percentile(bootstrap_means, alpha / 2 * 100)
    ci_upper = np.percentile(bootstrap_means, (1 - alpha / 2) * 100)

    return ci_lower, ci_upper


def test_summary_vs_baseline(
    evaluator: str,
    summary_version: str,
    ground_truth: dict
) -> dict:
    """
    Test whether a summary significantly changes accuracy compared to baseline.

    Args:
        evaluator: Name of evaluator model
        summary_version: Name of summary version
        ground_truth: Ground truth labels

    Returns:
        Dict with test results
    """
    # Load baseline predictions (original text)
    baseline_path = BASELINE_DIR / f"{evaluator}_original_text.csv"
    baseline_preds, baseline_ratings = load_predictions(baseline_path)

    # Load summary predictions
    summary_path = SUMMARY_DIR / f"{evaluator}_summary_{summary_version}.csv"
    summary_preds, summary_ratings = load_predictions(summary_path)

    # Get common keys
    common_keys = set(baseline_preds.keys()) & set(summary_preds.keys()) & set(ground_truth.keys())
    common_keys = sorted(common_keys)

    # Calculate correctness
    baseline_correct = []
    summary_correct = []

    for key in common_keys:
        baseline_correct.append(baseline_preds[key] == ground_truth[key])
        summary_correct.append(summary_preds[key] == ground_truth[key])

    # Calculate accuracies
    baseline_acc = sum(baseline_correct) / len(baseline_correct)
    summary_acc = sum(summary_correct) / len(summary_correct)
    acc_diff = summary_acc - baseline_acc

    # Build contingency table for McNemar's test
    n01 = sum(1 for b, s in zip(baseline_correct, summary_correct) if not b and s)  # summary better
    n10 = sum(1 for b, s in zip(baseline_correct, summary_correct) if b and not s)  # baseline better

    # McNemar's test
    n_discordant, p_value = mcnemar_test_exact(n01, n10)

    # Paired bootstrap CI
    ci_lower, ci_upper = paired_bootstrap_ci(baseline_correct, summary_correct)

    # Determine significance
    is_significant = p_value < 0.05

    return {
        'evaluator': evaluator,
        'summary_version': summary_version,
        'baseline_acc': baseline_acc,
        'summary_acc': summary_acc,
        'acc_diff': acc_diff,
        'n_cases': len(common_keys),
        'n01': n01,
        'n10': n10,
        'n_discordant': n_discordant,
        'mcnemar_p': p_value,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'significant': is_significant
    }


def main():
    """Run comprehensive statistical tests for all summaries."""

    # Load ground truth
    print("Loading ground truth...")
    ground_truth = load_ground_truth()

    # Get all evaluators and summaries
    evaluators = ['gpt-4o', 'gpt-5.2', 'claude-sonnet-4.5', 'deepseek-v3.2']

    # All 12 summary versions
    summary_versions = [
        'claude_sonnet_4_5_v1', 'claude_sonnet_4_5_v2', 'claude_sonnet_4_5_v3',
        'deepseek_v3_2_v1', 'deepseek_v3_2_v2', 'deepseek_v3_2_v3',
        'gpt4o_v1', 'gpt4o_v2', 'gpt4o_v3',
        'gpt5_2_v1', 'gpt5_2_v2', 'gpt5_2_v3'
    ]

    print(f"\nTesting {len(evaluators)} evaluators × {len(summary_versions)} summaries = {len(evaluators) * len(summary_versions)} comparisons")
    print()

    all_results = []

    for evaluator in evaluators:
        print(f"\n{evaluator.upper()}")
        print("-" * 80)

        for summary_version in summary_versions:
            try:
                result = test_summary_vs_baseline(evaluator, summary_version, ground_truth)
                all_results.append(result)

                sig_marker = "***" if result['significant'] else "ns"
                print(f"  {summary_version:<25s}: Δ={result['acc_diff']:+.3f}, p={result['mcnemar_p']:.4f} {sig_marker}")

            except FileNotFoundError as e:
                print(f"  {summary_version:<25s}: FILE NOT FOUND")
            except Exception as e:
                print(f"  {summary_version:<25s}: ERROR - {e}")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / 'rq1_comprehensive_tests.csv', index=False)

    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total tests: {len(all_results)}")
    print(f"Significant (p < 0.05): {sum(r['significant'] for r in all_results)}")
    print(f"Non-significant: {sum(not r['significant'] for r in all_results)}")
    print()

    # Identify significant results
    significant_results = [r for r in all_results if r['significant']]
    if significant_results:
        print("\nSIGNIFICANT RESULTS:")
        print("-" * 80)
        for r in significant_results:
            direction = "IMPROVES" if r['acc_diff'] > 0 else "WORSENS"
            print(f"  {r['evaluator']:<20s} + {r['summary_version']:<25s}: {direction} (Δ={r['acc_diff']:+.3f}, p={r['mcnemar_p']:.4f})")
    else:
        print("\nNo significant results found (all p >= 0.05)")

    print("\n" + "=" * 80)
    print("RESULTS SAVED")
    print("=" * 80)
    print(f"  CSV: {OUTPUT_DIR / 'rq1_comprehensive_tests.csv'}")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
