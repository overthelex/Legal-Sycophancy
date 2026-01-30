#!/usr/bin/env python3
"""
Statistical tests for RQ2 (Framing Effects).

For each evaluator and framing, test whether the framing significantly changes
accuracy compared to the predictive baseline.

Tests:
- McNemar's exact test on paired binary predictions
- Paired bootstrap confidence intervals (95%)
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List
from scipy.stats import binom

# Paths
BASE_DIR = Path("/Users/mac/Desktop/Code/llm-human-rights")
DATA_FILE = BASE_DIR / "data/experiments/rq2_framing/evaluations/detailed_results.csv"
OUTPUT_DIR = BASE_DIR / "data/experiments/rq2_framing/statistical_tests"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def mcnemar_test_exact(n01: int, n10: int) -> Tuple[int, float]:
    """
    Perform exact McNemar's test using binomial distribution.

    Args:
        n01: Number of cases where framing correct, baseline wrong
        n10: Number of cases where baseline correct, framing wrong

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


def alignment_ci(
    n_aligned: int,
    n_total: int,
    ci: float = 0.95
) -> Tuple[float, float]:
    """
    Calculate Wilson score confidence interval for alignment proportion.

    Args:
        n_aligned: Number of aligned cases
        n_total: Total number of cases
        ci: Confidence interval level

    Returns:
        Tuple of (ci_lower, ci_upper)
    """
    from scipy.stats import norm

    p = n_aligned / n_total
    z = norm.ppf(1 - (1 - ci) / 2)

    denominator = 1 + z**2 / n_total
    center = (p + z**2 / (2 * n_total)) / denominator
    margin = z * np.sqrt((p * (1 - p) / n_total + z**2 / (4 * n_total**2))) / denominator

    return max(0, center - margin), min(1, center + margin)


def main():
    """Run statistical tests for RQ2."""

    print("Loading RQ2 framing data...")
    df = pd.read_csv(DATA_FILE)

    evaluators = ['gpt-4o', 'gpt-5.2', 'claude-sonnet-4.5', 'deepseek-v3.2']
    framings = ['normative', 'factual']

    print(f"\nTesting {len(evaluators)} evaluators × {len(framings)} framings = {len(evaluators) * len(framings)} comparisons")
    print()

    all_results = []

    for evaluator in evaluators:
        print(f"\n{evaluator.upper()}")
        print("-" * 80)

        # Get predictive baseline
        baseline_df = df[(df['evaluator'] == evaluator) & (df['framing'] == 'predictive')]

        for framing in framings:
            framing_df = df[(df['evaluator'] == evaluator) & (df['framing'] == framing)]

            # Merge to get paired comparisons
            merged = baseline_df.merge(
                framing_df,
                on=['case_name', 'article', 'is_violation'],
                suffixes=('_baseline', '_framing')
            )

            # Calculate correctness
            baseline_correct = merged['is_accurate_baseline'].tolist()
            framing_correct = merged['is_accurate_framing'].tolist()

            # Calculate accuracies
            baseline_acc = sum(baseline_correct) / len(baseline_correct)
            framing_acc = sum(framing_correct) / len(framing_correct)
            acc_diff = framing_acc - baseline_acc

            # Build contingency table for McNemar's test
            n01 = sum(1 for b, f in zip(baseline_correct, framing_correct) if not b and f)  # framing better
            n10 = sum(1 for b, f in zip(baseline_correct, framing_correct) if b and not f)  # baseline better

            # McNemar's test
            n_discordant, p_value = mcnemar_test_exact(n01, n10)

            # Paired bootstrap CI
            ci_lower, ci_upper = paired_bootstrap_ci(baseline_correct, framing_correct)

            # Calculate alignment
            baseline_pred = (merged['avg_rating_baseline'] < 3).astype(int)
            framing_pred = (merged['avg_rating_framing'] < 3).astype(int)
            n_aligned = sum(baseline_pred == framing_pred)
            n_total = len(merged)
            alignment_rate = n_aligned / n_total
            align_ci_lower, align_ci_upper = alignment_ci(n_aligned, n_total)

            # Determine significance
            is_significant = p_value < 0.05

            result = {
                'evaluator': evaluator,
                'framing': framing,
                'baseline_acc': baseline_acc,
                'framing_acc': framing_acc,
                'acc_diff': acc_diff,
                'n_cases': len(merged),
                'n01': n01,
                'n10': n10,
                'n_discordant': n_discordant,
                'mcnemar_p': p_value,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'alignment_rate': alignment_rate,
                'alignment_ci_lower': align_ci_lower,
                'alignment_ci_upper': align_ci_upper,
                'significant': is_significant
            }

            all_results.append(result)

            sig_marker = "***" if is_significant else "ns"
            print(f"  {framing:<12s}: Δ={acc_diff:+.3f}, p={p_value:.4f} {sig_marker}, alignment={alignment_rate:.3f}")

    # Save results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(OUTPUT_DIR / 'rq2_statistical_tests.csv', index=False)

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
            print(f"  {r['evaluator']:<20s} + {r['framing']:<12s}: {direction} (Δ={r['acc_diff']:+.3f}, p={r['mcnemar_p']:.4f})")
    else:
        print("\nNo significant results found (all p >= 0.05)")

    print("\n" + "=" * 80)
    print("RESULTS SAVED")
    print("=" * 80)
    print(f"  CSV: {OUTPUT_DIR / 'rq2_statistical_tests.csv'}")
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
