#!/usr/bin/env python3
"""
Analyze directional changes in ratings for RQ2 (framing) and RQ3 (confidence challenge).

For each model, calculate:
- % of changes toward more violation (rating increase)
- % toward less violation (rating decrease)
- % moving to/from abstention (rating = 3)

This reveals whether models systematically shift toward or away from violation findings.
"""

import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple
import json

# Paths
BASE_DIR = Path("/Users/mac/Desktop/Code/llm-human-rights")
FRAMING_DIR = BASE_DIR / "data_updated/experiments/framing_comparison_multi"
CONFIDENCE_DIR = BASE_DIR / "data_updated/experiments/confidence_challenge_new"
OUTPUT_DIR = BASE_DIR / "data_updated/experiments/directional_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Rating scale: 1 (strong no violation) -> 5 (strong violation), 3 (abstention)


def analyze_rating_changes(
    baseline_ratings: List[float],
    comparison_ratings: List[float],
    condition_name: str
) -> Dict:
    """
    Analyze directional changes between baseline and comparison ratings.

    Returns:
        Dict with percentages of different types of changes
    """
    assert len(baseline_ratings) == len(comparison_ratings), "Rating lists must be same length"

    changes = []
    for baseline, comparison in zip(baseline_ratings, comparison_ratings):
        if baseline != comparison:
            changes.append({
                'baseline': baseline,
                'comparison': comparison,
                'direction': 'toward_violation' if comparison > baseline else 'away_from_violation',
                'magnitude': abs(comparison - baseline),
                'baseline_abstention': baseline == 3,
                'comparison_abstention': comparison == 3,
                'moved_to_abstention': baseline != 3 and comparison == 3,
                'moved_from_abstention': baseline == 3 and comparison != 3
            })

    total_changes = len(changes)
    if total_changes == 0:
        return {
            'condition': condition_name,
            'total_ratings': len(baseline_ratings),
            'total_changes': 0,
            'pct_changed': 0.0,
            'pct_toward_violation': 0.0,
            'pct_away_violation': 0.0,
            'pct_to_abstention': 0.0,
            'pct_from_abstention': 0.0,
            'avg_magnitude': 0.0
        }

    toward_violation = sum(1 for c in changes if c['direction'] == 'toward_violation')
    away_violation = sum(1 for c in changes if c['direction'] == 'away_from_violation')
    to_abstention = sum(1 for c in changes if c['moved_to_abstention'])
    from_abstention = sum(1 for c in changes if c['moved_from_abstention'])
    avg_magnitude = sum(c['magnitude'] for c in changes) / total_changes

    return {
        'condition': condition_name,
        'total_ratings': len(baseline_ratings),
        'total_changes': total_changes,
        'pct_changed': 100 * total_changes / len(baseline_ratings),
        'pct_toward_violation': 100 * toward_violation / total_changes,
        'pct_away_violation': 100 * away_violation / total_changes,
        'pct_to_abstention': 100 * to_abstention / total_changes,
        'pct_from_abstention': 100 * from_abstention / total_changes,
        'avg_magnitude': avg_magnitude
    }


def load_framing_data(evaluator: str) -> Tuple[List[float], List[float], List[float]]:
    """
    Load baseline (predictive), normative, and factual ratings for an evaluator.

    Returns:
        Tuple of (baseline_ratings, normative_ratings, factual_ratings)
    """
    detailed_df = pd.read_csv(FRAMING_DIR / "detailed_results.csv")

    # Filter by evaluator
    evaluator_df = detailed_df[detailed_df['evaluator'] == evaluator].copy()

    # Get baseline (predictive framing)
    baseline_df = evaluator_df[evaluator_df['framing'] == 'predictive'].sort_values(['case_name', 'article'])

    # Get normative framing
    normative_df = evaluator_df[evaluator_df['framing'] == 'normative'].sort_values(['case_name', 'article'])

    # Get factual framing
    factual_df = evaluator_df[evaluator_df['framing'] == 'factual'].sort_values(['case_name', 'article'])

    # Ensure same order by merging on case_name and article
    baseline_df = baseline_df.set_index(['case_name', 'article']).sort_index()
    normative_df = normative_df.set_index(['case_name', 'article']).sort_index()
    factual_df = factual_df.set_index(['case_name', 'article']).sort_index()

    # Get common indices
    common_idx = baseline_df.index.intersection(normative_df.index).intersection(factual_df.index)

    baseline_ratings = baseline_df.loc[common_idx, 'avg_rating'].tolist()
    normative_ratings = normative_df.loc[common_idx, 'avg_rating'].tolist()
    factual_ratings = factual_df.loc[common_idx, 'avg_rating'].tolist()

    return baseline_ratings, normative_ratings, factual_ratings


def load_confidence_challenge_data(evaluator: str) -> Tuple[List[float], List[float]]:
    """
    Load original and challenged ratings for an evaluator.

    Returns:
        Tuple of (original_ratings, challenged_ratings)
    """
    import ast

    # Load the individual evaluator file
    csv_path = CONFIDENCE_DIR / f"{evaluator}_confidence_challenge.csv"
    results_df = pd.read_csv(csv_path)

    # Sort by case_name and article for consistent ordering
    results_df = results_df.sort_values(['case_name', 'article'])

    # Parse the list strings and calculate average ratings
    original_ratings = []
    challenged_ratings = []

    for _, row in results_df.iterrows():
        # Parse the string representations of lists
        orig_list = ast.literal_eval(row['original_ratings'])
        chal_list = ast.literal_eval(row['challenged_ratings'])

        # Calculate average ratings
        original_ratings.append(sum(orig_list) / len(orig_list))
        challenged_ratings.append(sum(chal_list) / len(chal_list))

    return original_ratings, challenged_ratings


def main():
    """
    Analyze directional changes for RQ2 (framing) and RQ3 (confidence challenge).
    """
    evaluators = ['gpt-4o', 'gpt-5.2', 'claude-sonnet-4.5', 'deepseek-v3.2']

    print("=" * 100)
    print("RQ2: FRAMING EFFECT - DIRECTIONAL ANALYSIS")
    print("=" * 100)
    print()

    rq2_results = []

    for evaluator in evaluators:
        print(f"\n{evaluator.upper()}")
        print("-" * 80)

        try:
            baseline, normative, factual = load_framing_data(evaluator)

            # Analyze normative vs baseline
            normative_analysis = analyze_rating_changes(baseline, normative, 'Normative')
            print(f"  Predictive → Normative:")
            print(f"    Total changes: {normative_analysis['total_changes']}/{normative_analysis['total_ratings']} ({normative_analysis['pct_changed']:.1f}%)")
            print(f"    → Toward violation: {normative_analysis['pct_toward_violation']:.1f}%")
            print(f"    → Away from violation: {normative_analysis['pct_away_violation']:.1f}%")
            print(f"    → To abstention (3): {normative_analysis['pct_to_abstention']:.1f}%")
            print(f"    → From abstention (3): {normative_analysis['pct_from_abstention']:.1f}%")
            print(f"    Avg magnitude: {normative_analysis['avg_magnitude']:.2f}")

            # Analyze factual vs baseline
            factual_analysis = analyze_rating_changes(baseline, factual, 'Factual')
            print(f"\n  Predictive → Factual:")
            print(f"    Total changes: {factual_analysis['total_changes']}/{factual_analysis['total_ratings']} ({factual_analysis['pct_changed']:.1f}%)")
            print(f"    → Toward violation: {factual_analysis['pct_toward_violation']:.1f}%")
            print(f"    → Away from violation: {factual_analysis['pct_away_violation']:.1f}%")
            print(f"    → To abstention (3): {factual_analysis['pct_to_abstention']:.1f}%")
            print(f"    → From abstention (3): {factual_analysis['pct_from_abstention']:.1f}%")
            print(f"    Avg magnitude: {factual_analysis['avg_magnitude']:.2f}")

            rq2_results.append({
                'evaluator': evaluator,
                'normative': normative_analysis,
                'factual': factual_analysis
            })

        except Exception as e:
            print(f"  Error loading data: {e}")

    # Save RQ2 results
    rq2_df_rows = []
    for result in rq2_results:
        for framing in ['normative', 'factual']:
            analysis = result[framing]
            rq2_df_rows.append({
                'evaluator': result['evaluator'],
                'framing': framing,
                'total_ratings': analysis['total_ratings'],
                'total_changes': analysis['total_changes'],
                'pct_changed': analysis['pct_changed'],
                'pct_toward_violation': analysis['pct_toward_violation'],
                'pct_away_violation': analysis['pct_away_violation'],
                'pct_to_abstention': analysis['pct_to_abstention'],
                'pct_from_abstention': analysis['pct_from_abstention'],
                'avg_magnitude': analysis['avg_magnitude']
            })

    rq2_df = pd.DataFrame(rq2_df_rows)
    rq2_df.to_csv(OUTPUT_DIR / "rq2_directional_analysis.csv", index=False)

    print("\n\n" + "=" * 100)
    print("RQ3: CONFIDENCE CHALLENGE - DIRECTIONAL ANALYSIS")
    print("=" * 100)
    print()

    rq3_results = []

    for evaluator in evaluators:
        print(f"\n{evaluator.upper()}")
        print("-" * 80)

        try:
            original, challenged = load_confidence_challenge_data(evaluator)

            # Analyze challenged vs original
            challenge_analysis = analyze_rating_changes(original, challenged, 'After Challenge')
            print(f"  Original → After Challenge:")
            print(f"    Total changes: {challenge_analysis['total_changes']}/{challenge_analysis['total_ratings']} ({challenge_analysis['pct_changed']:.1f}%)")
            print(f"    → Toward violation: {challenge_analysis['pct_toward_violation']:.1f}%")
            print(f"    → Away from violation: {challenge_analysis['pct_away_violation']:.1f}%")
            print(f"    → To abstention (3): {challenge_analysis['pct_to_abstention']:.1f}%")
            print(f"    → From abstention (3): {challenge_analysis['pct_from_abstention']:.1f}%")
            print(f"    Avg magnitude: {challenge_analysis['avg_magnitude']:.2f}")

            rq3_results.append({
                'evaluator': evaluator,
                'analysis': challenge_analysis
            })

        except Exception as e:
            print(f"  Error loading data: {e}")

    # Save RQ3 results
    rq3_df_rows = []
    for result in rq3_results:
        analysis = result['analysis']
        rq3_df_rows.append({
            'evaluator': result['evaluator'],
            'total_ratings': analysis['total_ratings'],
            'total_changes': analysis['total_changes'],
            'pct_changed': analysis['pct_changed'],
            'pct_toward_violation': analysis['pct_toward_violation'],
            'pct_away_violation': analysis['pct_away_violation'],
            'pct_to_abstention': analysis['pct_to_abstention'],
            'pct_from_abstention': analysis['pct_from_abstention'],
            'avg_magnitude': analysis['avg_magnitude']
        })

    rq3_df = pd.DataFrame(rq3_df_rows)
    rq3_df.to_csv(OUTPUT_DIR / "rq3_directional_analysis.csv", index=False)

    # Save combined JSON summary
    summary = {
        'rq2_framing': rq2_results,
        'rq3_confidence_challenge': rq3_results
    }

    with open(OUTPUT_DIR / "directional_analysis_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print("\n\n" + "=" * 100)
    print("RESULTS SAVED")
    print("=" * 100)
    print(f"  RQ2 (Framing): {OUTPUT_DIR / 'rq2_directional_analysis.csv'}")
    print(f"  RQ3 (Confidence Challenge): {OUTPUT_DIR / 'rq3_directional_analysis.csv'}")
    print(f"  Summary (JSON): {OUTPUT_DIR / 'directional_analysis_summary.json'}")
    print()


if __name__ == "__main__":
    main()
