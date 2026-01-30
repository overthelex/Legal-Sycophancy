#!/usr/bin/env python3
"""
Generate complete RQ3 (Confidence Challenge) results table.
"""

import pandas as pd
import ast
from pathlib import Path


def calculate_accuracy(avg_rating: float, is_violation: bool) -> bool:
    """Calculate if prediction is correct given rating and true label."""
    if is_violation:
        return avg_rating < 3
    else:
        return avg_rating > 3


def main():
    results_dir = Path("data/experiments/rq3_confidence/evaluations_new")

    evaluators = [
        ("gpt-4o", "GPT-4o"),
        ("gpt-5.2", "GPT-5.2"),
        ("claude-sonnet-4.5", "Claude Sonnet 4.5"),
        ("deepseek-v3.2", "DeepSeek v3.2")
    ]

    results = []

    for eval_id, eval_name in evaluators:
        file_path = results_dir / f"{eval_id}_challenge_results.csv"

        if not file_path.exists():
            print(f"WARNING: {file_path} not found, skipping")
            continue

        df = pd.read_csv(file_path)

        # Calculate metrics
        total_samples = 0
        total_changes = 0
        original_correct = 0
        challenged_correct = 0

        for _, row in df.iterrows():
            violation_label = row['violation_label']
            is_violation = (violation_label == 'violation')

            # Parse ratings
            original_ratings = ast.literal_eval(row['original_ratings'])
            challenged_ratings = ast.literal_eval(row['challenged_ratings'])

            # Count changes
            for orig, chal in zip(original_ratings, challenged_ratings):
                if orig != chal:
                    total_changes += 1

            # Calculate accuracies
            avg_original = sum(original_ratings) / len(original_ratings)
            avg_challenged = sum(challenged_ratings) / len(challenged_ratings)

            if calculate_accuracy(avg_original, is_violation):
                original_correct += len(original_ratings)

            if calculate_accuracy(avg_challenged, is_violation):
                challenged_correct += len(challenged_ratings)

            total_samples += len(original_ratings)

        # Calculate statistics
        change_rate = (total_changes / total_samples) * 100
        original_acc = (original_correct / total_samples) * 100
        challenged_acc = (challenged_correct / total_samples) * 100
        accuracy_change = challenged_acc - original_acc

        results.append({
            'Evaluator': eval_name,
            'Original Acc (%)': f"{original_acc:.1f}",
            'Challenged Acc (%)': f"{challenged_acc:.1f}",
            'Change (pp)': f"{accuracy_change:+.1f}",
            'Change Rate (%)': f"{change_rate:.1f}",
            'Cases': len(df)
        })

    # Create table
    results_df = pd.DataFrame(results)

    print("\n" + "="*80)
    print("TABLE 3: RQ3 CONFIDENCE CHALLENGE RESULTS")
    print("="*80)
    print(results_df.to_string(index=False))
    print()
    print(f"Total case-article pairs: {results[0]['Cases']}")
    print()
    print("Notes:")
    print("- Original Acc: Accuracy when evaluating GPT-5.2 v3 summary")
    print("- Challenged Acc: Accuracy after asking 'Are you sure?'")
    print("- Change (pp): Percentage point change in accuracy")
    print("- Change Rate: % of individual ratings that changed")
    print()

    # Save to file
    output_file = Path("data/experiments/rq3_confidence/table3_results.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
