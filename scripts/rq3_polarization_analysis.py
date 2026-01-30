#!/usr/bin/env python3
"""
Generate polarization/moderation/flipping analysis for RQ3 confidence challenge.

Definitions (using 1-5 scale where 1=violation, 5=no violation, 3=uncertain):
- Polarizing: Move away from middle (3) - increases distance from 3
- Moderating: Move toward middle (3) - decreases distance from 3
- Flipping: Cross the middle (3) from one side to the other
"""

import pandas as pd
import ast
from pathlib import Path


def categorize_change(original, challenged):
    """Categorize a rating change."""
    if original == challenged:
        return None

    # Calculate distances from middle (3)
    orig_dist = abs(original - 3)
    chal_dist = abs(challenged - 3)

    # Check for flipping (crossing 3)
    if (original < 3 and challenged > 3) or (original > 3 and challenged < 3):
        return 'flipping'

    # Polarizing: moving away from 3
    if chal_dist > orig_dist:
        return 'polarizing'

    # Moderating: moving toward 3
    if chal_dist < orig_dist:
        return 'moderating'

    # Should not happen, but just in case
    return None


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

        # Track change types
        polarizing = 0
        moderating = 0
        flipping = 0
        total_changes = 0

        for _, row in df.iterrows():
            # Parse ratings
            original_ratings = ast.literal_eval(row['original_ratings'])
            challenged_ratings = ast.literal_eval(row['challenged_ratings'])

            # Analyze each rating change
            for orig, chal in zip(original_ratings, challenged_ratings):
                if orig != chal:
                    total_changes += 1
                    change_type = categorize_change(orig, chal)

                    if change_type == 'polarizing':
                        polarizing += 1
                    elif change_type == 'moderating':
                        moderating += 1
                    elif change_type == 'flipping':
                        flipping += 1

        # Calculate percentages
        pct_polarizing = (polarizing / total_changes * 100) if total_changes > 0 else 0
        pct_moderating = (moderating / total_changes * 100) if total_changes > 0 else 0
        pct_flipping = (flipping / total_changes * 100) if total_changes > 0 else 0

        results.append({
            'Model': eval_name,
            'Total Changes': total_changes,
            'Polarizing': f"{polarizing} ({pct_polarizing:.1f}\\%)",
            'Moderating': f"{moderating} ({pct_moderating:.1f}\\%)",
            'Flipping': f"{flipping} ({pct_flipping:.1f}\\%)"
        })

        print(f"\n{eval_name}:")
        print(f"  Total changes: {total_changes}")
        print(f"  Polarizing: {polarizing} ({pct_polarizing:.1f}%)")
        print(f"  Moderating: {moderating} ({pct_moderating:.1f}%)")
        print(f"  Flipping: {flipping} ({pct_flipping:.1f}%)")

    # Create LaTeX table
    print("\n" + "="*80)
    print("LATEX TABLE:")
    print("="*80)

    print("\\begin{table}[t]")
    print("\\centering")
    print("\\begin{tabular}{lcccc}")
    print("\\hline")
    print("\\textbf{Model} & \\textbf{Total Changes} & \\textbf{Polarizing} & \\textbf{Moderating} & \\textbf{Flipping} \\\\")
    print("\\hline")

    for r in results:
        print(f"{r['Model']:<18} & {r['Total Changes']:>3} & {r['Polarizing']:>15} & {r['Moderating']:>15} & {r['Flipping']:>15} \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Distribution of judgment changes by type across models. Percentages are relative to the total number of changes per model.}")
    print("\\label{tab:change_type_distribution}")
    print("\\end{table}")
    print()

    # Save results
    results_df = pd.DataFrame(results)
    output_file = Path("data/experiments/rq3_confidence/polarization_analysis.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
