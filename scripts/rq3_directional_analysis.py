#!/usr/bin/env python3
"""
Generate directional analysis table for RQ3 confidence challenge.

Analyzes:
- % of ratings that changed
- % of changes that moved toward violation (rating decreased)
- % of changes that moved away from violation (rating increased)
- Average magnitude of change when a change occurred
"""

import pandas as pd
import ast
from pathlib import Path


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

        # Track changes
        total_samples = 0
        total_changes = 0
        toward_violation = 0  # rating decreased (1 = violation, 5 = no violation)
        away_from_violation = 0  # rating increased
        total_magnitude = 0

        for _, row in df.iterrows():
            # Parse ratings
            original_ratings = ast.literal_eval(row['original_ratings'])
            challenged_ratings = ast.literal_eval(row['challenged_ratings'])

            # Analyze each rating
            for orig, chal in zip(original_ratings, challenged_ratings):
                total_samples += 1

                if orig != chal:
                    total_changes += 1
                    magnitude = abs(chal - orig)
                    total_magnitude += magnitude

                    # Determine direction
                    if chal < orig:
                        # Rating decreased: moved toward violation
                        toward_violation += 1
                    else:
                        # Rating increased: moved away from violation
                        away_from_violation += 1

        # Calculate statistics
        pct_changed = (total_changes / total_samples) * 100 if total_samples > 0 else 0
        pct_toward = (toward_violation / total_changes) * 100 if total_changes > 0 else 0
        pct_away = (away_from_violation / total_changes) * 100 if total_changes > 0 else 0
        avg_magnitude = total_magnitude / total_changes if total_changes > 0 else 0

        results.append({
            'Model': eval_name,
            '% Changed': f"{pct_changed:.1f}\\%",
            '% → Violation': f"{pct_toward:.1f}\\%",
            '% → Away': f"{pct_away:.1f}\\%",
            'Avg. Magnitude': f"{avg_magnitude:.2f}"
        })

        print(f"\n{eval_name}:")
        print(f"  Total samples: {total_samples}")
        print(f"  Changed: {total_changes} ({pct_changed:.1f}%)")
        print(f"  → Toward violation: {toward_violation} ({pct_toward:.1f}%)")
        print(f"  → Away from violation: {away_from_violation} ({pct_away:.1f}%)")
        print(f"  Average magnitude: {avg_magnitude:.2f}")

    # Create LaTeX table
    print("\n" + "="*80)
    print("LATEX TABLE:")
    print("="*80)

    print("\\clearpage")
    print("\\begin{table}[t]")
    print("\\centering")
    print("\\begin{tabular}{lcccc}")
    print("\\hline")
    print("\\textbf{Model} & \\textbf{\\% Changed} & \\textbf{\\% $\\rightarrow$ Violation} & \\textbf{\\% $\\rightarrow$ Away} & \\textbf{Avg. Magnitude} \\\\")
    print("\\hline")

    for r in results:
        print(f"{r['Model']:<20} & {r['% Changed']:>6} & {r['% → Violation']:>6} & {r['% → Away']:>7} & {r['Avg. Magnitude']:>4} \\\\")

    print("\\hline")
    print("\\end{tabular}")
    print("\\caption{Directional breakdown of judgment changes and average magnitude of change across models.}")
    print("\\label{tab:directional_change_stats}")
    print("\\end{table}")
    print()

    # Save results
    results_df = pd.DataFrame(results)
    output_file = Path("data/experiments/rq3_confidence/directional_analysis.csv")
    results_df.to_csv(output_file, index=False)
    print(f"Saved to: {output_file}")


if __name__ == "__main__":
    main()
