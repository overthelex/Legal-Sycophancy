#!/usr/bin/env python3
"""
Check completeness of RQ1 summary evaluations.

Expected: 141 case-article pairs for each evaluator-summary combination.
"""

import pandas as pd
from pathlib import Path
import json


def main():
    # Load original dataset to get expected count
    with open('data/processed/echr_cases_final_clean.json') as f:
        original_data = json.load(f)

    expected_count = len(original_data)
    print(f"Expected case-article pairs per evaluation: {expected_count}")
    print()

    # Check original text evaluations
    print("="*80)
    print("ORIGINAL TEXT EVALUATIONS")
    print("="*80)

    original_dir = Path('data/experiments/rq1_summarization/evaluations/original_text')
    original_files = sorted(original_dir.glob('*.csv'))

    for file_path in original_files:
        df = pd.read_csv(file_path)
        count = len(df)
        status = "✓ COMPLETE" if count == expected_count else f"✗ MISSING {expected_count - count}"
        print(f"{file_path.name:50} {count:3}/{expected_count}  {status}")

    # Check summary evaluations
    print()
    print("="*80)
    print("SUMMARY EVALUATIONS")
    print("="*80)

    detailed_dir = Path('data/experiments/rq1_summarization/evaluations/detailed')
    summary_files = sorted(detailed_dir.glob('*.csv'))

    # Group by evaluator
    evaluators = ['gpt-4o', 'gpt-5.2', 'claude-sonnet-4.5', 'deepseek-v3.2']

    incomplete_files = []

    for evaluator in evaluators:
        print(f"\n{evaluator}:")
        eval_files = [f for f in summary_files if f.name.startswith(evaluator)]

        for file_path in sorted(eval_files):
            df = pd.read_csv(file_path)
            count = len(df)
            status = "✓" if count == expected_count else f"✗ MISSING {expected_count - count}"

            if count != expected_count:
                incomplete_files.append({
                    'file': file_path.name,
                    'count': count,
                    'missing': expected_count - count
                })

            # Extract summary name
            summary_name = file_path.stem.replace(f"{evaluator}_summary_", "")
            print(f"  {summary_name:30} {count:3}/{expected_count}  {status}")

    # Summary of incomplete files
    print()
    print("="*80)
    print("SUMMARY OF INCOMPLETE EVALUATIONS")
    print("="*80)

    if incomplete_files:
        print(f"\nFound {len(incomplete_files)} incomplete evaluation files:")
        for item in incomplete_files:
            print(f"  {item['file']:55} Missing: {item['missing']:2} cases")

        print(f"\nTotal missing evaluations: {sum(item['missing'] for item in incomplete_files)}")
    else:
        print("\n✓ All evaluations are complete!")

    # Save incomplete list
    if incomplete_files:
        incomplete_df = pd.DataFrame(incomplete_files)
        output_file = 'data/experiments/rq1_summarization/incomplete_evaluations.csv'
        incomplete_df.to_csv(output_file, index=False)
        print(f"\nSaved incomplete file list to: {output_file}")


if __name__ == "__main__":
    main()
