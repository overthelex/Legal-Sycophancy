#!/usr/bin/env python3
"""
RQ1: Summary Comparison Matrix Experiment (Queue-Based)

Evaluates a 2D matrix of:
- Rows: Text types (full text + summaries from different models)
- Columns: Evaluator models

Each cell contains accuracy/alignment metrics for that combination.

Usage:
    python experiments/summary_comparison_matrix_optimized.py --evaluator gpt-5.2 --summary gpt5_2_v3
    python experiments/summary_comparison_matrix_optimized.py --evaluator claude-sonnet-4.5 --all-summaries
    python experiments/summary_comparison_matrix_optimized.py --all-evaluators --all-summaries
"""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import List, Dict
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.queued_client import QueuedLLMClient
from lib.evaluation import prepare_evaluation_requests, process_evaluation_results
from lib.prompts import EVALUATION_SYSTEM_PROMPT, BASELINE_EVALUATION_TEMPLATE
from lib.metrics import calculate_accuracy

# Evaluator model configurations
EVALUATORS = {
    'gpt-4o': {
        'model_id': 'gpt-4o',
        'use_openrouter': False,
        'api_key_env': 'OPENAI_API_KEY'
    },
    'gpt-5.2': {
        'model_id': 'gpt-5.2',
        'use_openrouter': False,
        'api_key_env': 'OPENAI_API_KEY'
    },
    'claude-sonnet-4.5': {
        'model_id': 'anthropic/claude-sonnet-4.5',
        'use_openrouter': True,
        'api_key_env': 'OPENROUTER_API_KEY'
    },
    'deepseek-v3.2': {
        'model_id': 'deepseek/deepseek-v3.2',
        'use_openrouter': True,
        'api_key_env': 'OPENROUTER_API_KEY'
    }
}

# Summary file configurations
SUMMARY_CONFIGS = {
    'gpt4o_v1': 'data/summaries/gpt4o_v1.json',
    'gpt4o_v2': 'data/summaries/gpt4o_v2.json',
    'gpt4o_v3': 'data/summaries/gpt4o_v3.json',
    'gpt5_2_v1': 'data/summaries/gpt5_2_v1.json',
    'gpt5_2_v2': 'data/summaries/gpt5_2_v2.json',
    'gpt5_2_v3': 'data/summaries/gpt5_2_v3.json',
    'claude_sonnet_4_5_v1': 'data/summaries/claude_sonnet_4_5_v1.json',
    'claude_sonnet_4_5_v2': 'data/summaries/claude_sonnet_4_5_v2.json',
    'claude_sonnet_4_5_v3': 'data/summaries/claude_sonnet_4_5_v3.json',
    'deepseek_v3_2_v1': 'data/summaries/deepseek_v3_2_v1.json',
    'deepseek_v3_2_v2': 'data/summaries/deepseek_v3_2_v2.json',
    'deepseek_v3_2_v3': 'data/summaries/deepseek_v3_2_v3.json',
}


def load_summary_data(summary_key: str) -> List[Dict]:
    """Load summary data from file."""
    summary_path = Path(SUMMARY_CONFIGS[summary_key])

    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    with open(summary_path) as f:
        return json.load(f)


def evaluate_text_type(
    cases: List[Dict],
    evaluator_name: str,
    evaluator_config: Dict,
    api_key: str,
    text_field: str,
    num_samples: int,
    temperature: float,
    max_workers: int
) -> pd.DataFrame:
    """
    Evaluate cases using queue-based execution.

    Args:
        cases: List of case dictionaries
        evaluator_name: Name of evaluator
        evaluator_config: Evaluator configuration
        api_key: API key
        text_field: Which text field to evaluate ('full_case_text' or 'summary')
        num_samples: Number of samples per case
        temperature: Sampling temperature
        max_workers: Number of concurrent workers

    Returns:
        DataFrame with evaluation results
    """
    print(f"\n{'='*80}")
    print(f"Evaluating with {evaluator_name} on {text_field}")
    print(f"Cases: {len(cases)}, Samples: {num_samples}, Workers: {max_workers}")
    print(f"{'='*80}")

    # Prepare all evaluation requests
    print("\nPreparing requests...")
    requests = prepare_evaluation_requests(
        cases=cases,
        model_id=evaluator_config['model_id'],
        api_key=api_key,
        use_openrouter=evaluator_config['use_openrouter'],
        system_prompt=EVALUATION_SYSTEM_PROMPT,
        user_prompt_template=BASELINE_EVALUATION_TEMPLATE,
        num_samples=num_samples,
        temperature=temperature,
        max_tokens=100,
        case_text_key=text_field
    )

    # Execute all requests via queue
    print("\nExecuting evaluation requests...")
    client = QueuedLLMClient(max_workers=max_workers, timeout=120)

    def progress_callback(completed, total):
        if completed % 100 == 0 or completed == total:
            print(f"  Progress: {completed}/{total} ({100*completed/total:.1f}%)")

    results = client.execute_all(requests, progress_callback=progress_callback)

    # Process results
    print("\nProcessing results...")
    case_results = process_evaluation_results(results, cases, num_samples)

    # Build DataFrame
    rows = []
    for case_result in case_results:
        # Calculate binary prediction accuracy
        is_accurate = calculate_accuracy(
            case_result['avg_rating'],
            case_result['violation_label']
        )

        rows.append({
            'case_name': case_result['case_name'],
            'article': case_result['article'],
            'violation_label': case_result['violation_label'],
            'avg_rating': case_result['avg_rating'],
            'is_accurate': is_accurate,
            'num_abstentions': case_result['num_abstentions'],
            'num_samples': num_samples,
            'sample_ratings': str(case_result['sample_ratings'])
        })

    df = pd.DataFrame(rows)

    # Print summary statistics
    accuracy = df['is_accurate'].mean()
    abstention_rate = df['num_abstentions'].sum() / (len(df) * num_samples)
    print(f"\nResults:")
    print(f"  Accuracy: {accuracy:.1%}")
    print(f"  Abstention rate: {abstention_rate:.1%}")

    return df


def main():
    parser = argparse.ArgumentParser(description='RQ1: Summary Comparison Matrix Experiment')
    parser.add_argument('--evaluator', choices=list(EVALUATORS.keys()),
                        help='Evaluator model to use')
    parser.add_argument('--summary', choices=list(SUMMARY_CONFIGS.keys()),
                        help='Summary version to evaluate')
    parser.add_argument('--all-evaluators', action='store_true',
                        help='Run all evaluators')
    parser.add_argument('--all-summaries', action='store_true',
                        help='Evaluate all summary versions')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples per case (default: 10)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (default: 1.0)')
    parser.add_argument('--max-workers', type=int, default=50,
                        help='Number of concurrent workers (default: 50)')
    parser.add_argument('--output-dir', type=Path, default=Path('data/experiments/rq1_summarization'),
                        help='Output directory for results')

    args = parser.parse_args()

    # Validate arguments
    if not args.all_evaluators and not args.evaluator:
        parser.error('Must specify either --evaluator or --all-evaluators')
    if not args.all_summaries and not args.summary:
        parser.error('Must specify either --summary or --all-summaries')

    # Determine which evaluators and summaries to process
    evaluators_to_run = list(EVALUATORS.keys()) if args.all_evaluators else [args.evaluator]
    summaries_to_evaluate = list(SUMMARY_CONFIGS.keys()) if args.all_summaries else [args.summary]

    # Also evaluate full text baseline
    text_types = ['full_text'] + summaries_to_evaluate

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("RQ1: SUMMARY COMPARISON MATRIX EXPERIMENT")
    print("="*80)
    print(f"Evaluators: {', '.join(evaluators_to_run)}")
    print(f"Text types: {len(text_types)} (full_text + {len(summaries_to_evaluate)} summaries)")
    print(f"Samples per case: {args.num_samples}")
    print(f"Max workers: {args.max_workers}")
    print(f"Output: {args.output_dir}")
    print("="*80)

    # Load base case data
    base_cases_path = Path('data/processed/echr_cases_final_clean.json')
    print(f"\nLoading base cases from {base_cases_path}...")
    with open(base_cases_path) as f:
        base_cases = json.load(f)
    print(f"Loaded {len(base_cases)} cases")

    # Run evaluations
    for evaluator_name in evaluators_to_run:
        evaluator_config = EVALUATORS[evaluator_name]

        # Get API key
        api_key = os.getenv(evaluator_config['api_key_env'])
        if not api_key:
            print(f"\n⚠ Skipping {evaluator_name}: {evaluator_config['api_key_env']} not set")
            continue

        for text_type in text_types:
            # Load appropriate data
            if text_type == 'full_text':
                cases = base_cases
                text_field = 'full_case_text'
                text_label = 'full_text'
            else:
                # Load summary data
                summary_cases = load_summary_data(text_type)
                cases = summary_cases
                text_field = 'summary'
                text_label = text_type

            # Evaluate
            df = evaluate_text_type(
                cases=cases,
                evaluator_name=evaluator_name,
                evaluator_config=evaluator_config,
                api_key=api_key,
                text_field=text_field,
                num_samples=args.num_samples,
                temperature=args.temperature,
                max_workers=args.max_workers
            )

            # Save results
            output_filename = f"{evaluator_name}_{text_label}_samples{args.num_samples}.csv"
            output_path = args.output_dir / output_filename
            df.to_csv(output_path, index=False)
            print(f"✓ Saved results to {output_path}")

    print("\n" + "="*80)
    print("✓ RQ1 Summary Comparison Matrix Experiment Complete!")
    print("="*80)


if __name__ == '__main__':
    main()
