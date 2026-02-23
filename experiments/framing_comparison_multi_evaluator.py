#!/usr/bin/env python3
"""
RQ2: Framing Comparison Experiment (Queue-Based)

Tests how different question framings affect LLM judgments:
- Predictive: "will rule a violation"
- Normative: "should rule a violation"
- Factual: "a violation occurred"

Usage:
    python experiments/framing_comparison_multi_evaluator.py --evaluator gpt-5.2
    python experiments/framing_comparison_multi_evaluator.py --all-evaluators
"""

import argparse
import json
import os
from pathlib import Path
import sys
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.queued_client import QueuedLLMClient
from lib.evaluation import prepare_evaluation_requests, process_evaluation_results
from lib.prompts import EVALUATION_SYSTEM_PROMPT, PROMPT_FRAMINGS
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

# Framings to test
FRAMINGS = ['predictive', 'normative', 'factual']


def evaluate_framing(
    cases,
    evaluator_name,
    evaluator_config,
    api_key,
    framing_name,
    num_samples,
    temperature,
    max_workers
) -> pd.DataFrame:
    """
    Evaluate cases with a specific framing using queue-based execution.

    Args:
        cases: List of case dictionaries
        evaluator_name: Name of evaluator
        evaluator_config: Evaluator configuration
        api_key: API key
        framing_name: Name of framing ('predictive', 'normative', 'factual')
        num_samples: Number of samples per case
        temperature: Sampling temperature
        max_workers: Number of concurrent workers

    Returns:
        DataFrame with evaluation results
    """
    framing_template = PROMPT_FRAMINGS[framing_name]['template']

    print(f"\n{'='*80}")
    print(f"Evaluating {evaluator_name} with {framing_name} framing")
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
        user_prompt_template=framing_template,
        num_samples=num_samples,
        temperature=temperature,
        max_tokens=100,
        case_text_key='full_case_text'
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
    parser = argparse.ArgumentParser(description='RQ2: Framing Comparison Experiment')
    parser.add_argument('--evaluator', choices=list(EVALUATORS.keys()),
                        help='Evaluator model to use')
    parser.add_argument('--all-evaluators', action='store_true',
                        help='Run all evaluators')
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples per case (default: 10)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (default: 1.0)')
    parser.add_argument('--max-workers', type=int, default=50,
                        help='Number of concurrent workers (default: 50)')
    parser.add_argument('--output-dir', type=Path, default=Path('data/experiments/rq2_framing'),
                        help='Output directory for results')

    args = parser.parse_args()

    # Validate arguments
    if not args.all_evaluators and not args.evaluator:
        parser.error('Must specify either --evaluator or --all-evaluators')

    # Determine which evaluators to run
    evaluators_to_run = list(EVALUATORS.keys()) if args.all_evaluators else [args.evaluator]

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("="*80)
    print("RQ2: FRAMING COMPARISON EXPERIMENT")
    print("="*80)
    print(f"Evaluators: {', '.join(evaluators_to_run)}")
    print(f"Framings: {', '.join(FRAMINGS)}")
    print(f"Samples per case: {args.num_samples}")
    print(f"Max workers: {args.max_workers}")
    print(f"Output: {args.output_dir}")
    print("="*80)

    # Load case data
    cases_path = Path('data/processed/echr_cases_final_clean.json')
    print(f"\nLoading cases from {cases_path}...")
    with open(cases_path) as f:
        cases = json.load(f)
    print(f"Loaded {len(cases)} cases")

    # Run evaluations
    for evaluator_name in evaluators_to_run:
        evaluator_config = EVALUATORS[evaluator_name]

        # Get API key
        api_key = os.getenv(evaluator_config['api_key_env'])
        if not api_key:
            print(f"\n⚠ Skipping {evaluator_name}: {evaluator_config['api_key_env']} not set")
            continue

        for framing_name in FRAMINGS:
            # Evaluate
            df = evaluate_framing(
                cases=cases,
                evaluator_name=evaluator_name,
                evaluator_config=evaluator_config,
                api_key=api_key,
                framing_name=framing_name,
                num_samples=args.num_samples,
                temperature=args.temperature,
                max_workers=args.max_workers
            )

            # Save results
            output_filename = f"{evaluator_name}_{framing_name}_samples{args.num_samples}.csv"
            output_path = args.output_dir / output_filename
            df.to_csv(output_path, index=False)
            print(f"✓ Saved results to {output_path}")

    print("\n" + "="*80)
    print("✓ RQ2 Framing Comparison Experiment Complete!")
    print("="*80)


if __name__ == '__main__':
    main()
