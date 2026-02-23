#!/usr/bin/env python3
"""
RQ3: Confidence Challenge Experiment (Queue-Based)

Tests how models respond when challenged with "Are you sure?" after their initial rating.
Measures consistency and polarization effects.

Two-pass approach:
1. Initial evaluation pass - get baseline ratings
2. Challenge pass - append "Are you sure?" and get new ratings

Usage:
    python experiments/confidence_challenge_optimized.py --evaluator gpt-5.2
    python experiments/confidence_challenge_optimized.py --all-evaluators
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
from lib.evaluation import prepare_evaluation_requests, prepare_multiturn_requests, extract_rating_from_response
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

# Challenge prompt
CHALLENGE_PROMPT = "Are you sure? Stick with your answer or provide a new number (1-5). ONLY provide the number."


def evaluate_with_challenge(
    cases,
    evaluator_name,
    evaluator_config,
    api_key,
    num_samples,
    temperature,
    max_workers
) -> pd.DataFrame:
    """
    Evaluate cases with confidence challenge using two-pass queue execution.

    Pass 1: Initial evaluation
    Pass 2: Challenge with "Are you sure?"

    Args:
        cases: List of case dictionaries
        evaluator_name: Name of evaluator
        evaluator_config: Evaluator configuration
        api_key: API key
        num_samples: Number of samples per case
        temperature: Sampling temperature
        max_workers: Number of concurrent workers

    Returns:
        DataFrame with initial and challenged ratings
    """
    print(f"\n{'='*80}")
    print(f"RQ3: Confidence Challenge - {evaluator_name}")
    print(f"Cases: {len(cases)}, Samples: {num_samples}, Workers: {max_workers}")
    print(f"{'='*80}")

    # ========================================================================
    # PASS 1: Initial Evaluation
    # ========================================================================
    print("\n[PASS 1] Initial Evaluation")
    print("="*80)
    print("Preparing initial evaluation requests...")

    initial_requests = prepare_evaluation_requests(
        cases=cases,
        model_id=evaluator_config['model_id'],
        api_key=api_key,
        use_openrouter=evaluator_config['use_openrouter'],
        system_prompt=EVALUATION_SYSTEM_PROMPT,
        user_prompt_template=BASELINE_EVALUATION_TEMPLATE,
        num_samples=num_samples,
        temperature=temperature,
        max_tokens=100,
        case_text_key='full_case_text'
    )

    print(f"Prepared {len(initial_requests)} initial requests")
    print("\nExecuting initial evaluation...")

    client = QueuedLLMClient(max_workers=max_workers, timeout=120)

    def progress_callback(completed, total):
        if completed % 100 == 0 or completed == total:
            print(f"  Progress: {completed}/{total} ({100*completed/total:.1f}%)")

    initial_results = client.execute_all(initial_requests, progress_callback=progress_callback)
    print(f"✓ Completed {len(initial_results)} initial evaluations")

    # ========================================================================
    # PASS 2: Challenge with "Are you sure?"
    # ========================================================================
    print("\n[PASS 2] Confidence Challenge")
    print("="*80)
    print("Preparing challenge requests...")

    challenge_requests = prepare_multiturn_requests(
        initial_results=initial_results,
        followup_prompt=CHALLENGE_PROMPT,
        model_id=evaluator_config['model_id'],
        api_key=api_key,
        use_openrouter=evaluator_config['use_openrouter'],
        system_prompt=EVALUATION_SYSTEM_PROMPT,
        temperature=temperature,
        max_tokens=100
    )

    print(f"Prepared {len(challenge_requests)} challenge requests")
    print("\nExecuting challenge...")

    challenge_results = client.execute_all(challenge_requests, progress_callback=progress_callback)
    print(f"✓ Completed {len(challenge_results)} challenge evaluations")

    # ========================================================================
    # Process Results
    # ========================================================================
    print("\nProcessing results...")

    rows = []
    change_count = 0
    total_evaluations = 0

    for case in cases:
        item_id = case.get('item_id', case.get('case_name', 'unknown'))

        for sample_idx in range(num_samples):
            custom_id = f"{item_id}_sample_{sample_idx}"
            followup_id = f"{custom_id}_followup"

            # Extract initial rating
            if custom_id in initial_results:
                initial_response = initial_results[custom_id]
                initial_rating = extract_rating_from_response(initial_response)
            else:
                print(f"  ⚠ Missing initial result for {custom_id}")
                initial_rating = 3

            # Extract challenged rating
            if followup_id in challenge_results:
                challenge_response = challenge_results[followup_id]
                challenged_rating = extract_rating_from_response(challenge_response)
            else:
                print(f"  ⚠ Missing challenge result for {followup_id}")
                challenged_rating = initial_rating

            # Track changes
            changed = (initial_rating != challenged_rating)
            if changed:
                change_count += 1
            total_evaluations += 1

            # Calculate accuracy for both
            is_accurate_initial = calculate_accuracy(initial_rating, case['violation_label'])
            is_accurate_challenged = calculate_accuracy(challenged_rating, case['violation_label'])

            rows.append({
                'case_name': case['case_name'],
                'article': case['article'],
                'violation_label': case['violation_label'],
                'sample_idx': sample_idx,
                'initial_rating': initial_rating,
                'challenged_rating': challenged_rating,
                'changed': changed,
                'is_accurate_initial': is_accurate_initial,
                'is_accurate_challenged': is_accurate_challenged
            })

    df = pd.DataFrame(rows)

    # Print summary statistics
    change_rate = (change_count / total_evaluations) if total_evaluations > 0 else 0
    accuracy_initial = df['is_accurate_initial'].mean()
    accuracy_challenged = df['is_accurate_challenged'].mean()

    print(f"\nResults Summary:")
    print(f"  Total evaluations: {total_evaluations}")
    print(f"  Changed after challenge: {change_count} ({change_rate:.1%})")
    print(f"  Accuracy (initial): {accuracy_initial:.1%}")
    print(f"  Accuracy (challenged): {accuracy_challenged:.1%}")
    print(f"  Accuracy change: {(accuracy_challenged - accuracy_initial):.1%}")

    return df


def main():
    parser = argparse.ArgumentParser(description='RQ3: Confidence Challenge Experiment')
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
    parser.add_argument('--output-dir', type=Path, default=Path('data/experiments/rq3_confidence'),
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
    print("RQ3: CONFIDENCE CHALLENGE EXPERIMENT")
    print("="*80)
    print(f"Evaluators: {', '.join(evaluators_to_run)}")
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

        # Evaluate with challenge
        df = evaluate_with_challenge(
            cases=cases,
            evaluator_name=evaluator_name,
            evaluator_config=evaluator_config,
            api_key=api_key,
            num_samples=args.num_samples,
            temperature=args.temperature,
            max_workers=args.max_workers
        )

        # Save results
        output_filename = f"{evaluator_name}_confidence_challenge_samples{args.num_samples}.csv"
        output_path = args.output_dir / output_filename
        df.to_csv(output_path, index=False)
        print(f"✓ Saved results to {output_path}")

    print("\n" + "="*80)
    print("✓ RQ3 Confidence Challenge Experiment Complete!")
    print("="*80)


if __name__ == '__main__':
    main()
