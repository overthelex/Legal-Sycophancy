"""
Generate summary versions for ECHR cases using queue-based parallel execution.

This script generates summaries of ECHR case texts using various LLM models.
It uses temperature sampling to create multiple variations for robustness testing.

Usage:
    python experiments/generate_summary_versions.py --model gpt-5.2 --version v2
    python experiments/generate_summary_versions.py --model claude-sonnet-4.5 --version v3
    python experiments/generate_summary_versions.py --all-models --all-versions
"""

import argparse
import json
import os
from pathlib import Path
import sys
from typing import List, Dict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.queued_client import QueuedLLMClient
from lib.models import build_messages, prepare_request
from lib.prompts import STEP1_SUMMARIZATION_PROMPT

# Model configurations
MODEL_CONFIGS = {
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
    },
    'gemini-3-pro': {
        'model_id': 'google/gemini-3-pro',
        'use_openrouter': True,
        'api_key_env': 'OPENROUTER_API_KEY'
    }
}


def generate_summary_version(
    model_name: str,
    version: str,
    model_config: Dict,
    api_key: str,
    max_workers: int = 50
):
    """
    Generate a new summary version for a model using queue-based execution.

    Args:
        model_name: Name of the model (e.g., 'gpt-4o')
        version: Version suffix (e.g., 'v2', 'v3')
        model_config: Model configuration dict
        api_key: API key for the model
        max_workers: Number of concurrent workers
    """
    # Load base dataset
    base_dataset_path = 'data/processed/echr_cases_final_clean.json'
    print(f"\nLoading base dataset from {base_dataset_path}...")

    with open(base_dataset_path) as f:
        cases = json.load(f)

    print(f"Generating {version} summaries for {model_name}...")
    print(f"Total cases: {len(cases)}")
    print(f"Workers: {max_workers}")

    # Pre-generate all requests
    print("\nPreparing requests...")
    requests = []

    for case in cases:
        # Format summarization prompt
        prompt = STEP1_SUMMARIZATION_PROMPT.format(
            case_name=case['case_name'],
            full_text=case['full_case_text']
        )

        # Build messages (no system prompt for summarization)
        messages = [{"role": "user", "content": prompt}]

        # Prepare request
        custom_id = case.get('item_id', case['case_name'])

        request = prepare_request(
            custom_id=custom_id,
            model_id=model_config['model_id'],
            messages=messages,
            api_key=api_key,
            use_openrouter=model_config['use_openrouter'],
            temperature=1.0,  # Temperature sampling for variety
            max_tokens=2000   # Summaries need more tokens
        )

        requests.append(request)

    print(f"Prepared {len(requests)} requests")

    # Execute all requests via queue
    print("\nExecuting summarization requests...")
    client = QueuedLLMClient(max_workers=max_workers, timeout=180)

    def progress_callback(completed, total):
        if completed % 10 == 0 or completed == total:
            print(f"  Progress: {completed}/{total} ({100*completed/total:.1f}%)")

    results = client.execute_all(requests, progress_callback=progress_callback)

    # Process results and build output
    print("\nProcessing results...")
    output = []
    success_count = 0
    error_count = 0

    for case in cases:
        custom_id = case.get('item_id', case['case_name'])

        if custom_id in results:
            summary = results[custom_id]

            # Check if summary is error or empty
            if summary and not summary.startswith("ERROR:"):
                success_count += 1
            else:
                error_count += 1
                print(f"  ⚠ Error for {case['case_name']}: {summary}")
                summary = ""  # Use empty string on error
        else:
            error_count += 1
            print(f"  ⚠ Missing result for {case['case_name']}")
            summary = ""

        output.append({
            'item_id': case.get('item_id', ''),
            'case_name': case['case_name'],
            'article': case['article'],
            'violation_label': case['violation_label'],
            'full_case_text': case['full_case_text'],
            'summary': summary
        })

    # Save results
    output_filename = f"{model_name.replace('.', '_').replace('-', '_')}_{version}.json"
    output_path = Path('data/summaries') / output_filename

    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Saved {success_count}/{len(cases)} summaries to {output_path}")
    if error_count > 0:
        print(f"  ⚠ {error_count} errors occurred")


def main():
    parser = argparse.ArgumentParser(description='Generate summary versions for models')
    parser.add_argument('--model', choices=list(MODEL_CONFIGS.keys()),
                        help='Model to generate summaries for')
    parser.add_argument('--version', choices=['v1', 'v2', 'v3'],
                        help='Version to generate (v1, v2, or v3)')
    parser.add_argument('--all-models', action='store_true',
                        help='Generate for all models')
    parser.add_argument('--all-versions', action='store_true',
                        help='Generate v1, v2, and v3')
    parser.add_argument('--max-workers', type=int, default=50,
                        help='Number of concurrent workers (default: 50)')

    args = parser.parse_args()

    # Validate arguments
    if not args.all_models and not args.model:
        parser.error('Must specify either --model or --all-models')
    if not args.all_versions and not args.version:
        parser.error('Must specify either --version or --all-versions')

    # Determine which models and versions to generate
    models_to_process = list(MODEL_CONFIGS.keys()) if args.all_models else [args.model]
    versions_to_generate = ['v1', 'v2', 'v3'] if args.all_versions else [args.version]

    print("="*80)
    print("SUMMARY VERSION GENERATION (Queue-Based)")
    print("="*80)
    print(f"Models: {', '.join(models_to_process)}")
    print(f"Versions: {', '.join(versions_to_generate)}")
    print(f"Max workers: {args.max_workers}")
    print("="*80)

    # Generate summaries
    for model_name in models_to_process:
        model_config = MODEL_CONFIGS[model_name]

        # Get API key
        api_key = os.getenv(model_config['api_key_env'])
        if not api_key:
            print(f"⚠ Skipping {model_name}: {model_config['api_key_env']} not set")
            continue

        for version in versions_to_generate:
            generate_summary_version(
                model_name=model_name,
                version=version,
                model_config=model_config,
                api_key=api_key,
                max_workers=args.max_workers
            )

    print("\n" + "="*80)
    print("✓ Summary version generation complete!")
    print("="*80)


if __name__ == '__main__':
    main()
