"""
Generate v2 and v3 summary versions for all models.

This script generates additional summary versions using temperature sampling
to create multiple variations of summaries from the same model. This is useful
for testing the stability and variability of model-generated summaries.

Usage:
    python experiments/generate_summary_versions.py --model gpt-4o --version v2
    python experiments/generate_summary_versions.py --model claude-sonnet-4.5 --version v3
    python experiments/generate_summary_versions.py --all-models --all-versions
"""

import asyncio
import argparse
import json
import os
from pathlib import Path
import sys
from typing import List, Dict

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.evaluation import call_llm
from lib.models import create_client
from lib.prompts import STEP1_SUMMARIZATION_PROMPT

# Model configurations
MODEL_CONFIGS = {
    'gpt-4o': {
        'model_id': 'gpt-4o',
        'use_openrouter': False,
        'api_key_env': 'OPENAI_API_KEY',
        'base_dataset': 'data/datasets/echr_cases_100_gpt4o_v1.json'
    },
    'gpt-5.2': {
        'model_id': 'gpt-5.2',
        'use_openrouter': False,
        'api_key_env': 'OPENAI_API_KEY',
        'base_dataset': 'data/datasets/echr_cases_100_gpt5_2_v1.json'
    },
    'claude-sonnet-4.5': {
        'model_id': 'anthropic/claude-sonnet-4.5',
        'use_openrouter': True,
        'api_key_env': 'OPENROUTER_API_KEY',
        'base_dataset': 'data/datasets/echr_cases_100_claude_sonnet_4_5_v1.json'
    },
    'deepseek-v3.2': {
        'model_id': 'deepseek/deepseek-v3.2',
        'use_openrouter': True,
        'api_key_env': 'OPENROUTER_API_KEY',
        'base_dataset': 'data/datasets/echr_cases_100_deepseek_v3_2_v1.json'
    },
    'gemini-3-pro': {
        'model_id': 'google/gemini-3-pro',
        'use_openrouter': True,
        'api_key_env': 'OPENROUTER_API_KEY',
        'base_dataset': 'data/datasets/echr_cases_100_gemini_3_pro_v1.json'
    }
}


async def generate_summary_version(
    model_name: str,
    version: str,
    base_dataset_path: str,
    model_config: Dict,
    api_key: str,
    batch_size: int = 5
):
    """
    Generate a new summary version for a model.

    Args:
        model_name: Name of the model (e.g., 'gpt-4o')
        version: Version suffix (e.g., 'v2', 'v3')
        base_dataset_path: Path to the base dataset to load full case texts from
        model_config: Model configuration dict
        api_key: API key for the model
        batch_size: Number of cases to process in parallel
    """
    # Load base dataset to get full case texts
    print(f"\nLoading base dataset from {base_dataset_path}...")
    with open(base_dataset_path) as f:
        cases = json.load(f)

    print(f"Generating {version} summaries for {model_name}...")
    print(f"Total cases: {len(cases)}")

    # Create client
    client, final_model_id = create_client(
        model_config['model_id'],
        api_key,
        model_config['use_openrouter']
    )

    # Generate summaries in batches
    results = []
    for batch_start in range(0, len(cases), batch_size):
        batch_end = min(batch_start + batch_size, len(cases))
        batch = cases[batch_start:batch_end]

        print(f"  Processing batch {batch_start+1}-{batch_end}/{len(cases)}...")

        # Create tasks for this batch
        tasks = []
        for case in batch:
            prompt = STEP1_SUMMARIZATION_PROMPT.format(
                case_name=case['case_name'],
                full_text=case['full_case_text']
            )
            # Create async task to call API directly with temperature
            async def get_summary(client, model, prompt):
                response = await client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=1.0
                )
                return response.choices[0].message.content

            task = get_summary(client, final_model_id, prompt)
            tasks.append((case, task))

        # Execute batch in parallel
        batch_results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)

        # Process results
        for (case, _), summary in zip(tasks, batch_results):
            if isinstance(summary, Exception):
                print(f"    ⚠ Error for {case['case_name']}: {summary}")
                summary = ""  # Use empty string on error

            result = {
                'case_name': case['case_name'],
                'article': case['article'],
                'violation_label': case['violation_label'],
                'full_case_text': case['full_case_text'],
                'summary': summary
            }
            results.append(result)

    # Save results
    output_filename = f"echr_cases_100_{model_name.replace('.', '_').replace('-', '_')}_{version}.json"
    output_path = Path('data/datasets') / output_filename

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    success_count = sum(1 for r in results if r['summary'])
    print(f"\n✓ Saved {success_count}/{len(results)} summaries to {output_path}")
    return output_path


async def main():
    parser = argparse.ArgumentParser(description='Generate summary versions for models')
    parser.add_argument('--model', choices=list(MODEL_CONFIGS.keys()),
                        help='Model to generate summaries for')
    parser.add_argument('--version', choices=['v2', 'v3'],
                        help='Version to generate (v2 or v3)')
    parser.add_argument('--all-models', action='store_true',
                        help='Generate for all models')
    parser.add_argument('--all-versions', action='store_true',
                        help='Generate both v2 and v3')
    parser.add_argument('--batch-size', type=int, default=5,
                        help='Number of cases to process in parallel (default: 5)')

    args = parser.parse_args()

    # Validate arguments
    if not args.all_models and not args.model:
        parser.error('Must specify either --model or --all-models')
    if not args.all_versions and not args.version:
        parser.error('Must specify either --version or --all-versions')

    # Determine which models and versions to generate
    models_to_process = list(MODEL_CONFIGS.keys()) if args.all_models else [args.model]
    versions_to_generate = ['v2', 'v3'] if args.all_versions else [args.version]

    print("="*80)
    print("SUMMARY VERSION GENERATION")
    print("="*80)
    print(f"Models: {', '.join(models_to_process)}")
    print(f"Versions: {', '.join(versions_to_generate)}")
    print(f"Batch size: {args.batch_size}")
    print("="*80)

    # Generate summaries
    for model_name in models_to_process:
        model_config = MODEL_CONFIGS[model_name]

        # Get API key
        api_key = os.getenv(model_config['api_key_env'])
        if not api_key:
            print(f"⚠ Skipping {model_name}: {model_config['api_key_env']} not set")
            continue

        # Check if base dataset exists
        if not Path(model_config['base_dataset']).exists():
            print(f"⚠ Skipping {model_name}: Base dataset not found at {model_config['base_dataset']}")
            continue

        for version in versions_to_generate:
            await generate_summary_version(
                model_name=model_name,
                version=version,
                base_dataset_path=model_config['base_dataset'],
                model_config=model_config,
                api_key=api_key,
                batch_size=args.batch_size
            )

    print("\n" + "="*80)
    print("✓ Summary version generation complete!")
    print("="*80)


if __name__ == '__main__':
    asyncio.run(main())
