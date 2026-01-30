#!/usr/bin/env python3
"""
OPTIMIZED Summary Comparison Matrix Experiment

Key optimizations:
1. Producer-Consumer pattern with asyncio.Queue (no batch waves)
2. Persistent connection pooling via shared aiohttp connector
3. Higher concurrency (50-100 workers vs 10)
4. No artificial delays between batches
5. Continuous pipeline - always 100% full

Evaluates a 2D matrix of:
- Rows: Text types (full text + summaries from different models)
- Columns: Evaluator models

Each cell contains accuracy/abstention_rate for that combination.
"""

import asyncio
import argparse
import json
import os
from pathlib import Path
import sys
from typing import List, Dict, Tuple
import pandas as pd
from dataclasses import dataclass
import aiohttp

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.prompts import ARTICLE_TITLES, EVALUATION_SYSTEM_PROMPT, BASELINE_EVALUATION_TEMPLATE
from lib.models import build_messages, build_api_params, get_response_content
from lib.evaluation import extract_rating_from_response
from lib.metrics import calculate_accuracy

# Configuration for evaluator models
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


@dataclass
class EvaluationTask:
    """Single evaluation task."""
    case_name: str
    case_text: str
    article: str
    article_title: str
    violation_label: str
    sample_idx: int  # Which sample (0-9)


@dataclass
class EvaluationResult:
    """Result of a single evaluation."""
    case_name: str
    article: str
    violation_label: str
    sample_idx: int
    rating: int


class OptimizedEvaluator:
    """Optimized evaluator using producer-consumer pattern with connection pooling."""

    def __init__(
        self,
        model_id: str,
        api_key: str,
        use_openrouter: bool,
        temperature: float,
        num_workers: int = 50,
        max_retries: int = 3
    ):
        self.model_id = model_id
        self.api_key = api_key
        self.use_openrouter = use_openrouter
        self.temperature = temperature
        self.num_workers = num_workers
        self.max_retries = max_retries

        # Determine base URL and final model ID
        if use_openrouter:
            self.base_url = "https://openrouter.ai/api/v1"
            self.final_model_id = model_id
            self.default_headers = {
                "Authorization": f"Bearer {api_key}",
                "HTTP-Referer": "https://github.com/anthropics/llm-human-rights",
                "X-Title": "LLM Human Rights Research",
                "Content-Type": "application/json"
            }
        else:
            self.base_url = "https://api.openai.com/v1"
            self.final_model_id = model_id.split('/')[-1] if '/' in model_id else model_id
            self.default_headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

        # Statistics
        self.total_tasks = 0
        self.completed_tasks = 0
        self.failed_tasks = 0

    async def _call_api(
        self,
        session: aiohttp.ClientSession,
        messages: List[Dict],
        attempt: int = 0
    ) -> str:
        """Make API call with retry logic."""
        api_params = build_api_params(
            self.final_model_id,
            messages,
            self.temperature
        )

        try:
            async with session.post(
                f"{self.base_url}/chat/completions",
                json=api_params,
                timeout=aiohttp.ClientTimeout(total=60)
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    if data.get('choices') and data['choices'][0].get('message', {}).get('content'):
                        return data['choices'][0]['message']['content'].strip()
                    return ""

                elif response.status == 408:  # Timeout
                    raise asyncio.TimeoutError("Request timed out")

                elif response.status == 429:  # Rate limit
                    text = await response.text()
                    # Extract wait time if available, otherwise use exponential backoff
                    wait_time = 2 ** attempt
                    if attempt < self.max_retries - 1:
                        await asyncio.sleep(wait_time)
                        return await self._call_api(session, messages, attempt + 1)
                    else:
                        raise Exception(f"Rate limit exceeded after {self.max_retries} retries: {text}")

                else:
                    text = await response.text()
                    raise Exception(f"API error {response.status}: {text}")

        except asyncio.TimeoutError as e:
            if attempt < self.max_retries - 1:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
                return await self._call_api(session, messages, attempt + 1)
            else:
                raise

        except Exception as e:
            if attempt < self.max_retries - 1:
                wait_time = 2 ** attempt
                await asyncio.sleep(wait_time)
                return await self._call_api(session, messages, attempt + 1)
            else:
                raise

    async def _worker(
        self,
        worker_id: int,
        queue: asyncio.Queue,
        results: List[EvaluationResult],
        session: aiohttp.ClientSession
    ):
        """Worker that processes tasks from the queue."""
        while True:
            try:
                task = await queue.get()
                if task is None:  # Poison pill
                    break

                # Format prompt
                user_prompt = BASELINE_EVALUATION_TEMPLATE.format(
                    case_text=task.case_text,
                    article=task.article,
                    article_title=task.article_title
                )

                # Build messages
                messages = build_messages(
                    EVALUATION_SYSTEM_PROMPT,
                    user_prompt,
                    self.final_model_id
                )

                # Call API
                try:
                    content = await self._call_api(session, messages)
                    rating = extract_rating_from_response(content) if content else 3

                    results.append(EvaluationResult(
                        case_name=task.case_name,
                        article=task.article,
                        violation_label=task.violation_label,
                        sample_idx=task.sample_idx,
                        rating=rating
                    ))

                    self.completed_tasks += 1

                except Exception as e:
                    print(f"  Worker {worker_id}: Error on {task.case_name} sample {task.sample_idx}: {e}")
                    # Use neutral rating on failure
                    results.append(EvaluationResult(
                        case_name=task.case_name,
                        article=task.article,
                        violation_label=task.violation_label,
                        sample_idx=task.sample_idx,
                        rating=3
                    ))
                    self.failed_tasks += 1

                finally:
                    queue.task_done()

                    # Progress update every 100 tasks
                    if self.completed_tasks % 100 == 0:
                        progress_pct = (self.completed_tasks / self.total_tasks) * 100
                        print(f"  Progress: {self.completed_tasks}/{self.total_tasks} ({progress_pct:.1f}%)")

            except Exception as e:
                print(f"  Worker {worker_id}: Unexpected error: {e}")
                continue

    async def evaluate_all(
        self,
        cases: List[Dict],
        text_field: str,
        num_samples: int
    ) -> List[EvaluationResult]:
        """
        Evaluate all cases with all samples using producer-consumer pattern.

        Args:
            cases: List of case dictionaries
            text_field: Which text field to evaluate
            num_samples: Number of samples per case

        Returns:
            List of EvaluationResult objects
        """
        # Create task queue
        queue = asyncio.Queue()

        # Create results list (will be populated by workers)
        results = []

        # Populate queue with all tasks
        self.total_tasks = len(cases) * num_samples
        for case in cases:
            case_text = case[text_field]
            article = case['article']
            article_title = ARTICLE_TITLES.get(article, f"Article {article}")

            for sample_idx in range(num_samples):
                task = EvaluationTask(
                    case_name=case['case_name'],
                    case_text=case_text,
                    article=article,
                    article_title=article_title,
                    violation_label=case['violation_label'],
                    sample_idx=sample_idx
                )
                queue.put_nowait(task)

        print(f"  Queued {self.total_tasks} tasks for {len(cases)} cases × {num_samples} samples")

        # Create persistent connection with connection pooling
        connector = aiohttp.TCPConnector(
            limit=self.num_workers,  # Max concurrent connections
            limit_per_host=self.num_workers,
            ttl_dns_cache=300
        )

        async with aiohttp.ClientSession(
            connector=connector,
            headers=self.default_headers
        ) as session:
            # Create workers
            workers = [
                asyncio.create_task(
                    self._worker(worker_id, queue, results, session)
                )
                for worker_id in range(self.num_workers)
            ]

            # Wait for all tasks to complete
            await queue.join()

            # Send poison pills to stop workers
            for _ in range(self.num_workers):
                queue.put_nowait(None)

            # Wait for all workers to finish
            await asyncio.gather(*workers)

        print(f"  Completed: {self.completed_tasks}, Failed: {self.failed_tasks}")

        return results


async def evaluate_text_type_optimized(
    cases: List[Dict],
    text_field: str,
    evaluator_name: str,
    model_id: str,
    use_openrouter: bool,
    api_key: str,
    num_samples: int,
    temperature: float,
    output_file: Path,
    num_workers: int = 50
) -> Tuple[float, float]:
    """
    Evaluate a specific text type with a specific evaluator (OPTIMIZED).

    Args:
        cases: List of case dictionaries
        text_field: Which text field to evaluate
        evaluator_name: Name of evaluator model
        model_id: Model ID to use
        use_openrouter: Whether to use OpenRouter
        api_key: API key to use
        num_samples: Number of samples per case
        temperature: Sampling temperature
        output_file: CSV file to save detailed results
        num_workers: Number of concurrent workers (default: 50)

    Returns:
        Tuple of (accuracy, abstention_rate)
    """
    print(f"\n  Evaluating with {evaluator_name} (optimized, {num_workers} workers)...")

    # Create evaluator
    evaluator = OptimizedEvaluator(
        model_id=model_id,
        api_key=api_key,
        use_openrouter=use_openrouter,
        temperature=temperature,
        num_workers=num_workers
    )

    # Run evaluation
    eval_results = await evaluator.evaluate_all(cases, text_field, num_samples)

    # Group results by (case_name, article) pair
    case_results = {}
    for result in eval_results:
        key = (result.case_name, result.article)
        if key not in case_results:
            case_results[key] = {
                'case_name': result.case_name,
                'article': result.article,
                'violation_label': result.violation_label,
                'ratings': []
            }
        case_results[key]['ratings'].append(result.rating)

    # Calculate metrics for each case
    output_rows = []
    for key, data in case_results.items():
        ratings = data['ratings']
        avg_rating = sum(ratings) / len(ratings)
        is_violation = data['violation_label'] == 'violation'
        is_accurate = calculate_accuracy(avg_rating, is_violation)
        abstentions = sum(1 for r in ratings if r == 3)

        output_rows.append({
            'case_name': data['case_name'],
            'article': data['article'],
            'violation_label': data['violation_label'],
            'avg_rating': avg_rating,
            'is_accurate': is_accurate,
            'num_abstentions': abstentions,
            'num_samples': len(ratings),
            'sample_ratings': ratings
        })

    # Save detailed results
    df = pd.DataFrame(output_rows)
    df.to_csv(output_file, index=False)

    # Calculate overall metrics
    total_cases = len(output_rows)
    total_samples = sum(r['num_samples'] for r in output_rows)
    total_abstentions = sum(r['num_abstentions'] for r in output_rows)

    accuracy = sum(1 for r in output_rows if r['is_accurate']) / total_cases
    abstention_rate = total_abstentions / total_samples

    return accuracy, abstention_rate


async def run_matrix_experiment_optimized(
    num_samples: int = 10,
    temperature: float = 1.0,
    output_dir: Path = None,
    num_workers: int = 50
):
    """
    Run the full 2D matrix experiment with optimizations.

    Args:
        num_samples: Number of samples per case
        temperature: Sampling temperature
        output_dir: Output directory
        num_workers: Number of concurrent workers per experiment
    """
    if output_dir is None:
        output_dir = Path('data/experiments/summary_comparison_matrix')

    output_dir.mkdir(parents=True, exist_ok=True)
    detailed_dir = output_dir / 'detailed'
    detailed_dir.mkdir(exist_ok=True)

    # Get API keys from config.py first, fallback to env vars
    try:
        import config
        OPENAI_KEY = config.OPENAI_API_KEY or os.getenv('OPENAI_API_KEY')
        OPENROUTER_KEY = config.OPENROUTER_API_KEY or os.getenv('OPENROUTER_API_KEY')
    except ImportError:
        print("⚠ config.py not found, using environment variables")
        OPENAI_KEY = os.getenv('OPENAI_API_KEY')
        OPENROUTER_KEY = os.getenv('OPENROUTER_API_KEY')

    # Map API keys
    api_keys = {
        'OPENAI_API_KEY': OPENAI_KEY,
        'OPENROUTER_API_KEY': OPENROUTER_KEY
    }

    for key_name, key_value in api_keys.items():
        if not key_value:
            print(f"⚠ Warning: {key_name} not set")

    print("="*80)
    print("OPTIMIZED SUMMARY COMPARISON MATRIX EXPERIMENT")
    print("="*80)
    print(f"Workers per experiment: {num_workers}")
    print(f"Samples per case: {num_samples}")
    print(f"Temperature: {temperature}")
    print()

    # Store results in a list
    all_results = []

    # Import SUMMARY_DATASETS from the run script
    sys.path.insert(0, str(Path(__file__).parent.parent / 'scripts'))
    from run_new_summary_matrix import SUMMARY_DATASETS, EVALUATORS as EVAL_CONFIG

    # Run experiments
    total_experiments = len(EVAL_CONFIG) * len(SUMMARY_DATASETS)
    completed = 0

    for summary_name, summary_path in SUMMARY_DATASETS.items():
        print(f"\n{'='*80}")
        print(f"SUMMARY: {summary_name}")
        print(f"{'='*80}\n")

        # Check if summary file exists
        if not Path(summary_path).exists():
            print(f"⚠ Skipping {summary_name}: file not found at {summary_path}")
            continue

        # Load summary dataset
        with open(summary_path, 'r') as f:
            cases = json.load(f)

        print(f"Loaded {len(cases)} cases from {summary_name}")

        for evaluator_name, evaluator_config in EVAL_CONFIG.items():
            completed += 1
            print(f"\n  [{completed}/{total_experiments}] Evaluator: {evaluator_name}")

            model_id = evaluator_config['model_id']
            use_openrouter = evaluator_config['use_openrouter']
            api_key = api_keys[evaluator_config['api_key_env']]

            if not api_key:
                print(f"  ⚠ Skipping: API key not available")
                continue

            # Output file for detailed results
            output_file = detailed_dir / f"{evaluator_name}_summary_{summary_name}.csv"

            # Skip if already completed
            if output_file.exists():
                print(f"  ⏭ Skipping: Already completed (found {output_file.name})")
                # Load existing results to include in final summary
                try:
                    existing_df = pd.read_csv(output_file)
                    if len(existing_df) > 0:
                        accuracy = existing_df['is_accurate'].mean()
                        abstention_rate = existing_df['num_abstentions'].sum() / existing_df['num_samples'].sum()
                        all_results.append({
                            'summary': summary_name,
                            'evaluator': evaluator_name,
                            'accuracy': accuracy,
                            'abstention_rate': abstention_rate
                        })
                        print(f"    ℹ Loaded existing: Accuracy: {accuracy:.3f}, Abstention: {abstention_rate:.3f}")
                except Exception as e:
                    print(f"    ⚠ Could not load existing results: {e}")
                continue

            # Run evaluation
            try:
                accuracy, abstention_rate = await evaluate_text_type_optimized(
                    cases=cases,
                    text_field='step1_summary',
                    evaluator_name=evaluator_name,
                    model_id=model_id,
                    use_openrouter=use_openrouter,
                    api_key=api_key,
                    num_samples=num_samples,
                    temperature=temperature,
                    output_file=output_file,
                    num_workers=num_workers
                )

                print(f"    ✓ Accuracy: {accuracy:.3f}, Abstention: {abstention_rate:.3f}")

                # Store result
                all_results.append({
                    'summary': summary_name,
                    'evaluator': evaluator_name,
                    'accuracy': accuracy,
                    'abstention_rate': abstention_rate
                })

            except Exception as e:
                print(f"  ✗ Error: {e}")
                import traceback
                traceback.print_exc()
                continue

    # Save results matrix
    results_df = pd.DataFrame(all_results)
    results_file = output_dir / 'results.csv'
    results_df.to_csv(results_file, index=False)

    print("\n" + "="*80)
    print("EXPERIMENT COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {results_file}")
    print(f"Detailed results in: {detailed_dir}")

    # Print summary table
    print("\n" + "="*80)
    print("RESULTS SUMMARY")
    print("="*80 + "\n")

    # Pivot table for easier reading
    pivot_accuracy = results_df.pivot(index='summary', columns='evaluator', values='accuracy')
    pivot_abstention = results_df.pivot(index='summary', columns='evaluator', values='abstention_rate')

    print("ACCURACY:")
    print(pivot_accuracy.to_string())
    print("\nABSTENTION RATE:")
    print(pivot_abstention.to_string())

    # Save pivot tables
    pivot_accuracy.to_csv(output_dir / 'accuracy_matrix.csv')
    pivot_abstention.to_csv(output_dir / 'abstention_matrix.csv')

    print(f"\nMatrices saved to:")
    print(f"  {output_dir / 'accuracy_matrix.csv'}")
    print(f"  {output_dir / 'abstention_matrix.csv'}")


async def main():
    parser = argparse.ArgumentParser(
        description='Optimized summary comparison matrix experiment'
    )
    parser.add_argument('--num-samples', type=int, default=10,
                        help='Number of samples per case (default: 10)')
    parser.add_argument('--temperature', type=float, default=1.0,
                        help='Sampling temperature (default: 1.0)')
    parser.add_argument('--output-dir', type=str,
                        default='data_updated/experiments/summary_matrix_clean',
                        help='Output directory')
    parser.add_argument('--num-workers', type=int, default=50,
                        help='Number of concurrent workers (default: 50)')

    args = parser.parse_args()

    await run_matrix_experiment_optimized(
        num_samples=args.num_samples,
        temperature=args.temperature,
        output_dir=Path(args.output_dir),
        num_workers=args.num_workers
    )


if __name__ == '__main__':
    asyncio.run(main())
