#!/usr/bin/env python3
"""
Multi-Evaluator Framing Comparison Experiment

Runs framing comparison with multiple evaluator models.
Tests normative and factual framings against predictive baseline.
"""

import asyncio
import json
import logging
import pandas as pd
from pathlib import Path
import os
from scipy import stats
import numpy as np
import sys

# Add parent directory to path for lib imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.prompts import ARTICLE_TITLES, EVALUATION_SYSTEM_PROMPT, PROMPT_FRAMINGS
from lib.metrics import calculate_sample_metrics
from experiments.summary_comparison_matrix_optimized import OptimizedEvaluator

# Configure logger
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Suppress verbose logs
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("openai").setLevel(logging.WARNING)

# Evaluator models
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

# Framings to test (normative and factual only - predictive is baseline)
FRAMINGS_TO_TEST = ['normative', 'factual']


async def evaluate_with_framing(
    cases,
    evaluator_name,
    model_id,
    api_key,
    use_openrouter,
    framing_name,
    framing_template,
    num_samples,
    sample_temperature,
    num_workers
):
    """Evaluate cases with a specific framing using optimized queue method."""

    logger.info(f"  {evaluator_name} - {framing_name}: Evaluating {len(cases)} cases with {num_workers} workers")

    # Need to modify OptimizedEvaluator to support custom prompt templates
    # For now, we'll create a custom implementation using the same queue pattern

    from dataclasses import dataclass
    from lib.evaluation import extract_rating_from_response
    from lib.models import build_messages, build_api_params
    import aiohttp

    @dataclass
    class EvaluationTask:
        case_name: str
        case_text: str
        article: str
        article_title: str
        violation_label: str
        sample_idx: int

    @dataclass
    class EvaluationResult:
        case_name: str
        article: str
        violation_label: str
        sample_idx: int
        rating: int

    # Create evaluator with custom template support
    evaluator = OptimizedEvaluator(
        model_id=model_id,
        api_key=api_key,
        use_openrouter=use_openrouter,
        temperature=sample_temperature,
        num_workers=num_workers,
        max_retries=3
    )

    # Override the worker to use custom framing template
    async def custom_worker(worker_id, queue, results, session):
        """Worker that uses custom framing template."""
        while True:
            try:
                task = await queue.get()
                if task is None:  # Poison pill
                    break

                # Format prompt with custom framing
                user_prompt = framing_template.format(
                    case_text=task.case_text,
                    article=task.article,
                    article_title=task.article_title
                )

                # Build messages
                messages = build_messages(
                    EVALUATION_SYSTEM_PROMPT,
                    user_prompt,
                    evaluator.final_model_id
                )

                # Call API
                try:
                    content = await evaluator._call_api(session, messages)
                    rating = extract_rating_from_response(content) if content else 3

                    results.append(EvaluationResult(
                        case_name=task.case_name,
                        article=task.article,
                        violation_label=task.violation_label,
                        sample_idx=task.sample_idx,
                        rating=rating
                    ))

                    evaluator.completed_tasks += 1

                except Exception as e:
                    logger.error(f"  Worker {worker_id}: Error on {task.case_name} sample {task.sample_idx}: {e}")
                    results.append(EvaluationResult(
                        case_name=task.case_name,
                        article=task.article,
                        violation_label=task.violation_label,
                        sample_idx=task.sample_idx,
                        rating=3
                    ))
                    evaluator.failed_tasks += 1

                finally:
                    queue.task_done()

                    if evaluator.completed_tasks % 100 == 0:
                        progress_pct = (evaluator.completed_tasks / evaluator.total_tasks) * 100
                        logger.info(f"  Progress: {evaluator.completed_tasks}/{evaluator.total_tasks} ({progress_pct:.1f}%)")

            except Exception as e:
                logger.error(f"  Worker {worker_id}: Unexpected error: {e}")
                continue

    # Create task queue
    queue = asyncio.Queue()
    eval_results = []

    # Populate queue
    evaluator.total_tasks = len(cases) * num_samples
    for case in cases:
        case_text = case.get('summary') or case.get('step1_summary')
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

    # Create persistent connection
    connector = aiohttp.TCPConnector(
        limit=num_workers,
        limit_per_host=num_workers,
        ttl_dns_cache=300
    )

    async with aiohttp.ClientSession(
        connector=connector,
        headers=evaluator.default_headers
    ) as session:
        # Create workers
        workers = [
            asyncio.create_task(custom_worker(worker_id, queue, eval_results, session))
            for worker_id in range(num_workers)
        ]

        # Wait for completion
        await queue.join()

        # Stop workers
        for _ in range(num_workers):
            queue.put_nowait(None)

        await asyncio.gather(*workers)

    logger.info(f"  Completed: {evaluator.completed_tasks}, Failed: {evaluator.failed_tasks}")

    # Group by case-article and calculate metrics
    case_results = {}
    for result in eval_results:
        key = (result.case_name, result.article)
        if key not in case_results:
            case_results[key] = {
                'case_name': result.case_name,
                'article': result.article,
                'violation_label': result.violation_label,
                'evaluator': evaluator_name,
                'framing': framing_name,
                'ratings': []
            }
        case_results[key]['ratings'].append(result.rating)

    # Calculate metrics
    results = []
    for data in case_results.values():
        ratings = data['ratings']
        is_violation = data['violation_label'] == 'violation'
        metrics = calculate_sample_metrics(ratings, is_violation)

        results.append({
            'case_name': data['case_name'],
            'article': data['article'],
            'is_violation': is_violation,
            'evaluator': data['evaluator'],
            'framing': data['framing'],
            **metrics
        })

    return results


async def run_experiment(
    dataset_path: str,
    baseline_dir: str,
    num_samples: int,
    sample_temperature: float,
    batch_size: int,
    output_dir: str
):
    """Run multi-evaluator framing comparison experiment."""

    # Load dataset
    with open(dataset_path, 'r') as f:
        cases = json.load(f)

    logger.info("="*80)
    logger.info("MULTI-EVALUATOR FRAMING COMPARISON EXPERIMENT")
    logger.info("="*80)
    logger.info(f"Dataset: {dataset_path}")
    logger.info(f"Cases: {len(cases)}")
    logger.info(f"Evaluators: {list(EVALUATORS.keys())}")
    logger.info(f"Framings to test: {FRAMINGS_TO_TEST}")
    logger.info(f"Baseline (predictive): Loading from {baseline_dir}")
    logger.info(f"Samples per case: {num_samples}")
    logger.info(f"Sample temperature: {sample_temperature}")
    logger.info("")

    # Get API keys
    try:
        import config
        OPENAI_KEY = config.OPENAI_API_KEY or os.getenv('OPENAI_API_KEY')
        OPENROUTER_KEY = config.OPENROUTER_API_KEY or os.getenv('OPENROUTER_API_KEY')
    except ImportError:
        logger.info("⚠ config.py not found, using environment variables")
        OPENAI_KEY = os.getenv('OPENAI_API_KEY')
        OPENROUTER_KEY = os.getenv('OPENROUTER_API_KEY')

    api_keys = {
        'OPENAI_API_KEY': OPENAI_KEY,
        'OPENROUTER_API_KEY': OPENROUTER_KEY
    }

    # Store all results
    all_results = []

    # Load baseline (predictive) results for each evaluator
    baseline_results = {}
    baseline_path = Path(baseline_dir)

    for evaluator_name in EVALUATORS.keys():
        baseline_file = baseline_path / f"{evaluator_name}_summary_gpt5_2_v3.csv"

        if baseline_file.exists():
            logger.info(f"✓ Loading baseline for {evaluator_name} from {baseline_file.name}")
            df = pd.read_csv(baseline_file)

            # Convert to results format
            baseline_results[evaluator_name] = []
            for _, row in df.iterrows():
                baseline_results[evaluator_name].append({
                    'case_name': row['case_name'],
                    'article': row['article'],
                    'is_violation': row['violation_label'] == 'violation',
                    'evaluator': evaluator_name,
                    'framing': 'predictive',
                    'avg_rating': row['avg_rating'],
                    'is_accurate': row['is_accurate'],
                    'num_abstentions': row['num_abstentions'],
                    'num_samples': row['num_samples'],
                    'sample_ratings': eval(row['sample_ratings']) if isinstance(row['sample_ratings'], str) else row['sample_ratings'],
                    'distance_score': abs(row['avg_rating'] - (1 if row['violation_label'] == 'violation' else 5))
                })

            all_results.extend(baseline_results[evaluator_name])
            logger.info(f"  Loaded {len(baseline_results[evaluator_name])} baseline cases")
        else:
            logger.warning(f"⚠ Baseline file not found: {baseline_file}")
            baseline_results[evaluator_name] = None

    # Run new framings for each evaluator
    for evaluator_name, evaluator_config in EVALUATORS.items():
        logger.info("\n" + "="*80)
        logger.info(f"EVALUATOR: {evaluator_name}")
        logger.info("="*80)

        model_id = evaluator_config['model_id']
        use_openrouter = evaluator_config['use_openrouter']
        api_key = api_keys[evaluator_config['api_key_env']]

        if not api_key:
            logger.warning(f"⚠ Skipping {evaluator_name}: API key not available")
            continue

        if baseline_results.get(evaluator_name) is None:
            logger.warning(f"⚠ Skipping {evaluator_name}: No baseline results")
            continue

        # Evaluate with each new framing
        # Use fewer workers for OpenAI models to avoid rate limits
        num_workers = 10 if not use_openrouter else 50

        for framing_name in FRAMINGS_TO_TEST:
            logger.info(f"\n  Framing: {framing_name} ({PROMPT_FRAMINGS[framing_name]['description']})")

            framing_results = await evaluate_with_framing(
                cases=cases,
                evaluator_name=evaluator_name,
                model_id=model_id,
                api_key=api_key,
                use_openrouter=use_openrouter,
                framing_name=framing_name,
                framing_template=PROMPT_FRAMINGS[framing_name]['template'],
                num_samples=num_samples,
                sample_temperature=sample_temperature,
                num_workers=num_workers
            )

            all_results.extend(framing_results)
            logger.info(f"  ✓ Completed {len(framing_results)} cases")

    # Statistical analysis
    logger.info("\n" + "="*80)
    logger.info("STATISTICAL COMPARISON")
    logger.info("="*80)

    # Create dataframe
    df_all = pd.DataFrame(all_results)

    # Compare each evaluator's framings
    comparison_results = []

    for evaluator_name in EVALUATORS.keys():
        if baseline_results.get(evaluator_name) is None:
            continue

        logger.info(f"\n{evaluator_name.upper()}:")
        logger.info("-" * 40)

        df_eval = df_all[df_all['evaluator'] == evaluator_name]

        # Get baseline scores
        baseline_df = df_eval[df_eval['framing'] == 'predictive']

        if len(baseline_df) == 0:
            logger.warning(f"  ⚠ No baseline data for {evaluator_name}")
            continue

        logger.info(f"  Predictive (baseline):")
        logger.info(f"    Cases: {len(baseline_df)}")
        logger.info(f"    Mean score: {baseline_df['distance_score'].mean():.4f}")
        logger.info(f"    Accuracy:   {baseline_df['is_accurate'].mean():.4f}")

        # Compare each new framing to baseline
        for framing_name in FRAMINGS_TO_TEST:
            framing_df = df_eval[df_eval['framing'] == framing_name]

            if len(framing_df) == 0:
                continue

            logger.info(f"  {framing_name}: {len(framing_df)} cases")

            # Match cases between baseline and framing (only use cases that appear in both)
            baseline_df_sorted = baseline_df.set_index(['case_name', 'article']).sort_index()
            framing_df_sorted = framing_df.set_index(['case_name', 'article']).sort_index()

            # Find common cases
            common_index = baseline_df_sorted.index.intersection(framing_df_sorted.index)

            if len(common_index) == 0:
                logger.warning(f"    ⚠ No common cases between baseline and {framing_name}")
                continue

            # Get matched scores
            baseline_scores = baseline_df_sorted.loc[common_index, 'distance_score'].values
            framing_scores = framing_df_sorted.loc[common_index, 'distance_score'].values

            baseline_accuracy = baseline_df_sorted.loc[common_index, 'is_accurate'].mean()
            framing_accuracy = framing_df_sorted.loc[common_index, 'is_accurate'].mean()

            logger.info(f"    Comparing {len(common_index)} matched cases")

            # Paired t-test
            t_stat, p_value = stats.ttest_rel(baseline_scores, framing_scores)

            score_diff = framing_scores.mean() - baseline_scores.mean()
            acc_diff = framing_accuracy - baseline_accuracy
            is_significant = p_value < 0.05
            sig_marker = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"

            logger.info(f"\n  {framing_name.capitalize()}:")
            logger.info(f"    Mean score: {framing_scores.mean():.4f}")
            logger.info(f"    Accuracy:   {framing_accuracy:.4f}")
            logger.info(f"    vs Predictive:")
            logger.info(f"      Score diff:  {score_diff:+.4f}")
            logger.info(f"      Acc diff:    {acc_diff:+.4f}")
            logger.info(f"      t-stat:      {t_stat:.4f}")
            logger.info(f"      p-value:     {p_value:.4f} {sig_marker}")

            if is_significant:
                direction = "worse" if score_diff > 0 else "better"
                logger.info(f"      ✓ SIGNIFICANT: {framing_name} is {direction} than predictive")
            else:
                logger.info(f"      ✗ NOT SIGNIFICANT")

            comparison_results.append({
                'evaluator': evaluator_name,
                'framing': framing_name,
                'baseline_mean_score': float(baseline_scores.mean()),
                'baseline_accuracy': float(baseline_accuracy),
                'framing_mean_score': float(framing_scores.mean()),
                'framing_accuracy': float(framing_accuracy),
                'score_difference': float(score_diff),
                'accuracy_difference': float(acc_diff),
                't_statistic': float(t_stat),
                'p_value': float(p_value),
                'is_significant': bool(is_significant),
                'significance_marker': sig_marker
            })

    # Save results
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save detailed results
    detailed_file = output_path / 'detailed_results.csv'
    df_all.to_csv(detailed_file, index=False)

    # Save comparison statistics
    comparison_file = output_path / 'comparison_statistics.csv'
    pd.DataFrame(comparison_results).to_csv(comparison_file, index=False)

    # Save summary JSON
    summary = {
        'dataset': dataset_path,
        'num_cases': len(cases),
        'num_samples': num_samples,
        'evaluators': list(EVALUATORS.keys()),
        'framings_tested': FRAMINGS_TO_TEST,
        'baseline_framing': 'predictive',
        'comparisons': comparison_results
    }

    summary_file = output_path / 'summary.json'
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info("\n" + "="*80)
    logger.info(f"✓ Results saved to: {output_path}")
    logger.info(f"  - {detailed_file.name}")
    logger.info(f"  - {comparison_file.name}")
    logger.info(f"  - {summary_file.name}")
    logger.info("="*80)


async def main():
    """Main function."""

    await run_experiment(
        dataset_path='data_updated/summaries_clean/echr_cases_new_gpt5_2_v3.json',
        baseline_dir='data_updated/experiments/summary_matrix_clean/detailed',
        num_samples=10,
        sample_temperature=1.0,
        batch_size=10,
        output_dir='data_updated/experiments/framing_comparison_multi'
    )


if __name__ == "__main__":
    asyncio.run(main())
