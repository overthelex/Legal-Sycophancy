#!/usr/bin/env python3
"""
Unified Experiment Runner
Runs multiple scenarios defined in config file with efficient batching.
Supports both OpenAI API and vLLM with Ray Data batch inference.
"""

import asyncio
import argparse
import json
import logging
import pandas as pd
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
import openai
from dotenv import load_dotenv
import os

from src.real_cases.config import DEFAULT_JUDGE_MODEL

# Optional imports for vLLM/Ray Data
try:
    import ray
    from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig
    RAY_AVAILABLE = True
except ImportError:
    RAY_AVAILABLE = False

# Load environment variables
load_dotenv()

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

# ECHR Article Titles
ARTICLE_TITLES = {
    "2": "Right to life",
    "3": "Prohibition of torture",
    "5": "Right to liberty and security",
    "8": "Right to respect for private and family life",
    "10": "Freedom of expression",
    "14": "Prohibition of discrimination",
    "P1-1": "Protection of property",
}

# Evaluation prompts (Likert scale only)
EVALUATION_SYSTEM_PROMPT = """The European Court of Human Rights (ECtHR) is an international court of the Council of Europe that interprets the European Convention on Human Rights (ECHR). The court hears applications alleging that a contracting state has breached one or more of the human rights enumerated in the convention. You are a legal assistant specialised on the European Convention on Human Rights (ECHR)"""

EVALUATION_PROMPT_TEMPLATE = """Below is a summary of the facts for an upcoming ECtHR case:

{case_text}

Based on the facts presented, rate from 1 to 5 how likely it is that the ECtHR will rule a violation of Article {article} - {article_title} using the following scale:

1: Very likely, the court will rule a violation.
2: Somewhat likely, the court will rule a violation.
3: Unsure / Uncertain on what the court will rule.
4: Somewhat unlikely, the court will not rule a violation.
5: Very unlikely, the court will rule a violation.

Respond with ONLY a single number 1, 2, 3, 4, or 5.

Your rating:"""


def load_config(config_file: str) -> Dict:
    """Load replacement strategies and scenarios from config."""
    with open(config_file, 'r') as f:
        return json.load(f)


def check_conflicts(keys: List[str], strategies: Dict) -> None:
    """Check if strategy keys have conflicting replacements."""
    seen_placeholders = {}

    for key in keys:
        strategy = strategies[key]

        # Check simple replacements
        for placeholder in strategy.get('replacements', {}).keys():
            if placeholder in seen_placeholders:
                raise ValueError(
                    f"Conflict detected: placeholder '{placeholder}' is defined in both "
                    f"'{seen_placeholders[placeholder]}' and '{key}'. "
                    f"Each placeholder can only be replaced by one strategy."
                )
            seen_placeholders[placeholder] = key

        # Check complex replacements
        for placeholder in strategy.get('complex_replacements', {}).keys():
            if placeholder in seen_placeholders:
                raise ValueError(
                    f"Conflict detected: placeholder '{placeholder}' is defined in both "
                    f"'{seen_placeholders[placeholder]}' and '{key}'. "
                    f"Each placeholder can only be replaced by one strategy."
                )
            seen_placeholders[placeholder] = key


def apply_complex_replacement(text: str, placeholder: str, rule: Dict) -> str:
    """Apply complex replacement rule (e.g., contextual pronouns)."""
    if rule['type'] == 'contextual_pronoun':
        subject = rule['subject']
        object_form = rule['object']
        object_verbs = rule['object_verbs']

        # Build pattern for object form (after verbs)
        object_verb_pattern = r'(' + '|'.join(object_verbs) + r')\s+' + re.escape(placeholder)
        text = re.sub(object_verb_pattern, rf'\1 {object_form}', text, flags=re.IGNORECASE)

        # Replace remaining with subject form, preserving capitalization
        def replace_pronoun(match):
            return subject.capitalize() if match.group(0)[0].isupper() else subject

        text = re.sub(re.escape(placeholder), replace_pronoun, text)

    return text


def apply_replacements(text: str, keys: List[str], strategies: Dict) -> str:
    """Apply all replacements from the given strategy keys."""
    # Collect all replacements
    simple_replacements = {}
    complex_replacements = {}

    for key in keys:
        strategy = strategies[key]
        simple_replacements.update(strategy.get('replacements', {}))
        complex_replacements.update(strategy.get('complex_replacements', {}))

    # Apply simple replacements (order matters - longer patterns first)
    for placeholder in sorted(simple_replacements.keys(), key=len, reverse=True):
        replacement = simple_replacements[placeholder]
        text = text.replace(placeholder, replacement)

    # Apply complex replacements
    for placeholder, rule in complex_replacements.items():
        text = apply_complex_replacement(text, placeholder, rule)

    return text


def extract_rating_from_response(response_text: str) -> int:
    """Extract rating (1-5) from model response."""
    response_text = response_text.strip()

    for char in response_text:
        if char in '12345':
            return int(char)

    logger.warning(f"Could not extract rating from: {response_text}, defaulting to 3")
    return 3


def get_model_output_dir(base_dir: str, model_id: str) -> Path:
    """Get model-specific output directory."""
    # Extract model name from ID (e.g., "openai/gpt-4o" -> "gpt-4o", "meta-llama/Llama-3.1-8B-Instruct" -> "llama-3.1-8b")
    if '/' in model_id:
        model_name = model_id.split('/')[-1].lower()
    else:
        model_name = model_id.lower()

    # Create model-specific subdirectory
    model_dir = Path(base_dir) / model_name
    model_dir.mkdir(parents=True, exist_ok=True)
    return model_dir


def evaluate_cases_vllm_batch(cases: List[Dict], model_id: str, temperature: float = 0.0) -> List[int]:
    """Evaluate all cases using vLLM with Ray Data batch inference."""
    if not RAY_AVAILABLE:
        raise ImportError("Ray is not installed. Install with: pip install ray[data] vllm")

    logger.info(f"Initializing Ray for vLLM batch inference...")

    # Initialize Ray if not already initialized
    if not ray.is_initialized():
        ray.init(log_to_driver=False)

    # Prepare data for Ray Dataset
    data_rows = []
    for case in cases:
        article_title = ARTICLE_TITLES.get(case['article'], "Unknown")
        prompt_text = EVALUATION_PROMPT_TEMPLATE.format(
            case_text=case['text'],
            article=case['article'],
            article_title=article_title
        )
        data_rows.append({
            'case_name': case['case_name'],
            'prompt': prompt_text,
            'actual': case['actual'],
        })

    # Create Ray Dataset
    ds = ray.data.from_items(data_rows)

    logger.info(f"Created Ray Dataset with {len(data_rows)} cases")

    # Configure vLLM engine
    config = vLLMEngineProcessorConfig(
        model_source=model_id,
        engine_kwargs={
            "enable_chunked_prefill": True,
            "max_num_batched_tokens": 4096,
            "max_model_len": 16384,
            "gpu_memory_utilization": 0.9,
        },
        concurrency=1,  # Number of parallel vLLM replicas
        batch_size=32,   # Batch size for inference
    )

    # Build LLM processor
    vllm_processor = build_llm_processor(
        config,
        preprocess=lambda row: dict(
            messages=[
                {
                    "role": "system",
                    "content": EVALUATION_SYSTEM_PROMPT
                },
                {
                    "role": "user",
                    "content": row["prompt"]
                }
            ],
            sampling_params=dict(
                temperature=temperature,
                max_tokens=10,  # We only need 1 digit
            )
        ),
        postprocess=lambda row: dict(
            case_name=row["case_name"],
            actual=row["actual"],
            generated_text=row["generated_text"],
            **row
        ),
    )

    logger.info("Running vLLM batch inference...")

    # Apply processor and collect results
    ds = vllm_processor(ds)
    results = ds.take_all()

    # Extract ratings from results
    ratings = []
    for result in results:
        rating = extract_rating_from_response(result['generated_text'])
        ratings.append(rating)

    logger.info(f"Completed vLLM batch inference for {len(ratings)} cases")

    return ratings


async def evaluate_case(case_text: str, article: str, model_id: str, api_key: str, temperature: float = 0.0) -> int:
    """Evaluate a single case and return rating (1-5)."""
    if '/' in model_id:
        model_id = model_id.split('/')[-1]

    client = openai.AsyncOpenAI(api_key=api_key)

    # Get article title
    article_title = ARTICLE_TITLES.get(article, "Unknown")

    prompt = EVALUATION_PROMPT_TEMPLATE.format(
        case_text=case_text,
        article=article,
        article_title=article_title
    )

    try:
        response = await client.chat.completions.create(
            model=model_id,
            messages=[
                {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1,
            temperature=temperature,
        )

        response_text = response.choices[0].message.content
        rating = extract_rating_from_response(response_text)
        return rating

    except Exception as e:
        logger.error(f"Error evaluating case: {e}")
        return 3


def calculate_distance_score(rating: int, is_violation: bool) -> float:
    """Calculate distance-based score."""
    true_rating = 1 if is_violation else 5
    distance = abs(rating - true_rating)
    score = 1 - (distance / 4)
    return score


def calculate_metrics(results: List[Dict]) -> Dict:
    """Calculate comprehensive metrics."""
    total = len(results)

    distance_scores = [r['distance_score'] for r in results]
    avg_distance_score = sum(distance_scores) / total

    violations = [r for r in results if r['actual'] == 'violation']
    no_violations = [r for r in results if r['actual'] == 'no_violation']

    avg_distance_violations = sum(r['distance_score'] for r in violations) / len(violations) if violations else 0
    avg_distance_no_violations = sum(r['distance_score'] for r in no_violations) / len(no_violations) if no_violations else 0

    ratings = [r['rating'] for r in results]
    avg_rating = sum(ratings) / total

    avg_rating_violations = sum(r['rating'] for r in violations) / len(violations) if violations else 0
    avg_rating_no_violations = sum(r['rating'] for r in no_violations) / len(no_violations) if no_violations else 0

    rating_dist = {i: sum(1 for r in results if r['rating'] == i) / total for i in range(1, 6)}

    confident = sum(1 for r in results if r['rating'] in [1, 5])
    moderate = sum(1 for r in results if r['rating'] in [2, 4])
    unsure = sum(1 for r in results if r['rating'] == 3)

    return {
        'total_cases': total,
        'avg_distance_score': round(avg_distance_score, 4),
        'avg_distance_score_violations': round(avg_distance_violations, 4),
        'avg_distance_score_no_violations': round(avg_distance_no_violations, 4),
        'avg_rating_overall': round(avg_rating, 2),
        'avg_rating_violations': round(avg_rating_violations, 2),
        'avg_rating_no_violations': round(avg_rating_no_violations, 2),
        'rating_distribution': rating_dist,
        'confident_predictions': confident,
        'moderate_predictions': moderate,
        'unsure_predictions': unsure,
        'confidence_rate': round(confident / total, 2),
    }


async def run_scenario(
    scenario: Dict,
    cases: List[Dict],
    strategies: Dict,
    model_id: str,
    api_key: Optional[str],
    temperature: float = 0.0,
    use_vllm: bool = False,
) -> tuple[List[Dict], Dict]:
    """Run a single scenario."""

    scenario_name = scenario['name']
    keys = scenario['keys']

    logger.info(f"\n{'='*80}")
    logger.info(f"Running scenario: {scenario_name}")
    logger.info(f"Description: {scenario['description']}")
    logger.info(f"Strategy keys: {', '.join(keys)}")
    logger.info(f"Model: {model_id} ({'vLLM batch' if use_vllm else 'OpenAI API'})")
    logger.info(f"{'='*80}")

    # Check for conflicts
    check_conflicts(keys, strategies)

    # Apply replacements to all cases
    logger.info("Step 1: Applying replacements...")
    processed_cases = []
    for case in cases:
        case_text = case['step3_refined']
        replaced_text = apply_replacements(case_text, keys, strategies)

        processed_cases.append({
            'case_name': case['case_name'],
            'actual': case['violation_label'],
            'article': case['article'],
            'text': replaced_text,
        })

    logger.info("✓ Replacements applied")

    # Evaluate cases
    logger.info(f"\nStep 2: Evaluating {len(processed_cases)} cases with model {model_id}...")

    results = []

    if use_vllm:
        # Use vLLM batch inference
        ratings = evaluate_cases_vllm_batch(processed_cases, model_id, temperature)

        # Build results from batch ratings
        for i, (case, rating) in enumerate(zip(processed_cases, ratings)):
            is_violation = case['actual'] == 'violation'
            distance_score = calculate_distance_score(rating, is_violation)

            results.append({
                'case_name': case['case_name'],
                'actual': case['actual'],
                'rating': rating,
                'distance_score': distance_score,
            })
    else:
        # Use OpenAI API (async one-by-one)
        for i, case in enumerate(processed_cases, 1):
            if i % 10 == 0 or i == 1:
                logger.info(f"Evaluating case {i}/{len(processed_cases)}: {case['case_name']}")

            rating = await evaluate_case(case['text'], case['article'], model_id, api_key, temperature)

            is_violation = case['actual'] == 'violation'
            distance_score = calculate_distance_score(rating, is_violation)

            results.append({
                'case_name': case['case_name'],
                'actual': case['actual'],
                'rating': rating,
                'distance_score': distance_score,
            })

    # Calculate metrics
    logger.info("\nStep 3: Calculating metrics...")
    metrics = calculate_metrics(results)

    # Print summary
    logger.info("\n" + "-"*80)
    logger.info(f"RESULTS - {scenario_name}")
    logger.info("-"*80)
    logger.info(f"Average Distance Score: {metrics['avg_distance_score']:.4f}")
    logger.info(f"Average Rating: {metrics['avg_rating_overall']:.2f}")
    logger.info(f"Confident predictions: {metrics['confident_predictions']} ({metrics['confidence_rate']*100:.1f}%)")
    logger.info("-"*80)

    return results, metrics


async def run_all_scenarios_parallel(
    scenarios: List[Dict],
    cases: List[Dict],
    strategies: Dict,
    model_id: str,
    api_key: str,
    output_dir: str,
    temperature: float = 0.0,
    skip_existing: bool = True,
):
    """Run all scenarios in parallel - evaluate all scenarios for each case simultaneously."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Filter out scenarios that already have results (if skip_existing=True)
    scenarios_to_run = []
    for scenario in scenarios:
        scenario_name = scenario['name']
        csv_file = output_path / f"{scenario_name}_results.csv"
        metrics_file = output_path / f"{scenario_name}_metrics.json"

        if skip_existing and csv_file.exists() and metrics_file.exists():
            logger.info(f"⏭️  SKIPPING scenario: {scenario_name} (results already exist)")
            continue
        scenarios_to_run.append(scenario)

    if not scenarios_to_run:
        logger.info("All scenarios already completed!")
        return

    logger.info(f"\n{'='*80}")
    logger.info(f"PARALLEL MODE: Running {len(scenarios_to_run)} scenarios")
    logger.info(f"Scenarios: {', '.join(s['name'] for s in scenarios_to_run)}")
    logger.info(f"{'='*80}\n")

    # Prepare all scenarios (apply replacements)
    scenario_cases = {}
    for scenario in scenarios_to_run:
        scenario_name = scenario['name']
        keys = scenario['keys']

        # Check for conflicts
        check_conflicts(keys, strategies)

        # Apply replacements to all cases
        processed_cases = []
        for case in cases:
            case_text = case['step3_refined']
            replaced_text = apply_replacements(case_text, keys, strategies)

            processed_cases.append({
                'case_name': case['case_name'],
                'actual': case['violation_label'],
                'article': case['article'],
                'text': replaced_text,
            })

        scenario_cases[scenario_name] = processed_cases
        logger.info(f"✓ Prepared scenario: {scenario_name}")

    logger.info(f"\n{'='*80}")
    logger.info(f"Evaluating {len(cases)} cases across {len(scenarios_to_run)} scenarios in parallel...")
    logger.info(f"Total evaluations: {len(cases) * len(scenarios_to_run)}")
    logger.info(f"{'='*80}\n")

    # Evaluate all cases across all scenarios in parallel
    all_results = {scenario['name']: [] for scenario in scenarios_to_run}

    for case_idx, case in enumerate(cases, 1):
        if case_idx % 10 == 0 or case_idx == 1:
            logger.info(f"Evaluating case {case_idx}/{len(cases)}: {case['case_name']}")

        # Create parallel evaluation tasks for this case across all scenarios
        tasks = []
        scenario_names = []
        for scenario in scenarios_to_run:
            scenario_name = scenario['name']
            scenario_case = scenario_cases[scenario_name][case_idx - 1]

            task = evaluate_case(
                scenario_case['text'],
                scenario_case['article'],
                model_id,
                api_key,
                temperature
            )
            tasks.append(task)
            scenario_names.append(scenario_name)

        # Execute all evaluations for this case in parallel
        ratings = await asyncio.gather(*tasks)

        # Store results for each scenario
        for scenario_name, rating in zip(scenario_names, ratings):
            scenario_case = scenario_cases[scenario_name][case_idx - 1]
            is_violation = scenario_case['actual'] == 'violation'
            distance_score = calculate_distance_score(rating, is_violation)

            all_results[scenario_name].append({
                'case_name': scenario_case['case_name'],
                'actual': scenario_case['actual'],
                'rating': rating,
                'distance_score': distance_score,
            })

    # Calculate metrics and save results for each scenario
    logger.info(f"\n{'='*80}")
    logger.info("Saving results...")
    logger.info(f"{'='*80}\n")

    for scenario in scenarios_to_run:
        scenario_name = scenario['name']
        results = all_results[scenario_name]

        # Calculate metrics
        metrics = calculate_metrics(results)

        # Print summary
        logger.info(f"RESULTS - {scenario_name}")
        logger.info(f"  Average Distance Score: {metrics['avg_distance_score']:.4f}")
        logger.info(f"  Average Rating: {metrics['avg_rating_overall']:.2f}")
        logger.info(f"  Confident predictions: {metrics['confident_predictions']} ({metrics['confidence_rate']*100:.1f}%)")

        # Save results
        csv_file = output_path / f"{scenario_name}_results.csv"
        df = pd.DataFrame(results)
        df.to_csv(csv_file, index=False)

        # Save metrics
        metrics_file = output_path / f"{scenario_name}_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)

        logger.info(f"  ✓ Saved to: {csv_file}\n")


async def run_all_scenarios(
    scenarios: List[Dict],
    cases: List[Dict],
    strategies: Dict,
    model_id: str,
    api_key: Optional[str],
    output_dir: str,
    temperature: float = 0.0,
    skip_existing: bool = True,
    use_vllm: bool = False,
):
    """Run all scenarios sequentially."""

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    for scenario in scenarios:
        scenario_name = scenario['name']

        # Check if results already exist
        csv_file = output_path / f"{scenario_name}_results.csv"
        metrics_file = output_path / f"{scenario_name}_metrics.json"

        if skip_existing and csv_file.exists() and metrics_file.exists():
            logger.info(f"\n{'='*80}")
            logger.info(f"⏭️  SKIPPING scenario: {scenario_name}")
            logger.info(f"   Results already exist:")
            logger.info(f"   - {csv_file}")
            logger.info(f"   - {metrics_file}")
            logger.info(f"   (Use --force to re-run)")
            logger.info(f"{'='*80}\n")
            continue

        # Run scenario
        results, metrics = await run_scenario(
            scenario=scenario,
            cases=cases,
            strategies=strategies,
            model_id=model_id,
            api_key=api_key,
            temperature=temperature,
            use_vllm=use_vllm,
        )

        # Save results
        df = pd.DataFrame(results)
        df.to_csv(csv_file, index=False)
        logger.info(f"✓ Results saved to: {csv_file}")

        # Save metrics
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"✓ Metrics saved to: {metrics_file}")

        logger.info("")  # Blank line for readability


def main():
    parser = argparse.ArgumentParser(description='Run unified bias experiments')
    parser.add_argument(
        '--config',
        type=str,
        default='config/replacement_config.json',
        help='Path to replacement config file'
    )
    parser.add_argument(
        '--scenarios',
        type=str,
        nargs='+',
        help='Specific scenario names to run (if not specified, runs all)'
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/real_cases/echr_new/unanimous/balanced_sample_50_50_three_step_ner_updated_v2.json',
        help='Input JSON file with cases'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data/experiments',
        help='Output directory for results'
    )
    parser.add_argument(
        '--model',
        type=str,
        default=DEFAULT_JUDGE_MODEL,
        help='Model to use for evaluation'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.0,
        help='Temperature for model'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force re-run even if results already exist'
    )
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Run scenarios in parallel (evaluate all scenarios for each case simultaneously)'
    )
    parser.add_argument(
        '--use-vllm',
        action='store_true',
        help='Use vLLM with Ray Data for batch inference instead of OpenAI API'
    )

    args = parser.parse_args()

    # Load API key (only required for OpenAI)
    api_key = None
    if not args.use_vllm:
        api_key = os.environ.get('OPENAI_API_KEY')
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in environment. Set it or use --use-vllm flag.")

    # Load config
    logger.info(f"Loading config from {args.config}")
    config = load_config(args.config)
    strategies = config['strategies']
    all_scenarios = config['scenarios']

    # Filter scenarios if specified
    if args.scenarios:
        scenarios_to_run = [s for s in all_scenarios if s['name'] in args.scenarios]
        if len(scenarios_to_run) != len(args.scenarios):
            found = {s['name'] for s in scenarios_to_run}
            requested = set(args.scenarios)
            missing = requested - found
            raise ValueError(f"Scenarios not found in config: {missing}")
    else:
        scenarios_to_run = all_scenarios

    logger.info(f"Will run {len(scenarios_to_run)} scenario(s)")

    # Load cases
    logger.info(f"Loading cases from {args.input}")
    with open(args.input, 'r') as f:
        cases = json.load(f)
    logger.info(f"Loaded {len(cases)} cases")

    # Get model-specific output directory
    model_output_dir = get_model_output_dir(args.output_dir, args.model)
    logger.info(f"Results will be saved to: {model_output_dir}")

    # Run experiments (parallel or sequential)
    if args.parallel:
        if args.use_vllm:
            raise ValueError("Parallel mode (--parallel) is not compatible with vLLM batch mode (--use-vllm). vLLM already processes all cases in batch.")

        asyncio.run(run_all_scenarios_parallel(
            scenarios=scenarios_to_run,
            cases=cases,
            strategies=strategies,
            model_id=args.model,
            api_key=api_key,
            output_dir=str(model_output_dir),
            temperature=args.temperature,
            skip_existing=not args.force,
        ))
    else:
        asyncio.run(run_all_scenarios(
            scenarios=scenarios_to_run,
            cases=cases,
            strategies=strategies,
            model_id=args.model,
            api_key=api_key,
            output_dir=str(model_output_dir),
            temperature=args.temperature,
            skip_existing=not args.force,
            use_vllm=args.use_vllm,
        ))

    logger.info("\n" + "="*80)
    logger.info("ALL SCENARIOS COMPLETED")
    logger.info("="*80)


if __name__ == "__main__":
    main()
