#!/usr/bin/env python3
"""
Full-scale perturbation experiment orchestrator.

Runs all 5 models through all 4 RQs (baseline, summarization/RQ1, framing/RQ2,
reconsideration/RQ3) on the stratified sample (563 case-article pairs,
400 cases, 20 countries).

Supports checkpointing for resumption, dry-run mode, and per-model/per-RQ
filtering.

Usage:
  # Full run (all models, all RQs)
  python scripts/run_full_experiment.py

  # Dry run -- show what would be run
  python scripts/run_full_experiment.py --dry-run

  # Single model
  python scripts/run_full_experiment.py --model opus

  # Single RQ across all models
  python scripts/run_full_experiment.py --rq baseline

  # Specific model + RQ combo
  python scripts/run_full_experiment.py --model nova --rq rq2

  # Override sample count
  python scripts/run_full_experiment.py --samples 3
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    # ── Anthropic (Bedrock) ──────────────────────────────────────────────
    "sonnet": {
        "display_name": "Claude Sonnet 4.6",
        "model_id": "us.anthropic.claude-sonnet-4-6-20250514-v1:0",
        "backend": "bedrock",
        "region": "us-west-2",
        "input_cost_per_mtok": 3.0,
        "output_cost_per_mtok": 15.0,
    },
    # ── OpenAI (OpenRouter) - big + mini ─────────────────────────────────
    "gpt52": {
        "display_name": "GPT-5.2",
        "model_id": "openai/gpt-5.2",
        "backend": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "input_cost_per_mtok": 2.50,
        "output_cost_per_mtok": 10.0,
    },
    "gpt4omini": {
        "display_name": "GPT-4o-mini",
        "model_id": "openai/gpt-4o-mini",
        "backend": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "input_cost_per_mtok": 0.15,
        "output_cost_per_mtok": 0.60,
    },
    # ── Google (OpenRouter) - pro + flash ────────────────────────────────
    "gemini_pro": {
        "display_name": "Gemini Pro",
        "model_id": "google/gemini-2.5-pro-preview",
        "backend": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "input_cost_per_mtok": 1.25,
        "output_cost_per_mtok": 10.0,
    },
    "gemini_flash": {
        "display_name": "Gemini 3.1 Flash",
        "model_id": "google/gemini-3.1-flash",
        "backend": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "input_cost_per_mtok": 0.15,
        "output_cost_per_mtok": 0.60,
    },
    # ── Open weight (cheap APIs) ─────────────────────────────────────────
    "deepseek": {
        "display_name": "DeepSeek V4",
        "model_id": "deepseek-chat",
        "backend": "openai",
        "base_url": "https://api.deepseek.com",
        "api_key_env": "DEEPSEEK_API_KEY",
        "input_cost_per_mtok": 0.27,
        "output_cost_per_mtok": 1.10,
    },
    "kimi": {
        "display_name": "Kimi K2",
        "model_id": "moonshot/kimi-k2",
        "backend": "openai",
        "base_url": "https://openrouter.ai/api/v1",
        "api_key_env": "OPENROUTER_API_KEY",
        "input_cost_per_mtok": 0.20,
        "output_cost_per_mtok": 0.80,
    },
}

RQS = ["baseline", "rq1", "rq2", "rq3"]
RQ_LABELS = {
    "baseline": "Baseline (predictive, full text)",
    "rq1": "RQ1 - Summarization fidelity",
    "rq2": "RQ2 - Prompt framing effects",
    "rq3": "RQ3 - Reconsideration sycophancy",
}

MLFLOW_URL = "http://10.88.0.4:5000"
# Full-scale experiment uses its own MLflow experiment
MLFLOW_EXPERIMENT_NAME = "full_scale_perturbation"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_PATH = REPO_ROOT / "data" / "processed" / "stratified_sample.json"
OUTPUT_ROOT = REPO_ROOT / "data" / "experiments" / "full_scale"
CHECKPOINT_DIR = OUTPUT_ROOT / ".checkpoints"

# The existing runners live here
BEDROCK_RUNNER = REPO_ROOT / "experiments" / "run_perturbation_bedrock.py"
VLLM_RUNNER = REPO_ROOT / "experiments" / "run_perturbation_vllm.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_cases(path: Path) -> list:
    """Load cases and normalize field names for the runners."""
    with open(path) as f:
        cases = json.load(f)
    # The runners expect 'full_case_text_no_verdict' or 'full_case_text'.
    # The stratified sample uses 'verdict_free_text'. Map it.
    for case in cases:
        if "verdict_free_text" in case and "full_case_text_no_verdict" not in case:
            case["full_case_text_no_verdict"] = case["verdict_free_text"]
    return cases


def result_path(model_key: str, rq: str) -> Path:
    """Canonical output path for a given model + RQ result."""
    return OUTPUT_ROOT / model_key / f"{rq}_results.json"


def summaries_path(model_key: str) -> Path:
    return OUTPUT_ROOT / model_key / "summaries.json"


def checkpoint_path(model_key: str, rq: str) -> Path:
    return CHECKPOINT_DIR / f"{model_key}_{rq}.done"


def is_done(model_key: str, rq: str) -> bool:
    """Check whether a model+RQ combination has already completed."""
    return checkpoint_path(model_key, rq).exists()


def mark_done(model_key: str, rq: str) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    checkpoint_path(model_key, rq).write_text(
        datetime.now(timezone.utc).isoformat()
    )


def estimate_api_calls(n_cases: int, n_samples: int) -> dict:
    """Estimate number of API calls per RQ."""
    return {
        "baseline": n_cases * n_samples,
        "rq1": n_cases * 3 + n_cases * 3 * n_samples,  # 3 summaries + 3 evals
        "rq2": n_cases * 3 * n_samples,  # 3 framings
        "rq3": n_cases * n_samples * 2,  # initial + challenge
    }


def estimate_tokens_per_call(rq: str) -> tuple:
    """Rough estimate of (input_tokens, output_tokens) per API call."""
    # Full-text cases average ~15K tokens; summaries ~500 tokens
    estimates = {
        "baseline": (16000, 10),
        "rq1_summary": (15000, 600),
        "rq1_eval": (800, 10),
        "rq2": (800, 10),
        "rq3": (16000, 10),
    }
    return estimates.get(rq, (5000, 50))


def estimate_cost_per_rq(model_key: str, n_cases: int, n_samples: int) -> dict:
    """Estimate cost in USD per RQ for a model."""
    model = MODELS[model_key]
    costs = {}

    # Baseline: n_cases * n_samples calls, full text input, tiny output
    in_tok, out_tok = 16000, 10
    costs["baseline"] = n_cases * n_samples * (
        in_tok * model["input_cost_per_mtok"]
        + out_tok * model["output_cost_per_mtok"]
    ) / 1e6

    # RQ1: summary generation (3 per case, long in, ~600 out) +
    #       evaluation (3 versions * n_samples, short in, tiny out)
    summary_calls = n_cases * 3
    eval_calls = n_cases * 3 * n_samples
    costs["rq1"] = (
        summary_calls * (15000 * model["input_cost_per_mtok"] + 600 * model["output_cost_per_mtok"])
        + eval_calls * (800 * model["input_cost_per_mtok"] + 10 * model["output_cost_per_mtok"])
    ) / 1e6

    # RQ2: 3 framings * n_samples, summary text input (~800 tok)
    costs["rq2"] = n_cases * 3 * n_samples * (
        800 * model["input_cost_per_mtok"]
        + 10 * model["output_cost_per_mtok"]
    ) / 1e6

    # RQ3: n_samples * 2 calls (initial + challenge), full text input
    costs["rq3"] = n_cases * n_samples * 2 * (
        16000 * model["input_cost_per_mtok"]
        + 10 * model["output_cost_per_mtok"]
    ) / 1e6

    return costs


# ---------------------------------------------------------------------------
# Runners -- import and call the existing experiment functions directly
# ---------------------------------------------------------------------------

_mlflow_url = MLFLOW_URL  # mutable at runtime via CLI arg


def _setup_mlflow(model_key: str):
    """Set up MLflow tracking for a model run."""
    import mlflow

    mlflow.set_tracking_uri(_mlflow_url)
    # Create or get the experiment
    experiment = mlflow.get_experiment_by_name(MLFLOW_EXPERIMENT_NAME)
    if experiment is None:
        exp_id = mlflow.create_experiment(MLFLOW_EXPERIMENT_NAME)
    else:
        exp_id = experiment.experiment_id
    mlflow.set_experiment(experiment_id=exp_id)
    return exp_id


def run_bedrock_model(model_key: str, cases: list, rqs: list, n_samples: int):
    """Run experiments for a Bedrock-backed model."""
    import boto3
    import mlflow

    model = MODELS[model_key]
    client = boto3.client("bedrock-runtime", region_name=model["region"])
    exp_id = _setup_mlflow(model_key)

    # Import runner functions
    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    from run_perturbation_bedrock import (
        run_baseline as br_baseline,
        run_summarization as br_summarization,
        run_framing as br_framing,
        run_reconsideration as br_reconsideration,
    )

    model_dir = OUTPUT_ROOT / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

    # Run within a parent MLflow run
    run_name = f"full_{model_key}_{len(cases)}pairs"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model", model["model_id"])
        mlflow.log_param("model_display", model["display_name"])
        mlflow.log_param("n_cases", len(cases))
        mlflow.log_param("n_samples", n_samples)
        mlflow.log_param("stages", ",".join(rqs))
        mlflow.log_param("backend", "bedrock")
        mlflow.log_param("region", model["region"])
        mlflow.log_param("scale", "full")

        start = time.time()
        baseline_results = None
        summaries = None

        # Baseline
        if "baseline" in rqs:
            if is_done(model_key, "baseline"):
                print(f"  [skip] {model_key}/baseline -- already done")
                baseline_results = json.loads(result_path(model_key, "baseline").read_text())
            else:
                print(f"  [run]  {model_key}/baseline ...")
                baseline_results = br_baseline(client, model["model_id"], cases, n_samples)
                result_path(model_key, "baseline").parent.mkdir(parents=True, exist_ok=True)
                result_path(model_key, "baseline").write_text(
                    json.dumps(baseline_results, ensure_ascii=False, indent=2)
                )
                mark_done(model_key, "baseline")
        else:
            # Try to load baseline for dependent stages
            bp = result_path(model_key, "baseline")
            if bp.exists():
                baseline_results = json.loads(bp.read_text())

        # RQ1 -- summarization
        if "rq1" in rqs:
            if baseline_results is None:
                print(f"  [skip] {model_key}/rq1 -- no baseline results available")
            elif is_done(model_key, "rq1"):
                print(f"  [skip] {model_key}/rq1 -- already done")
                sp = summaries_path(model_key)
                if sp.exists():
                    summaries = json.loads(sp.read_text())
            else:
                print(f"  [run]  {model_key}/rq1 ...")
                rq1_results, summaries = br_summarization(
                    client, model["model_id"], cases, n_samples, baseline_results
                )
                result_path(model_key, "rq1").write_text(
                    json.dumps(rq1_results, ensure_ascii=False, indent=2)
                )
                summaries_path(model_key).write_text(
                    json.dumps(summaries, ensure_ascii=False, indent=2)
                )
                mark_done(model_key, "rq1")
        else:
            sp = summaries_path(model_key)
            if sp.exists():
                summaries = json.loads(sp.read_text())

        # RQ2 -- framing
        if "rq2" in rqs:
            if baseline_results is None or summaries is None:
                print(f"  [skip] {model_key}/rq2 -- missing baseline or summaries")
            elif is_done(model_key, "rq2"):
                print(f"  [skip] {model_key}/rq2 -- already done")
            else:
                print(f"  [run]  {model_key}/rq2 ...")
                rq2_results = br_framing(
                    client, model["model_id"], cases, n_samples,
                    summaries, baseline_results
                )
                result_path(model_key, "rq2").write_text(
                    json.dumps(rq2_results, ensure_ascii=False, indent=2)
                )
                mark_done(model_key, "rq2")

        # RQ3 -- reconsideration
        if "rq3" in rqs:
            if baseline_results is None:
                print(f"  [skip] {model_key}/rq3 -- no baseline results available")
            elif is_done(model_key, "rq3"):
                print(f"  [skip] {model_key}/rq3 -- already done")
            else:
                print(f"  [run]  {model_key}/rq3 ...")
                rq3_results = br_reconsideration(
                    client, model["model_id"], cases, n_samples, baseline_results
                )
                result_path(model_key, "rq3").write_text(
                    json.dumps(rq3_results, ensure_ascii=False, indent=2)
                )
                mark_done(model_key, "rq3")

        elapsed = time.time() - start
        mlflow.log_metric("total_time_seconds", elapsed)
        print(f"  {model_key} finished in {elapsed / 60:.1f} min")


def run_vllm_model(model_key: str, cases: list, rqs: list, n_samples: int):
    """Run experiments for a vLLM-backed model."""
    import openai as openai_lib
    import mlflow

    model = MODELS[model_key]
    client = openai_lib.OpenAI(base_url=model["vllm_url"], api_key="dummy")
    exp_id = _setup_mlflow(model_key)

    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    from run_perturbation_vllm import (
        run_baseline as vl_baseline,
        run_summarization as vl_summarization,
        run_framing as vl_framing,
        run_reconsideration as vl_reconsideration,
    )

    model_dir = OUTPUT_ROOT / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"full_{model_key}_{len(cases)}pairs"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model", model["model_id"])
        mlflow.log_param("model_display", model["display_name"])
        mlflow.log_param("n_cases", len(cases))
        mlflow.log_param("n_samples", n_samples)
        mlflow.log_param("stages", ",".join(rqs))
        mlflow.log_param("backend", "vllm-brev")
        mlflow.log_param("scale", "full")

        start = time.time()
        baseline_results = None
        summaries = None

        if "baseline" in rqs:
            if is_done(model_key, "baseline"):
                print(f"  [skip] {model_key}/baseline -- already done")
                baseline_results = json.loads(result_path(model_key, "baseline").read_text())
            else:
                print(f"  [run]  {model_key}/baseline ...")
                baseline_results = vl_baseline(client, model["model_id"], cases, n_samples)
                result_path(model_key, "baseline").parent.mkdir(parents=True, exist_ok=True)
                result_path(model_key, "baseline").write_text(
                    json.dumps(baseline_results, ensure_ascii=False, indent=2)
                )
                mark_done(model_key, "baseline")
        else:
            bp = result_path(model_key, "baseline")
            if bp.exists():
                baseline_results = json.loads(bp.read_text())

        if "rq1" in rqs:
            if baseline_results is None:
                print(f"  [skip] {model_key}/rq1 -- no baseline results available")
            elif is_done(model_key, "rq1"):
                print(f"  [skip] {model_key}/rq1 -- already done")
                sp = summaries_path(model_key)
                if sp.exists():
                    summaries = json.loads(sp.read_text())
            else:
                print(f"  [run]  {model_key}/rq1 ...")
                rq1_results, summaries = vl_summarization(
                    client, model["model_id"], cases, n_samples, baseline_results
                )
                result_path(model_key, "rq1").write_text(
                    json.dumps(rq1_results, ensure_ascii=False, indent=2)
                )
                summaries_path(model_key).write_text(
                    json.dumps(summaries, ensure_ascii=False, indent=2)
                )
                mark_done(model_key, "rq1")
        else:
            sp = summaries_path(model_key)
            if sp.exists():
                summaries = json.loads(sp.read_text())

        if "rq2" in rqs:
            if baseline_results is None or summaries is None:
                print(f"  [skip] {model_key}/rq2 -- missing baseline or summaries")
            elif is_done(model_key, "rq2"):
                print(f"  [skip] {model_key}/rq2 -- already done")
            else:
                print(f"  [run]  {model_key}/rq2 ...")
                rq2_results = vl_framing(
                    client, model["model_id"], cases, n_samples,
                    summaries, baseline_results
                )
                result_path(model_key, "rq2").write_text(
                    json.dumps(rq2_results, ensure_ascii=False, indent=2)
                )
                mark_done(model_key, "rq2")

        if "rq3" in rqs:
            if baseline_results is None:
                print(f"  [skip] {model_key}/rq3 -- no baseline results available")
            elif is_done(model_key, "rq3"):
                print(f"  [skip] {model_key}/rq3 -- already done")
            else:
                print(f"  [run]  {model_key}/rq3 ...")
                rq3_results = vl_reconsideration(
                    client, model["model_id"], cases, n_samples, baseline_results
                )
                result_path(model_key, "rq3").write_text(
                    json.dumps(rq3_results, ensure_ascii=False, indent=2)
                )
                mark_done(model_key, "rq3")

        elapsed = time.time() - start
        mlflow.log_metric("total_time_seconds", elapsed)
        print(f"  {model_key} finished in {elapsed / 60:.1f} min")


def run_openai_model(model_key: str, cases: list, rqs: list, n_samples: int):
    """Run experiments for an OpenAI-compatible API model (DeepSeek, OpenRouter, etc.)."""
    import openai as openai_lib
    import mlflow

    model = MODELS[model_key]
    api_key = os.environ.get(model["api_key_env"], "")
    if not api_key:
        print(f"  ERROR: Set {model['api_key_env']} environment variable")
        return
    client = openai_lib.OpenAI(base_url=model["base_url"], api_key=api_key)
    exp_id = _setup_mlflow(model_key)

    sys.path.insert(0, str(REPO_ROOT / "experiments"))
    from run_perturbation_openai import (
        run_baseline as oa_baseline,
        run_summarization as oa_summarization,
        run_framing as oa_framing,
        run_reconsideration as oa_reconsideration,
    )

    model_dir = OUTPUT_ROOT / model_key
    model_dir.mkdir(parents=True, exist_ok=True)

    run_name = f"full_{model_key}_{len(cases)}pairs"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model", model["model_id"])
        mlflow.log_param("model_display", model["display_name"])
        mlflow.log_param("n_cases", len(cases))
        mlflow.log_param("n_samples", n_samples)
        mlflow.log_param("stages", ",".join(rqs))
        mlflow.log_param("backend", "openai-api")
        mlflow.log_param("base_url", model["base_url"])
        mlflow.log_param("scale", "full")

        start = time.time()
        baseline_results = None
        summaries = None

        if "baseline" in rqs:
            if is_done(model_key, "baseline"):
                print(f"  [skip] {model_key}/baseline -- already done")
                baseline_results = json.loads(result_path(model_key, "baseline").read_text())
            else:
                print(f"  [run]  {model_key}/baseline ...")
                baseline_results = oa_baseline(client, model["model_id"], cases, n_samples)
                result_path(model_key, "baseline").parent.mkdir(parents=True, exist_ok=True)
                result_path(model_key, "baseline").write_text(
                    json.dumps(baseline_results, ensure_ascii=False, indent=2)
                )
                mark_done(model_key, "baseline")
        else:
            bp = result_path(model_key, "baseline")
            if bp.exists():
                baseline_results = json.loads(bp.read_text())

        if "rq1" in rqs:
            if baseline_results is None:
                print(f"  [skip] {model_key}/rq1 -- no baseline results available")
            elif is_done(model_key, "rq1"):
                print(f"  [skip] {model_key}/rq1 -- already done")
                sp = summaries_path(model_key)
                if sp.exists():
                    summaries = json.loads(sp.read_text())
            else:
                print(f"  [run]  {model_key}/rq1 ...")
                rq1_results, summaries = oa_summarization(
                    client, model["model_id"], cases, n_samples, baseline_results
                )
                result_path(model_key, "rq1").write_text(
                    json.dumps(rq1_results, ensure_ascii=False, indent=2)
                )
                summaries_path(model_key).write_text(
                    json.dumps(summaries, ensure_ascii=False, indent=2)
                )
                mark_done(model_key, "rq1")
        else:
            sp = summaries_path(model_key)
            if sp.exists():
                summaries = json.loads(sp.read_text())

        if "rq2" in rqs:
            if baseline_results is None or summaries is None:
                print(f"  [skip] {model_key}/rq2 -- missing baseline or summaries")
            elif is_done(model_key, "rq2"):
                print(f"  [skip] {model_key}/rq2 -- already done")
            else:
                print(f"  [run]  {model_key}/rq2 ...")
                rq2_results = oa_framing(
                    client, model["model_id"], cases, n_samples,
                    summaries, baseline_results
                )
                result_path(model_key, "rq2").write_text(
                    json.dumps(rq2_results, ensure_ascii=False, indent=2)
                )
                mark_done(model_key, "rq2")

        if "rq3" in rqs:
            if baseline_results is None:
                print(f"  [skip] {model_key}/rq3 -- no baseline results available")
            elif is_done(model_key, "rq3"):
                print(f"  [skip] {model_key}/rq3 -- already done")
            else:
                print(f"  [run]  {model_key}/rq3 ...")
                rq3_results = oa_reconsideration(
                    client, model["model_id"], cases, n_samples, baseline_results
                )
                result_path(model_key, "rq3").write_text(
                    json.dumps(rq3_results, ensure_ascii=False, indent=2)
                )
                mark_done(model_key, "rq3")

        elapsed = time.time() - start
        mlflow.log_metric("total_time_seconds", elapsed)
        print(f"  {model_key} finished in {elapsed / 60:.1f} min")


# ---------------------------------------------------------------------------
# Cost summary
# ---------------------------------------------------------------------------

def print_cost_estimate(model_keys: list, rqs: list, n_cases: int, n_samples: int):
    """Print estimated cost breakdown."""
    print("\n" + "=" * 70)
    print("  COST ESTIMATE")
    print("=" * 70)
    calls = estimate_api_calls(n_cases, n_samples)
    total_cost = 0.0

    print(f"\n  Cases: {n_cases}  |  Samples/case: {n_samples}")
    print(f"  RQs:   {', '.join(rqs)}")
    print()

    print(f"  {'Model':<22} {'RQ':>10} {'API calls':>12} {'Est. cost':>12}")
    print(f"  {'-'*22} {'-'*10} {'-'*12} {'-'*12}")

    for mk in model_keys:
        model = MODELS[mk]
        model_total = 0.0
        costs_rq = estimate_cost_per_rq(mk, n_cases, n_samples)
        for rq in rqs:
            n = calls[rq]
            cost_rq = costs_rq.get(rq, 0.0)
            model_total += cost_rq
            print(f"  {model['display_name']:<22} {rq:>10} {n:>12,} {cost_rq:>11.2f}$")
        total_cost += model_total
        print(f"  {'':>22} {'subtotal':>10} {'':>12} {model_total:>11.2f}$")
        print()

    print(f"  {'TOTAL':>44} {total_cost:>11.2f}$")
    print("=" * 70)
    return total_cost


def print_progress_summary(model_keys: list, rqs: list):
    """Print which model+RQ combos are done vs pending."""
    print("\n" + "=" * 70)
    print("  PROGRESS")
    print("=" * 70)

    header = f"  {'Model':<22}"
    for rq in rqs:
        header += f" {rq:>12}"
    print(header)
    print(f"  {'-'*22}" + f" {'-'*12}" * len(rqs))

    for mk in model_keys:
        row = f"  {MODELS[mk]['display_name']:<22}"
        for rq in rqs:
            status = "done" if is_done(mk, rq) else "PENDING"
            row += f" {status:>12}"
        print(row)
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Full-scale perturbation experiment orchestrator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model keys: sonnet, gpt52, gpt4omini, gemini_pro, gemini_flash, deepseek, kimi
RQ keys:    baseline, rq1, rq2, rq3

Examples:
  python scripts/run_full_experiment.py --dry-run
  python scripts/run_full_experiment.py --model deepseek --rq baseline
  python scripts/run_full_experiment.py --model llama405b
  python scripts/run_full_experiment.py --reset-checkpoint sonnet baseline
""",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=str(SAMPLE_PATH),
        help="Path to the stratified sample JSON (default: data/processed/stratified_sample.json)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        choices=list(MODELS.keys()),
        help="Run only this model (default: all)",
    )
    parser.add_argument(
        "--rq",
        type=str,
        default=None,
        choices=RQS,
        help="Run only this RQ (default: all)",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=5,
        help="Number of samples per case (default: 5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be run without executing",
    )
    parser.add_argument(
        "--reset-checkpoint",
        nargs=2,
        metavar=("MODEL", "RQ"),
        help="Reset a specific checkpoint to re-run it (e.g. --reset-checkpoint opus baseline)",
    )
    parser.add_argument(
        "--mlflow-url",
        type=str,
        default=MLFLOW_URL,
        help="MLflow tracking URI",
    )
    args = parser.parse_args()

    # Handle checkpoint reset
    if args.reset_checkpoint:
        mk, rq = args.reset_checkpoint
        cp = checkpoint_path(mk, rq)
        if cp.exists():
            cp.unlink()
            print(f"Reset checkpoint: {mk}/{rq}")
        else:
            print(f"No checkpoint found for {mk}/{rq}")
        return

    # Override module-level MLflow URL if provided via CLI
    global _mlflow_url
    _mlflow_url = args.mlflow_url

    # Load cases
    cases = load_cases(Path(args.cases))
    n_cases = len(cases)
    n_unique = len({c["item_id"] for c in cases})
    n_countries = len({c.get("respondent", "?") for c in cases})

    print(f"\nLoaded {n_cases} case-article pairs ({n_unique} unique cases, {n_countries} countries)")

    # Determine what to run
    model_keys = [args.model] if args.model else list(MODELS.keys())
    rqs = [args.rq] if args.rq else RQS

    # Show progress
    print_progress_summary(model_keys, rqs)

    # Cost estimate
    total_cost = print_cost_estimate(model_keys, rqs, n_cases, args.samples)

    if args.dry_run:
        print("\n  [DRY RUN] No experiments executed.\n")
        return

    # Confirm
    pending = sum(1 for mk in model_keys for rq in rqs if not is_done(mk, rq))
    if pending == 0:
        print("\n  All requested experiments are already complete.")
        print("  Use --reset-checkpoint MODEL RQ to re-run a specific combination.\n")
        return

    print(f"\n  {pending} experiment(s) pending. Starting ...\n")

    # Run experiments sequentially per model
    overall_start = time.time()

    for mk in model_keys:
        model = MODELS[mk]
        # Filter to only pending RQs for this model
        pending_rqs = [rq for rq in rqs if not is_done(mk, rq)]
        if not pending_rqs:
            print(f"\n[{model['display_name']}] all requested RQs already done, skipping")
            continue

        # Even if only some RQs are pending, we pass all requested RQs;
        # the runner functions check is_done() internally
        print(f"\n{'='*70}")
        print(f"  {model['display_name']}  ({model['backend']}, {model.get('region', 'n/a')})")
        print(f"  RQs to run: {', '.join(pending_rqs)}")
        print(f"{'='*70}")

        try:
            if model["backend"] == "bedrock":
                run_bedrock_model(mk, cases, rqs, args.samples)
            elif model["backend"] == "vllm":
                run_vllm_model(mk, cases, rqs, args.samples)
            elif model["backend"] == "openai":
                run_openai_model(mk, cases, rqs, args.samples)
            else:
                print(f"  Unknown backend '{model['backend']}' for {mk}, skipping")
        except KeyboardInterrupt:
            print(f"\n\nInterrupted during {mk}. Progress saved via checkpoints.")
            print("Re-run the same command to resume.\n")
            sys.exit(1)
        except Exception as e:
            print(f"\n  ERROR running {mk}: {e}")
            print("  Continuing with next model ...\n")
            import traceback
            traceback.print_exc()

    overall_elapsed = time.time() - overall_start
    print(f"\n{'='*70}")
    print(f"  ALL DONE in {overall_elapsed / 60:.1f} min ({overall_elapsed / 3600:.2f} h)")
    print(f"  Results in: {OUTPUT_ROOT}")
    print(f"{'='*70}\n")

    # Final progress
    print_progress_summary(model_keys, rqs)


if __name__ == "__main__":
    main()
