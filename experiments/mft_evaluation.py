"""
Step 1b — Evaluate MFT minimal case versions (the competence floor).

Reads the minimal one-paragraph versions produced by generate_mft.py, asks each
evaluator model to judge them (N samples, temp 1.0), and writes a per-case CSV
with pass/fail plus the stratification axes (group, respondent, ...) so the
analysis step can slice by group / country / time.

Input : data/mft/mft_cases.json
Output: data/experiments/mft/<evaluator>_mft_samples<N>.csv
"""

import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd
from dotenv import load_dotenv
load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from lib.queued_client import QueuedLLMClient
from lib.evaluation import prepare_evaluation_requests, process_evaluation_results
from lib.prompts import EVALUATION_SYSTEM_PROMPT, BASELINE_EVALUATION_TEMPLATE
from lib.metrics import calculate_accuracy

# Evaluator roster. Edit model_id strings to match valid OpenRouter ids on your
# account. Verify availability with:
#   curl -s https://openrouter.ai/api/v1/models -H "Authorization: Bearer $OPENROUTER_API_KEY"
# LATEST FRONTIER flagships (one per major lab, verified on account 2026-07-11):
EVALUATORS = {
    # --- cross-lab flagships (1 per lab) ---
    "gpt-5.6":           {"model_id": "openai/gpt-5.6-sol",           "use_openrouter": True, "api_key_env": "OPENROUTER_API_KEY"},
    "claude-opus-4.8":   {"model_id": "anthropic/claude-opus-4.8",   "use_openrouter": True, "api_key_env": "OPENROUTER_API_KEY"},
    "gemini-3.5-flash":  {"model_id": "google/gemini-3.5-flash",     "use_openrouter": True, "api_key_env": "OPENROUTER_API_KEY"},
    "deepseek-v4":       {"model_id": "deepseek/deepseek-v4-pro",    "use_openrouter": True, "api_key_env": "OPENROUTER_API_KEY"},
    # --- size track (open-weight ladders) ---
    "deepseek-v4-flash": {"model_id": "deepseek/deepseek-v4-flash",  "use_openrouter": True, "api_key_env": "OPENROUTER_API_KEY"},
    "qwen3-8b":          {"model_id": "qwen/qwen3-8b",               "use_openrouter": True, "api_key_env": "OPENROUTER_API_KEY"},
    "qwen3-32b":         {"model_id": "qwen/qwen3-32b",              "use_openrouter": True, "api_key_env": "OPENROUTER_API_KEY"},
    "qwen3-235b":        {"model_id": "qwen/qwen3-235b-a22b",        "use_openrouter": True, "api_key_env": "OPENROUTER_API_KEY"},
    # second Claude kept for reference; team chose ONE Claude for the balanced roster:
    # "claude-sonnet-5": {"model_id": "anthropic/claude-sonnet-5",   "use_openrouter": True, "api_key_env": "OPENROUTER_API_KEY"},
    # --- previous-gen comparators (already run on 2K; uncomment to redo) ---
    # "gpt-5.2":          {"model_id": "openai/gpt-5.2",              "use_openrouter": True, "api_key_env": "OPENROUTER_API_KEY"},
    # "claude-sonnet-4.5":{"model_id": "anthropic/claude-sonnet-4.5","use_openrouter": True, "api_key_env": "OPENROUTER_API_KEY"},
    # "deepseek-v3.2":    {"model_id": "deepseek/deepseek-v3.2",     "use_openrouter": True, "api_key_env": "OPENROUTER_API_KEY"},
    # "gpt-4o":           {"model_id": "openai/gpt-4o",              "use_openrouter": True, "api_key_env": "OPENROUTER_API_KEY"},
    # --- bleeding-edge OpenAI (codenamed variants) ---
    # "gpt-5.6-sol":      {"model_id": "openai/gpt-5.6-sol",         "use_openrouter": True, "api_key_env": "OPENROUTER_API_KEY"},
}

MFT_CASES_PATH = REPO_ROOT / "data" / "mft" / "mft_cases.json"
OUTPUT_DIR = REPO_ROOT / "data" / "experiments" / "mft"
MFT_TEXT_KEY = "mft_text"

# Extra per-case fields carried from the input into the result CSV for slicing.
STRATA_FIELDS = ["item_id", "respondent", "group", "decision_date", "importance"]


def evaluate_mft(cases, evaluator_name, evaluator_config, api_key,
                 num_samples, temperature, max_workers) -> pd.DataFrame:
    print(f"\n{'=' * 80}")
    print(f"MFT Evaluation - {evaluator_name}  ({evaluator_config['model_id']})")
    print(f"Cases: {len(cases)}, Samples: {num_samples}, Workers: {max_workers}")
    print(f"{'=' * 80}")

    # Lookup so we can re-attach strata after aggregation (keyed by item_id).
    strata_by_id = {
        str(c.get("item_id", "")): {k: c.get(k, "") for k in STRATA_FIELDS}
        for c in cases
    }

    print("\nPreparing evaluation requests...")
    requests = prepare_evaluation_requests(
        cases=cases,
        model_id=evaluator_config["model_id"],
        api_key=api_key,
        use_openrouter=evaluator_config["use_openrouter"],
        system_prompt=EVALUATION_SYSTEM_PROMPT,
        user_prompt_template=BASELINE_EVALUATION_TEMPLATE,
        num_samples=num_samples,
        temperature=temperature,
        max_tokens=100,
        case_text_key=MFT_TEXT_KEY,
    )
    print(f"Prepared {len(requests)} requests "
          f"({len(cases)} cases x {num_samples} samples)")

    print("\nExecuting evaluation...")
    client = QueuedLLMClient(max_workers=max_workers, timeout=120)

    def progress_callback(completed, total):
        if completed % 100 == 0 or completed == total:
            print(f"  Progress: {completed}/{total} ({100 * completed / total:.1f}%)")

    results = client.execute_all(requests, progress_callback=progress_callback)

    print("\nProcessing results...")
    case_results = process_evaluation_results(results, cases, num_samples)

    rows = []
    for case_result in case_results:
        # FIX: violation_label is the string 'violation'/'no_violation'. Convert
        # to bool before scoring — passing the raw string makes every label
        # truthy and silently inverts scoring for no_violation cases.
        is_violation = case_result["violation_label"] == "violation"
        is_accurate = calculate_accuracy(case_result["avg_rating"], is_violation)

        row = {
            "case_name": case_result["case_name"],
            "article": case_result["article"],
            "violation_label": case_result["violation_label"],
            "avg_rating": case_result["avg_rating"],
            "is_accurate": is_accurate,
            "num_abstentions": case_result["num_abstentions"],
            "num_samples": num_samples,
            "sample_ratings": str(case_result["sample_ratings"]),
        }
        row.update(strata_by_id.get(str(case_result.get("item_id", "")), {}))
        rows.append(row)

    df = pd.DataFrame(rows)
    accuracy = df["is_accurate"].mean()
    abstention_rate = df["num_abstentions"].sum() / (len(df) * num_samples)
    print("\nResults:")
    print(f"  MFT pass rate (accuracy): {accuracy:.1%}")
    print(f"  Abstention rate         : {abstention_rate:.1%}")
    return df


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate MFT minimal case versions (competence floor)"
    )
    parser.add_argument("--evaluator", choices=list(EVALUATORS.keys()),
                        help="Single evaluator to run")
    parser.add_argument("--all-evaluators", action="store_true",
                        help="Run every evaluator in EVALUATORS")
    parser.add_argument("--num-samples", type=int, default=10)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--max-workers", type=int, default=25)
    parser.add_argument("--mft-cases", type=Path, default=MFT_CASES_PATH)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--limit", type=int, default=None,
                        help="Only evaluate the first N cases (cheap smoke test)")
    args = parser.parse_args()

    if not args.all_evaluators and not args.evaluator:
        parser.error("Must specify either --evaluator or --all-evaluators")

    evaluators_to_run = (
        list(EVALUATORS.keys()) if args.all_evaluators else [args.evaluator]
    )

    print("MFT EVALUATION")
    print(f"Evaluators      : {', '.join(evaluators_to_run)}")
    print(f"Samples per case: {args.num_samples}")
    print(f"Max workers     : {args.max_workers}")
    print(f"MFT cases       : {args.mft_cases}")
    print(f"Output          : {args.output_dir}")

    if not args.mft_cases.exists():
        print(f"\nERROR: {args.mft_cases} not found. Run experiments/generate_mft.py")
        sys.exit(1)
    with open(args.mft_cases) as f:
        cases = json.load(f)
    print(f"\nLoaded {len(cases)} MFT cases")

    usable_cases = [c for c in cases if c.get(MFT_TEXT_KEY, "").strip()]
    dropped = len(cases) - len(usable_cases)
    if dropped:
        print(f"Dropping {dropped} cases with empty '{MFT_TEXT_KEY}' "
              f"(failed generation). Evaluating {len(usable_cases)} cases.")
    cases = usable_cases
    if not cases:
        print("ERROR: No usable MFT cases. Check generation output.")
        sys.exit(1)
    if args.limit:
        cases = cases[:args.limit]
        print(f"[--limit] Restricting to first {len(cases)} cases")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    for evaluator_name in evaluators_to_run:
        evaluator_config = EVALUATORS[evaluator_name]
        api_key = os.getenv(evaluator_config["api_key_env"])
        if not api_key:
            print(f"\nSkipping {evaluator_name}: "
                  f"{evaluator_config['api_key_env']} not set")
            continue

        df = evaluate_mft(
            cases=cases,
            evaluator_name=evaluator_name,
            evaluator_config=evaluator_config,
            api_key=api_key,
            num_samples=args.num_samples,
            temperature=args.temperature,
            max_workers=args.max_workers,
        )
        output_path = args.output_dir / f"{evaluator_name}_mft_samples{args.num_samples}.csv"
        df.to_csv(output_path, index=False)
        print(f"Saved results to {output_path}")

    print("\nMFT Evaluation complete!")


if __name__ == "__main__":
    main()
