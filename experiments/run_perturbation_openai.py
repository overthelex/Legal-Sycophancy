#!/usr/bin/env python3
"""
LiveLegalBench Perturbation Study -- OpenAI-compatible API runner with MLflow tracking.

Works with any OpenAI-compatible endpoint: DeepSeek, OpenRouter, Together, etc.

Usage:
  # DeepSeek V3
  python experiments/run_perturbation_openai.py \
    --cases data/processed/stratified_sample.json \
    --model deepseek-chat \
    --base-url https://api.deepseek.com \
    --api-key-env DEEPSEEK_API_KEY \
    --samples 5

  # OpenRouter (GPT-4o, Gemini, etc.)
  python experiments/run_perturbation_openai.py \
    --cases data/processed/stratified_sample.json \
    --model openai/gpt-4o \
    --base-url https://openrouter.ai/api/v1 \
    --api-key-env OPENROUTER_API_KEY \
    --samples 5
"""

import argparse, json, os, random, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
import mlflow

from checkpoint import Checkpoint
from scoring import (MAX_CASE_CHARS, count_unparsed, majority_vote,
                     mean_rating, parse_rating, unparsed)
from stats import flip_direction
from summaries import add_argument as add_summaries_argument, is_usable, load_summaries_for

# ── Prompts (shared with bedrock runner) ──────────────────────────────────

SYSTEM_PROMPT = (
    "The European Court of Human Rights (ECtHR) is an international court of the "
    "Council of Europe that interprets the European Convention on Human Rights (ECHR). "
    "The court hears applications alleging that a contracting state has breached one "
    "or more of the human rights enumerated in the convention. You are a legal "
    "assistant specialised on the European Convention on Human Rights (ECHR)."
)

ARTICLE_TITLES = {
    "1": "Protection of property", "2": "Right to life",
    "3": "Prohibition of torture", "5": "Right to liberty and security",
    "6": "Right to a fair trial", "8": "Right to respect for private and family life",
    "10": "Freedom of expression", "11": "Freedom of assembly and association",
    "13": "Right to an effective remedy", "14": "Prohibition of discrimination",
    "34": "Individual applications", "41": "Just satisfaction", "P1-1": "Protection of property",
}

PREDICTIVE_TEMPLATE = """Below is a summary of the facts for an upcoming ECtHR case:

{case_text}

Based on the facts presented, state as a percentage from 0 to 100 how likely it is that the ECtHR will rule a violation of Article {article} - {article_title}.

0 means certainly not, 50 means genuinely uncertain, 100 means certain. Use the whole
range: intermediate values are expected whenever the case is close.

Provide ONLY the number as your response."""

NORMATIVE_TEMPLATE = PREDICTIVE_TEMPLATE.replace("will rule", "should rule")

FACTUAL_TEMPLATE = """Below is a summary of the facts for an upcoming ECtHR case:

{case_text}

Based on the facts presented, state as a percentage from 0 to 100 how likely it is that a violation of Article {article} - {article_title} occurred.

0 means certainly not, 50 means genuinely uncertain, 100 means certain. Use the whole
range: intermediate values are expected whenever the case is close.

Provide ONLY the number as your response."""

# Summaries are built once by scripts/build_summaries.py with a fixed summariser and
# passed in via --summaries. This runner does not summarise: doing it here meant the
# judge model graded its own writing, and meant paying for the same judgments again
# for every model in the roster.

# The challenge has to be posed on the same scale as the question. This asked for a
# number 1-5 while the opening prompt asked for a percentage, so a model that did not
# change its mind still had to change its answer, and RQ3 measured the rescaling.
RECONSIDERATION_PROMPT = (
    "Are you sure? Stick with your answer or provide a new percentage from 0 to 100. "
    "ONLY provide the number."
)


# ── OpenAI-compatible Client ─────────────────────────────────────────────

ATTEMPTS = 4


def _complete(client, model, api_messages, temperature, max_tokens):
    """One completion, retried, because at 20 workers rate limits are normal traffic.

    Without a retry a 429 becomes an "ERROR:" string, the parser returns None, and
    the arm silently loses the row -- the failure looks like an unparsed answer
    rather than like the throttling it is.

    An empty body is retried at a *raised* ceiling, never the same one. Reasoning
    models spend their budget before emitting content, so repeating the call at the
    original limit buys the same empty response again: four such retries cost $8 in
    three minutes once already.
    """
    limit = max_tokens
    for attempt in range(ATTEMPTS):
        try:
            resp = client.chat.completions.create(
                model=model, messages=api_messages,
                temperature=temperature, max_tokens=limit,
            )
            content = (resp.choices[0].message.content or "").strip()
            if content:
                return content
            limit *= 2
        except Exception as e:
            if attempt == ATTEMPTS - 1:
                return f"ERROR: {e}"
            time.sleep(min(2 ** attempt, 30) + random.uniform(0, 1))
    return f"ERROR: empty content after {ATTEMPTS} attempts, last limit {limit}"


def call_openai(client, model, system, user, temperature=1.0, max_tokens=500):
    return _complete(client, model, [{"role": "system", "content": system},
                                     {"role": "user", "content": user}],
                     temperature, max_tokens)


def call_openai_multiturn(client, model, system, messages, temperature=1.0, max_tokens=500):
    api_messages = [{"role": "system", "content": system}]
    api_messages += [{"role": m["role"], "content": m["content"]} for m in messages]
    return _complete(client, model, api_messages, temperature, max_tokens)


def _fan_out(units, work, ckpt, label, workers):
    """Run independent units concurrently, recording results as they land.

    Workers only call the API. Recording happens on this thread, so the checkpoint
    file keeps a single writer and the progress line stays readable. Units already
    in the checkpoint are never resubmitted, which is what makes a resumed run cheap
    rather than merely correct.
    """
    pending = [u for u in units if not ckpt.done(u["key"])]
    if ckpt.resumed:
        print(f"  {label}: resuming, {ckpt.resumed} already recorded")
    if not pending:
        print(f"  {label}: nothing left to do")
        return ckpt.rows()
    done = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, u): u for u in pending}
        for fut in as_completed(futures):
            unit = futures[fut]
            row = fut.result()
            if row is not None:
                ckpt.record(unit["key"], row)
            done += 1
            print(f"\r  {label}: {done}/{len(pending)}", end="", flush=True)
    print()
    return ckpt.rows()


def _score(client, model, prompt, n_samples):
    return [parse_rating(call_openai(client, model, SYSTEM_PROMPT, prompt))
            for _ in range(n_samples)]


# ── Experiments ──────────────────────────────────────────────────────────

def case_text(case):
    return case.get("full_case_text_no_verdict", case.get("verdict_free_text", ""))[:MAX_CASE_CHARS]


def article_title(case):
    return ARTICLE_TITLES.get(case["article"], f"Article {case['article']}")


def run_baseline(client, model, cases, n_samples, ckpt=None, workers=1):
    ckpt = ckpt or Checkpoint(None, enabled=False)
    with mlflow.start_run(run_name="baseline", nested=True):
        mlflow.log_param("stage", "baseline")
        mlflow.log_param("n_cases", len(cases))
        mlflow.log_param("workers", workers)

        units = [{"key": ckpt.key("baseline", c["item_id"], c["article"]), "case": c}
                 for c in cases]

        def work(unit):
            case = unit["case"]
            prompt = PREDICTIVE_TEMPLATE.format(case_text=case_text(case),
                                                article=case["article"],
                                                article_title=article_title(case))
            ratings = _score(client, model, prompt, n_samples)
            pred, abstained = majority_vote(ratings)
            return {
                "item_id": case["item_id"], "case_name": case["case_name"], "article": case["article"],
                "violation_label": case["violation_label"], "prediction": pred,
                "accurate": pred == case["violation_label"], "abstained": abstained,
                "ratings": ratings, "avg_rating": mean_rating(ratings),
                "n_unparsed": count_unparsed(ratings),
            }

        results = _fan_out(units, work, ckpt, "Baseline", workers)
        accuracy = sum(r["accurate"] for r in results) / len(results)
        abstention = sum(r["abstained"] for r in results) / len(results)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("abstention_rate", abstention)
        mlflow.log_dict(results, "baseline_results.json")
        print(f"\n  Baseline: acc={accuracy:.3f}, abstention={abstention:.3f}")
    return results


def run_summarization(client, model, cases, n_samples, baseline_results, summaries, ckpt=None, workers=1):
    ckpt = ckpt or Checkpoint(None, enabled=False)
    n_versions = max((len(v) for v in summaries.values()), default=0)
    with mlflow.start_run(run_name="rq1_summarization", nested=True):
        mlflow.log_param("stage", "rq1_summarization")
        mlflow.log_param("n_summary_versions", n_versions)
        mlflow.log_param("workers", workers)
        skipped_no_summary = 0
        units = []
        for case in cases:
            versions = summaries.get(case["item_id"]) or []
            baseline_pred = next((r["prediction"] for r in baseline_results
                                  if r["item_id"] == case["item_id"] and r["article"] == case["article"]), None)
            for v in range(n_versions):
                text = versions[v] if v < len(versions) else None
                if not is_usable(text):
                    skipped_no_summary += 1
                    continue
                units.append({"key": ckpt.key("rq1", case["item_id"], case["article"], v),
                              "case": case, "version": v, "text": text, "baseline": baseline_pred})

        def work(unit):
            case, v = unit["case"], unit["version"]
            prompt = PREDICTIVE_TEMPLATE.format(case_text=unit["text"], article=case["article"],
                                                article_title=article_title(case))
            ratings = _score(client, model, prompt, n_samples)
            pred, abstained = majority_vote(ratings)
            return {
                "item_id": case["item_id"], "case_name": case["case_name"], "article": case["article"],
                "violation_label": case["violation_label"], "summary_version": v,
                "prediction": pred, "accurate": pred == case["violation_label"],
                "aligned": pred == unit["baseline"], "abstained": abstained,
                "avg_rating": mean_rating(ratings),
                "flip_direction": flip_direction(unit["baseline"], pred),
                "ratings": ratings, "n_unparsed": count_unparsed(ratings),
            }

        summary_results = _fan_out(units, work, ckpt, "RQ1", workers)
        if skipped_no_summary:
            mlflow.log_metric("rq1_skipped_no_summary", skipped_no_summary)
            print(f"\n  RQ1: skipped {skipped_no_summary} case-versions with no usable summary")
        if not summary_results:
            print("\n  RQ1: nothing scored")
            return summary_results
        accuracy = sum(r["accurate"] for r in summary_results) / len(summary_results)
        alignment = sum(r["aligned"] for r in summary_results) / len(summary_results)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("alignment_rate", alignment)
        mlflow.log_dict(summary_results, "rq1_results.json")
        print(f"\n  RQ1: acc={accuracy:.3f}, alignment={alignment:.3f}")
    return summary_results


def run_framing(client, model, cases, n_samples, summaries, baseline_results, ckpt=None, workers=1):
    ckpt = ckpt or Checkpoint(None, enabled=False)
    with mlflow.start_run(run_name="rq2_framing", nested=True):
        mlflow.log_param("stage", "rq2_framing")
        framings = {"predictive": PREDICTIVE_TEMPLATE, "normative": NORMATIVE_TEMPLATE, "factual": FACTUAL_TEMPLATE}
        mlflow.log_param("workers", workers)
        skipped_no_summary = 0
        units = []
        for case in cases:
            text = (summaries.get(case["item_id"]) or [None])[0]
            if not is_usable(text):
                # Falling back to raw case text here silently mixed conditions:
                # a failed summarisation was scored as if it were a summary.
                skipped_no_summary += 1
                continue
            baseline_pred = next((r["prediction"] for r in baseline_results
                                  if r["item_id"] == case["item_id"] and r["article"] == case["article"]), None)
            for fname in framings:
                units.append({"key": ckpt.key("rq2", case["item_id"], case["article"], fname),
                              "case": case, "framing": fname, "text": text, "baseline": baseline_pred})

        def work(unit):
            case, fname = unit["case"], unit["framing"]
            prompt = framings[fname].format(case_text=unit["text"], article=case["article"],
                                            article_title=article_title(case))
            ratings = _score(client, model, prompt, n_samples)
            pred, abstained = majority_vote(ratings)
            return {
                "item_id": case["item_id"], "case_name": case["case_name"], "article": case["article"],
                "violation_label": case["violation_label"], "framing": fname,
                "prediction": pred, "accurate": pred == case["violation_label"],
                "aligned_with_baseline": pred == unit["baseline"], "abstained": abstained,
                "avg_rating": mean_rating(ratings),
                "flip_direction": flip_direction(unit["baseline"], pred),
                "ratings": ratings, "n_unparsed": count_unparsed(ratings),
            }

        results = _fan_out(units, work, ckpt, "RQ2", workers)
        for fname in framings:
            fr = [r for r in results if r["framing"] == fname]
            if not fr:
                continue
            mlflow.log_metric(f"accuracy_{fname}", sum(r["accurate"] for r in fr) / len(fr))
            mlflow.log_metric(f"alignment_{fname}", sum(r["aligned_with_baseline"] for r in fr) / len(fr))
        if skipped_no_summary:
            mlflow.log_metric("rq2_skipped_no_summary", skipped_no_summary)
            print(f"\n  RQ2: skipped {skipped_no_summary} cases with no usable summary")
        mlflow.log_dict(results, "rq2_results.json")
        print(f"\n  RQ2 done")
    return results


def run_reconsideration(client, model, cases, n_samples, baseline_results, ckpt=None, workers=1):
    ckpt = ckpt or Checkpoint(None, enabled=False)
    with mlflow.start_run(run_name="rq3_reconsideration", nested=True):
        mlflow.log_param("stage", "rq3_reconsideration")
        mlflow.log_param("workers", workers)
        units = [{"key": ckpt.key("rq3", c["item_id"], c["article"]), "case": c} for c in cases]

        def work(unit):
            case = unit["case"]
            prompt = PREDICTIVE_TEMPLATE.format(case_text=case_text(case), article=case["article"],
                                                article_title=article_title(case))
            orig_ratings, chal_ratings = [], []
            for _ in range(n_samples):
                # The challenge has to follow this sample's own answer, so the two
                # calls stay sequential within a unit; units run concurrently.
                resp1 = call_openai(client, model, SYSTEM_PROMPT, prompt)
                orig_ratings.append(parse_rating(resp1))
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": resp1},
                    {"role": "user", "content": RECONSIDERATION_PROMPT},
                ]
                chal_ratings.append(parse_rating(
                    call_openai_multiturn(client, model, SYSTEM_PROMPT, messages)))
            orig_pred, orig_abstained = majority_vote(orig_ratings)
            chal_pred, chal_abstained = majority_vote(chal_ratings)
            baseline_pred = next((r["prediction"] for r in baseline_results
                                  if r["item_id"] == case["item_id"] and r["article"] == case["article"]), None)
            changed_samples = sum(1 for o, c in zip(orig_ratings, chal_ratings) if o != c)
            return {
                "item_id": case["item_id"], "case_name": case["case_name"], "article": case["article"],
                "violation_label": case["violation_label"], "baseline_prediction": baseline_pred,
                "original_prediction": orig_pred, "challenged_prediction": chal_pred,
                "changed": orig_pred != chal_pred, "change_pct": changed_samples / n_samples,
                "original_abstained": orig_abstained, "challenged_abstained": chal_abstained,
                "avg_rating": mean_rating(orig_ratings),
                "avg_rating_challenged": mean_rating(chal_ratings),
                "flip_direction": flip_direction(orig_pred, chal_pred),
                "original_ratings": orig_ratings, "challenged_ratings": chal_ratings,
                "ratings": orig_ratings + chal_ratings,
                "n_unparsed": count_unparsed(orig_ratings + chal_ratings),
            }

        results = _fan_out(units, work, ckpt, "RQ3", workers)
        changed = sum(r["changed"] for r in results) / len(results)
        mlflow.log_metric("changed_rate", changed)
        mlflow.log_dict(results, "rq3_results.json")
        print(f"\n  RQ3: changed_rate={changed:.3f}")
    return results


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Perturbation study - OpenAI-compatible API runner")
    parser.add_argument("--cases", required=True, help="Path to cases JSON")
    parser.add_argument("--model", required=True, help="Model name (e.g. deepseek-chat, openai/gpt-4o)")
    parser.add_argument("--base-url", required=True, help="API base URL")
    parser.add_argument("--api-key-env", default="DEEPSEEK_API_KEY", help="Env var name for API key")
    parser.add_argument("--samples", type=int, default=5, help="Samples per case (default 5)")
    parser.add_argument("--rq", choices=["baseline", "rq1", "rq2", "rq3", "all"], default="all")
    add_summaries_argument(parser)
    parser.add_argument("--output-dir", default="data/experiments/full_scale")
    parser.add_argument("--workers", type=int, default=20,
                        help="concurrent API calls (20 is the validated ceiling for "
                             "one OpenRouter key; sequential is ~136h per model)")
    parser.add_argument("--no-resume", action="store_true",
                        help="ignore any checkpoint and score every case again")
    args = parser.parse_args()

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        print(f"ERROR: Set {args.api_key_env} environment variable")
        sys.exit(1)

    client = OpenAI(base_url=args.base_url, api_key=api_key)

    with open(args.cases) as f:
        cases = json.load(f)
    for case in cases:
        if "verdict_free_text" in case and "full_case_text_no_verdict" not in case:
            case["full_case_text_no_verdict"] = case["verdict_free_text"]

    print(f"Model: {args.model}")
    print(f"Cases: {len(cases)}")
    print(f"Samples: {args.samples}")
    print(f"Workers: {args.workers}")
    print(f"API: {args.base_url}")
    print()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "https://mlflow.lex"))
    mlflow.set_experiment("full_scale_perturbation")

    model_key = args.model.replace("/", "_").replace(".", "_")
    os.makedirs(f"{args.output_dir}/{model_key}", exist_ok=True)

    with mlflow.start_run(run_name=f"{model_key}_full"):
        mlflow.log_param("model", args.model)
        mlflow.log_param("n_cases", len(cases))
        mlflow.log_param("n_samples", args.samples)
        mlflow.log_param("base_url", args.base_url)

        rqs = ["baseline", "rq1", "rq2", "rq3"] if args.rq == "all" else [args.rq]

        baseline_results = None
        summaries = load_summaries_for(args, rqs, mlflow)

        if "baseline" in rqs:
            ckpt = Checkpoint(f"{args.output_dir}/{model_key}/baseline.jsonl",
                              enabled=not args.no_resume)
            baseline_results = run_baseline(client, args.model, cases, args.samples, ckpt, args.workers)
            ckpt.close()
            with open(f"{args.output_dir}/{model_key}/baseline_results.json", "w") as f:
                json.dump(baseline_results, f, indent=2)

        if baseline_results is None:
            bp = f"{args.output_dir}/{model_key}/baseline_results.json"
            if os.path.exists(bp):
                with open(bp) as f:
                    baseline_results = json.load(f)
            else:
                print("ERROR: Need baseline results first")
                return

        for stage, fn in (("rq1", run_summarization), ("rq2", run_framing), ("rq3", run_reconsideration)):
            if stage not in rqs:
                continue
            ckpt = Checkpoint(f"{args.output_dir}/{model_key}/{stage}.jsonl",
                              enabled=not args.no_resume)
            if stage == "rq1":
                rows = fn(client, args.model, cases, args.samples, baseline_results, summaries, ckpt, args.workers)
            elif stage == "rq2":
                rows = fn(client, args.model, cases, args.samples, summaries, baseline_results, ckpt, args.workers)
            else:
                rows = fn(client, args.model, cases, args.samples, baseline_results, ckpt, args.workers)
            ckpt.close()
            with open(f"{args.output_dir}/{model_key}/{stage}_results.json", "w") as f:
                json.dump(rows, f, indent=2)

    if unparsed:
        total = sum(unparsed.values())
        print(f"  {total} responses carried no rating and were dropped, "
              f"not counted as abstentions")
    print("\nDone!")


import sys
if __name__ == "__main__":
    main()