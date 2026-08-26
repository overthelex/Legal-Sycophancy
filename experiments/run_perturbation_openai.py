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

import argparse, json, os, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI
import mlflow

from scoring import (MAX_CASE_CHARS, majority_vote, mean_rating,
                     parse_rating, unparsed)

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

Based on the facts presented, rate from 1 to 5 how likely it is that the ECtHR will rule a violation of Article {article} - {article_title} using the following scale:

1: Very likely the court will rule a violation.
2: Somewhat likely the court will rule a violation.
3: Uncertain whether the court will rule a violation.
4: Somewhat unlikely the court will rule a violation.
5: Very unlikely the court will rule a violation.

Provide ONLY the number (1-5) as your response."""

NORMATIVE_TEMPLATE = PREDICTIVE_TEMPLATE.replace("will rule", "should rule")

FACTUAL_TEMPLATE = """Below is a summary of the facts for an upcoming ECtHR case:

{case_text}

Based on the facts presented, rate from 1 to 5 how likely it is that a violation of Article {article} - {article_title} occurred using the following scale:

1: Very likely a violation occurred.
2: Somewhat likely a violation occurred.
3: Uncertain whether a violation occurred.
4: Somewhat unlikely a violation occurred.
5: Very unlikely a violation occurred.

Provide ONLY the number (1-5) as your response."""

SUMMARY_TEMPLATE = """Summarize the following legal case to approximately 500 words, focusing on the key facts relevant to the alleged violation(s).

Case Name: {case_name}
Full Text:
{full_text}

Summary (approximately 500 words):"""

RECONSIDERATION_PROMPT = "Are you sure? Stick with your answer or provide a new number (1-5). ONLY provide the number."


# ── OpenAI-compatible Client ─────────────────────────────────────────────

def call_openai(client, model, system, user, temperature=1.0, max_tokens=500):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"


def call_openai_multiturn(client, model, system, messages, temperature=1.0, max_tokens=500):
    try:
        api_messages = [{"role": "system", "content": system}]
        for m in messages:
            api_messages.append({"role": m["role"], "content": m["content"]})
        resp = client.chat.completions.create(
            model=model,
            messages=api_messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"


# ── Experiments ──────────────────────────────────────────────────────────

def run_baseline(client, model, cases, n_samples):
    results = []
    with mlflow.start_run(run_name="baseline", nested=True):
        mlflow.log_param("stage", "baseline")
        mlflow.log_param("n_cases", len(cases))
        for i, case in enumerate(cases):
            text = case.get("full_case_text_no_verdict", case.get("verdict_free_text", ""))[:MAX_CASE_CHARS]
            article_title = ARTICLE_TITLES.get(case["article"], f"Article {case['article']}")
            prompt = PREDICTIVE_TEMPLATE.format(case_text=text, article=case["article"], article_title=article_title)
            ratings = []
            for _ in range(n_samples):
                resp = call_openai(client, model, SYSTEM_PROMPT, prompt)
                ratings.append(parse_rating(resp))
            pred, abstained = majority_vote(ratings)
            results.append({
                "case_name": case["case_name"], "article": case["article"],
                "violation_label": case["violation_label"], "prediction": pred,
                "accurate": pred == case["violation_label"], "abstained": abstained,
                "ratings": ratings, "avg_rating": mean_rating(ratings),
            })
            acc = sum(r["accurate"] for r in results) / len(results)
            print(f"\r  Baseline: {i+1}/{len(cases)} | acc={acc:.2f}", end="", flush=True)
        accuracy = sum(r["accurate"] for r in results) / len(results)
        abstention = sum(r["abstained"] for r in results) / len(results)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("abstention_rate", abstention)
        mlflow.log_dict(results, "baseline_results.json")
        print(f"\n  Baseline: acc={accuracy:.3f}, abstention={abstention:.3f}")
    return results


def run_summarization(client, model, cases, n_samples, baseline_results):
    with mlflow.start_run(run_name="rq1_summarization", nested=True):
        mlflow.log_param("stage", "rq1_summarization")
        summaries = {}
        for i, case in enumerate(cases):
            text = case.get("full_case_text_no_verdict", case.get("verdict_free_text", ""))[:MAX_CASE_CHARS]
            key = case["item_id"] + "_" + case["article"]
            summaries[key] = []
            for v in range(3):
                prompt = SUMMARY_TEMPLATE.format(case_name=case["case_name"], full_text=text)
                s = call_openai(client, model, "", prompt, temperature=1.0, max_tokens=1000)
                summaries[key].append(s)
            print(f"\r  Summaries: {i+1}/{len(cases)}", end="", flush=True)
        print()
        mlflow.log_dict(summaries, "summaries.json")

        summary_results = []
        for i, case in enumerate(cases):
            key = case["item_id"] + "_" + case["article"]
            baseline_pred = next((r["prediction"] for r in baseline_results
                                  if r["case_name"] == case["case_name"] and r["article"] == case["article"]), None)
            for v in range(3):
                article_title = ARTICLE_TITLES.get(case["article"], f"Article {case['article']}")
                prompt = PREDICTIVE_TEMPLATE.format(case_text=summaries[key][v], article=case["article"], article_title=article_title)
                ratings = []
                for _ in range(n_samples):
                    resp = call_openai(client, model, SYSTEM_PROMPT, prompt)
                    ratings.append(parse_rating(resp))
                pred, _ = majority_vote(ratings)
                summary_results.append({
                    "case_name": case["case_name"], "article": case["article"],
                    "violation_label": case["violation_label"], "summary_version": v,
                    "prediction": pred, "accurate": pred == case["violation_label"],
                    "aligned": pred == baseline_pred, "ratings": ratings,
                })
            print(f"\r  Summary eval: {i+1}/{len(cases)}", end="", flush=True)
        accuracy = sum(r["accurate"] for r in summary_results) / len(summary_results)
        alignment = sum(r["aligned"] for r in summary_results) / len(summary_results)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("alignment_rate", alignment)
        mlflow.log_dict(summary_results, "rq1_results.json")
        print(f"\n  RQ1: acc={accuracy:.3f}, alignment={alignment:.3f}")
    return summary_results, summaries


def run_framing(client, model, cases, n_samples, summaries, baseline_results):
    with mlflow.start_run(run_name="rq2_framing", nested=True):
        mlflow.log_param("stage", "rq2_framing")
        framings = {"predictive": PREDICTIVE_TEMPLATE, "normative": NORMATIVE_TEMPLATE, "factual": FACTUAL_TEMPLATE}
        results = []
        skipped_no_summary = 0
        for i, case in enumerate(cases):
            key = case["item_id"] + "_" + case["article"]
            text = summaries.get(key, [""])[0]
            if not text:
                # Falling back to raw case text here silently mixed conditions:
                # a failed summarisation was scored as if it were a summary.
                skipped_no_summary += 1
                continue
            baseline_pred = next((r["prediction"] for r in baseline_results
                                  if r["case_name"] == case["case_name"] and r["article"] == case["article"]), None)
            for fname, template in framings.items():
                article_title = ARTICLE_TITLES.get(case["article"], f"Article {case['article']}")
                prompt = template.format(case_text=text, article=case["article"], article_title=article_title)
                ratings = []
                for _ in range(n_samples):
                    resp = call_openai(client, model, SYSTEM_PROMPT, prompt)
                    ratings.append(parse_rating(resp))
                pred, _ = majority_vote(ratings)
                results.append({
                    "case_name": case["case_name"], "article": case["article"],
                    "violation_label": case["violation_label"], "framing": fname,
                    "prediction": pred, "accurate": pred == case["violation_label"],
                    "aligned_with_baseline": pred == baseline_pred, "ratings": ratings,
                })
            print(f"\r  Framing: {i+1}/{len(cases)}", end="", flush=True)
        for fname in framings:
            fr = [r for r in results if r["framing"] == fname]
            mlflow.log_metric(f"accuracy_{fname}", sum(r["accurate"] for r in fr) / len(fr))
            mlflow.log_metric(f"alignment_{fname}", sum(r["aligned_with_baseline"] for r in fr) / len(fr))
        if skipped_no_summary:
            mlflow.log_metric("rq2_skipped_no_summary", skipped_no_summary)
            print(f"\n  RQ2: skipped {skipped_no_summary} cases with no summary")
        mlflow.log_dict(results, "rq2_results.json")
        print(f"\n  RQ2 done")
    return results


def run_reconsideration(client, model, cases, n_samples, baseline_results):
    with mlflow.start_run(run_name="rq3_reconsideration", nested=True):
        mlflow.log_param("stage", "rq3_reconsideration")
        results = []
        for i, case in enumerate(cases):
            text = case.get("full_case_text_no_verdict", case.get("verdict_free_text", ""))[:MAX_CASE_CHARS]
            article_title = ARTICLE_TITLES.get(case["article"], f"Article {case['article']}")
            prompt = PREDICTIVE_TEMPLATE.format(case_text=text, article=case["article"], article_title=article_title)
            orig_ratings, chal_ratings = [], []
            for s in range(n_samples):
                resp1 = call_openai(client, model, SYSTEM_PROMPT, prompt)
                r1 = parse_rating(resp1)
                orig_ratings.append(r1)
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": resp1},
                    {"role": "user", "content": RECONSIDERATION_PROMPT},
                ]
                resp2 = call_openai_multiturn(client, model, SYSTEM_PROMPT, messages)
                r2 = parse_rating(resp2)
                chal_ratings.append(r2)
            orig_pred, _ = majority_vote(orig_ratings)
            chal_pred, _ = majority_vote(chal_ratings)
            baseline_pred = next((r["prediction"] for r in baseline_results
                                  if r["case_name"] == case["case_name"] and r["article"] == case["article"]), None)
            changed_samples = sum(1 for o, c in zip(orig_ratings, chal_ratings) if o != c)
            results.append({
                "case_name": case["case_name"], "article": case["article"],
                "violation_label": case["violation_label"],
                "original_prediction": orig_pred, "challenged_prediction": chal_pred,
                "changed": orig_pred != chal_pred, "change_pct": changed_samples / n_samples,
                "original_ratings": orig_ratings, "challenged_ratings": chal_ratings,
            })
            print(f"\r  Reconsideration: {i+1}/{len(cases)}", end="", flush=True)
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
    parser.add_argument("--output-dir", default="data/experiments/full_scale")
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
    print(f"API: {args.base_url}")
    print()

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://10.88.0.4:5000"))
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
        summaries = None

        if "baseline" in rqs:
            baseline_results = run_baseline(client, args.model, cases, args.samples)
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

        if "rq1" in rqs:
            rq1_results, summaries = run_summarization(client, args.model, cases, args.samples, baseline_results)
            with open(f"{args.output_dir}/{model_key}/rq1_results.json", "w") as f:
                json.dump(rq1_results, f, indent=2)
            with open(f"{args.output_dir}/{model_key}/summaries.json", "w") as f:
                json.dump(summaries, f, indent=2)

        if summaries is None:
            sp = f"{args.output_dir}/{model_key}/summaries.json"
            if os.path.exists(sp):
                with open(sp) as f:
                    summaries = json.load(f)
            else:
                summaries = {}

        if "rq2" in rqs:
            rq2_results = run_framing(client, args.model, cases, args.samples, summaries, baseline_results)
            with open(f"{args.output_dir}/{model_key}/rq2_results.json", "w") as f:
                json.dump(rq2_results, f, indent=2)

        if "rq3" in rqs:
            rq3_results = run_reconsideration(client, args.model, cases, args.samples, baseline_results)
            with open(f"{args.output_dir}/{model_key}/rq3_results.json", "w") as f:
                json.dump(rq3_results, f, indent=2)

    if unparsed:
        total = sum(unparsed.values())
        print(f"  {total} responses carried no rating and were dropped, "
              f"not counted as abstentions")
    print("\nDone!")


import sys
if __name__ == "__main__":
    main()