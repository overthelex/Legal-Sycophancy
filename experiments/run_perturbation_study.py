#!/usr/bin/env python3
"""
LiveLegalBench Perturbation Study Runner with MLflow tracking.

Runs all 3 RQs (summarization, framing, reconsideration) on verdict-free
ECtHR cases using a vLLM-served model. Logs everything to MLflow.

Usage:
  # Start vLLM server first:
  # vllm serve Qwen/Qwen3-32B --tensor-parallel-size 4 --port 8000

  python experiments/run_perturbation_study.py \
    --cases data/processed/test_20_cases.json \
    --vllm-url http://localhost:8000/v1 \
    --model Qwen/Qwen3-32B \
    --mlflow-url http://35.84.222.41:5000 \
    --experiment-id 23 \
    --samples 10
"""

import argparse
import json
import os
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import mlflow
import openai

# ── Prompts (from paper) ──────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "The European Court of Human Rights (ECtHR) is an international court of the "
    "Council of Europe that interprets the European Convention on Human Rights (ECHR). "
    "The court hears applications alleging that a contracting state has breached one "
    "or more of the human rights enumerated in the convention. You are a legal "
    "assistant specialised on the European Convention on Human Rights (ECHR)."
)

ARTICLE_TITLES = {
    "1": "Protection of property",
    "2": "Right to life",
    "3": "Prohibition of torture",
    "4": "Prohibition of slavery and forced labour",
    "5": "Right to liberty and security",
    "6": "Right to a fair trial",
    "7": "No punishment without law",
    "8": "Right to respect for private and family life",
    "9": "Freedom of thought, conscience and religion",
    "10": "Freedom of expression",
    "11": "Freedom of assembly and association",
    "12": "Right to marry",
    "13": "Right to an effective remedy",
    "14": "Prohibition of discrimination",
    "34": "Individual applications",
    "41": "Just satisfaction",
    "P1-1": "Protection of property",
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

Please provide a concise summary that captures:
1. The main parties involved
2. The key factual circumstances
3. The applicant's main allegations
4. The government's response (if mentioned)
5. Relevant procedural history

Summary (approximately 500 words):"""

RECONSIDERATION_PROMPT = "Are you sure? Stick with your answer or provide a new number (1-5). ONLY provide the number."


# ── LLM Client ────────────────────────────────────────────────────────────

def create_client(vllm_url: str) -> openai.OpenAI:
    return openai.OpenAI(base_url=vllm_url, api_key="dummy")


def call_llm(client: openai.OpenAI, model: str, system: str, user: str,
             temperature: float = 1.0, max_tokens: int = 100) -> str:
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


def call_llm_multiturn(client: openai.OpenAI, model: str, messages: list,
                       temperature: float = 1.0, max_tokens: int = 100) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"


def parse_rating(response: str) -> int:
    """Extract 1-5 rating from response."""
    for ch in response.strip():
        if ch in "12345":
            return int(ch)
    return 3  # default to abstention


def predict_case(client, model, case_text, article, template, n_samples=10):
    """Run N predictions for a case-article pair."""
    article_title = ARTICLE_TITLES.get(article, f"Article {article}")
    prompt = template.format(case_text=case_text, article=article, article_title=article_title)

    ratings = []
    for _ in range(n_samples):
        resp = call_llm(client, model, SYSTEM_PROMPT, prompt)
        ratings.append(parse_rating(resp))

    return ratings


def majority_vote(ratings: list) -> tuple:
    """Returns (prediction, is_abstention). 1-2=violation, 4-5=no_violation, 3=abstention."""
    thresholded = []
    for r in ratings:
        if r <= 2:
            thresholded.append("violation")
        elif r >= 4:
            thresholded.append("no_violation")
        else:
            thresholded.append("abstention")

    counts = Counter(thresholded)
    top = counts.most_common(1)[0]
    return top[0], top[0] == "abstention"


# ── Experiments ───────────────────────────────────────────────────────────

def run_baseline(client, model, cases, n_samples, parent_run_id):
    """RQ0: Baseline evaluation on full verdict-free text."""
    results = []

    with mlflow.start_run(run_name="baseline", nested=True):
        mlflow.log_param("stage", "baseline")
        mlflow.log_param("n_cases", len(cases))
        mlflow.log_param("n_samples", n_samples)

        for i, case in enumerate(cases):
            text = case.get("full_case_text_no_verdict", case.get("full_case_text", ""))
            ratings = predict_case(client, model, text, case["article"], PREDICTIVE_TEMPLATE, n_samples)
            pred, abstained = majority_vote(ratings)
            accurate = pred == case["violation_label"]

            results.append({
                "case_name": case["case_name"],
                "article": case["article"],
                "violation_label": case["violation_label"],
                "prediction": pred,
                "accurate": accurate,
                "abstained": abstained,
                "ratings": ratings,
                "avg_rating": sum(ratings) / len(ratings),
            })
            print(f"\r  Baseline: {i+1}/{len(cases)} | acc={sum(r['accurate'] for r in results)/len(results):.2f}", end="", flush=True)

        accuracy = sum(r["accurate"] for r in results) / len(results)
        abstention_rate = sum(r["abstained"] for r in results) / len(results)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("abstention_rate", abstention_rate)
        mlflow.log_dict(results, "baseline_results.json")
        print(f"\n  Baseline: acc={accuracy:.3f}, abstention={abstention_rate:.3f}")

    return results


def run_summarization(client, model, cases, n_samples, baseline_results):
    """RQ1: Summarization effect."""
    with mlflow.start_run(run_name="rq1_summarization", nested=True):
        mlflow.log_param("stage", "rq1_summarization")

        # Generate 3 summary versions
        summaries = {}
        for i, case in enumerate(cases):
            text = case.get("full_case_text_no_verdict", case.get("full_case_text", ""))
            case_key = case["item_id"] + "_" + case["article"]
            summaries[case_key] = []

            for v in range(3):
                prompt = SUMMARY_TEMPLATE.format(case_name=case["case_name"], full_text=text[:30000])
                summary = call_llm(client, model, "", prompt, temperature=1.0, max_tokens=1000)
                summaries[case_key].append(summary)

            print(f"\r  Summaries: {i+1}/{len(cases)}", end="", flush=True)

        print()
        mlflow.log_dict(summaries, "summaries.json")

        # Evaluate on each summary
        summary_results = []
        for i, case in enumerate(cases):
            case_key = case["item_id"] + "_" + case["article"]
            for v in range(3):
                summary_text = summaries[case_key][v]
                ratings = predict_case(client, model, summary_text, case["article"], PREDICTIVE_TEMPLATE, n_samples)
                pred, abstained = majority_vote(ratings)

                # Find baseline prediction for alignment
                baseline_pred = None
                for br in baseline_results:
                    if br["case_name"] == case["case_name"] and br["article"] == case["article"]:
                        baseline_pred = br["prediction"]
                        break

                summary_results.append({
                    "case_name": case["case_name"],
                    "article": case["article"],
                    "violation_label": case["violation_label"],
                    "summary_version": v,
                    "prediction": pred,
                    "accurate": pred == case["violation_label"],
                    "aligned": pred == baseline_pred,
                    "ratings": ratings,
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
    """RQ2: Decision-question framing effect."""
    with mlflow.start_run(run_name="rq2_framing", nested=True):
        mlflow.log_param("stage", "rq2_framing")

        framings = {
            "predictive": PREDICTIVE_TEMPLATE,
            "normative": NORMATIVE_TEMPLATE,
            "factual": FACTUAL_TEMPLATE,
        }

        framing_results = []
        for i, case in enumerate(cases):
            case_key = case["item_id"] + "_" + case["article"]
            # Use first summary version
            summary_text = summaries.get(case_key, [""])[0]
            if not summary_text:
                summary_text = case.get("full_case_text_no_verdict", case.get("full_case_text", ""))[:5000]

            baseline_pred = None
            for br in baseline_results:
                if br["case_name"] == case["case_name"] and br["article"] == case["article"]:
                    baseline_pred = br["prediction"]
                    break

            for framing_name, template in framings.items():
                ratings = predict_case(client, model, summary_text, case["article"], template, n_samples)
                pred, abstained = majority_vote(ratings)

                framing_results.append({
                    "case_name": case["case_name"],
                    "article": case["article"],
                    "violation_label": case["violation_label"],
                    "framing": framing_name,
                    "prediction": pred,
                    "accurate": pred == case["violation_label"],
                    "aligned_with_baseline": pred == baseline_pred,
                    "ratings": ratings,
                })
            print(f"\r  Framing: {i+1}/{len(cases)}", end="", flush=True)

        for framing_name in framings:
            fr = [r for r in framing_results if r["framing"] == framing_name]
            acc = sum(r["accurate"] for r in fr) / len(fr)
            align = sum(r["aligned_with_baseline"] for r in fr) / len(fr)
            mlflow.log_metric(f"accuracy_{framing_name}", acc)
            mlflow.log_metric(f"alignment_{framing_name}", align)

        mlflow.log_dict(framing_results, "rq2_results.json")
        print(f"\n  RQ2 done")

    return framing_results


def run_reconsideration(client, model, cases, n_samples, baseline_results):
    """RQ3: Reconsideration effect ('Are you sure?')."""
    with mlflow.start_run(run_name="rq3_reconsideration", nested=True):
        mlflow.log_param("stage", "rq3_reconsideration")

        recon_results = []
        for i, case in enumerate(cases):
            text = case.get("full_case_text_no_verdict", case.get("full_case_text", ""))
            article_title = ARTICLE_TITLES.get(case["article"], f"Article {case['article']}")
            prompt = PREDICTIVE_TEMPLATE.format(case_text=text, article=case["article"], article_title=article_title)

            original_ratings = []
            challenged_ratings = []

            for _ in range(n_samples):
                # Turn 1: original prediction
                messages = [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ]
                resp1 = call_llm_multiturn(client, model, messages)
                r1 = parse_rating(resp1)
                original_ratings.append(r1)

                # Turn 2: reconsideration
                messages.append({"role": "assistant", "content": resp1})
                messages.append({"role": "user", "content": RECONSIDERATION_PROMPT})
                resp2 = call_llm_multiturn(client, model, messages)
                r2 = parse_rating(resp2)
                challenged_ratings.append(r2)

            orig_pred, _ = majority_vote(original_ratings)
            chal_pred, _ = majority_vote(challenged_ratings)

            baseline_pred = None
            for br in baseline_results:
                if br["case_name"] == case["case_name"] and br["article"] == case["article"]:
                    baseline_pred = br["prediction"]
                    break

            # Categorize changes
            changes = sum(1 for o, c in zip(original_ratings, challenged_ratings) if o != c)

            recon_results.append({
                "case_name": case["case_name"],
                "article": case["article"],
                "violation_label": case["violation_label"],
                "original_prediction": orig_pred,
                "challenged_prediction": chal_pred,
                "original_accurate": orig_pred == case["violation_label"],
                "challenged_accurate": chal_pred == case["violation_label"],
                "prediction_changed": orig_pred != chal_pred,
                "individual_changes": changes,
                "change_rate": changes / n_samples,
                "original_ratings": original_ratings,
                "challenged_ratings": challenged_ratings,
            })
            print(f"\r  Reconsideration: {i+1}/{len(cases)}", end="", flush=True)

        orig_acc = sum(r["original_accurate"] for r in recon_results) / len(recon_results)
        chal_acc = sum(r["challenged_accurate"] for r in recon_results) / len(recon_results)
        change_rate = sum(r["prediction_changed"] for r in recon_results) / len(recon_results)
        avg_individual_change = sum(r["change_rate"] for r in recon_results) / len(recon_results)

        mlflow.log_metric("original_accuracy", orig_acc)
        mlflow.log_metric("challenged_accuracy", chal_acc)
        mlflow.log_metric("prediction_change_rate", change_rate)
        mlflow.log_metric("avg_individual_change_rate", avg_individual_change)
        mlflow.log_dict(recon_results, "rq3_results.json")
        print(f"\n  RQ3: orig_acc={orig_acc:.3f}, chal_acc={chal_acc:.3f}, changed={change_rate:.3f}")

    return recon_results


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True, help="Path to cases JSON")
    parser.add_argument("--vllm-url", default="http://localhost:8000/v1")
    parser.add_argument("--model", default="Qwen/Qwen3-32B")
    parser.add_argument("--mlflow-url", default="http://35.84.222.41:5000")
    parser.add_argument("--experiment-id", default="23")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--stages", default="all", help="Comma-separated: baseline,rq1,rq2,rq3 or all")
    args = parser.parse_args()

    cases = json.load(open(args.cases))
    print(f"Loaded {len(cases)} cases")

    client = create_client(args.vllm_url)

    os.environ["MLFLOW_TRACKING_URI"] = args.mlflow_url
    os.environ["MLFLOW_TRACKING_USERNAME"] = "admin"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = "mNW73dLlYkel1ue7bmQPcIjxeVpSln-k"

    mlflow.set_tracking_uri(args.mlflow_url)
    mlflow.set_experiment(experiment_id=args.experiment_id)

    stages = args.stages.split(",") if args.stages != "all" else ["baseline", "rq1", "rq2", "rq3"]

    model_short = args.model.split("/")[-1]

    with mlflow.start_run(run_name=f"{model_short}_{len(cases)}cases") as parent_run:
        mlflow.log_param("model", args.model)
        mlflow.log_param("n_cases", len(cases))
        mlflow.log_param("n_samples", args.samples)
        mlflow.log_param("stages", ",".join(stages))
        mlflow.log_param("vllm_url", args.vllm_url)

        violation_count = sum(1 for c in cases if c["violation_label"] == "violation")
        mlflow.log_param("violation_rate", f"{violation_count}/{len(cases)}")

        start = time.time()

        baseline_results = None
        summaries = None

        if "baseline" in stages:
            baseline_results = run_baseline(client, args.model, cases, args.samples, parent_run.info.run_id)

        if "rq1" in stages:
            if baseline_results is None:
                print("ERROR: baseline required before rq1")
                return
            _, summaries = run_summarization(client, args.model, cases, args.samples, baseline_results)

        if "rq2" in stages:
            if baseline_results is None or summaries is None:
                print("ERROR: baseline + rq1 required before rq2")
                return
            run_framing(client, args.model, cases, args.samples, summaries, baseline_results)

        if "rq3" in stages:
            if baseline_results is None:
                print("ERROR: baseline required before rq3")
                return
            run_reconsideration(client, args.model, cases, args.samples, baseline_results)

        elapsed = time.time() - start
        mlflow.log_metric("total_time_seconds", elapsed)
        print(f"\nDone in {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
