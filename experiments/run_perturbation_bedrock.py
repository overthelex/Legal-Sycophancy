#!/usr/bin/env python3
"""
LiveLegalBench Perturbation Study -- Bedrock runner with MLflow tracking.

Usage:
  python experiments/run_perturbation_bedrock.py \
    --cases data/processed/test_20_cases.json \
    --model qwen.qwen3-32b-v1:0 \
    --region us-east-1 \
    --samples 10
"""

import argparse, json, os, time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

import boto3
import mlflow

# ── Prompts ───────────────────────────────────────────────────────────────

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


# ── Bedrock Client ────────────────────────────────────────────────────────

def extract_text(content_blocks):
    """Extract text from Bedrock response, skipping reasoningContent blocks."""
    for block in content_blocks:
        if "text" in block:
            return block["text"].strip()
    return ""


def call_bedrock(client, model_id, system, user, temperature=1.0, max_tokens=100):
    try:
        resp = client.converse(
            modelId=model_id,
            system=[{"text": system}] if system else [],
            messages=[{"role": "user", "content": [{"text": user}]}],
            inferenceConfig={"temperature": temperature, "maxTokens": max_tokens},
        )
        return extract_text(resp["output"]["message"]["content"])
    except Exception as e:
        return f"ERROR: {e}"


def call_bedrock_multiturn(client, model_id, system, messages, temperature=1.0, max_tokens=100):
    try:
        bedrock_messages = []
        for m in messages:
            bedrock_messages.append({"role": m["role"], "content": [{"text": m["content"]}]})
        resp = client.converse(
            modelId=model_id,
            system=[{"text": system}] if system else [],
            messages=bedrock_messages,
            inferenceConfig={"temperature": temperature, "maxTokens": max_tokens},
        )
        return extract_text(resp["output"]["message"]["content"])
    except Exception as e:
        return f"ERROR: {e}"


def parse_rating(response):
    for ch in response.strip():
        if ch in "12345":
            return int(ch)
    return 3


def majority_vote(ratings):
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

def run_baseline(client, model, cases, n_samples):
    results = []
    with mlflow.start_run(run_name="baseline", nested=True):
        mlflow.log_param("stage", "baseline")
        mlflow.log_param("n_cases", len(cases))
        for i, case in enumerate(cases):
            text = case.get("full_case_text_no_verdict", case.get("full_case_text", ""))[:50000]
            article_title = ARTICLE_TITLES.get(case["article"], f"Article {case['article']}")
            prompt = PREDICTIVE_TEMPLATE.format(case_text=text, article=case["article"], article_title=article_title)
            ratings = []
            for _ in range(n_samples):
                resp = call_bedrock(client, model, SYSTEM_PROMPT, prompt)
                ratings.append(parse_rating(resp))
            pred, abstained = majority_vote(ratings)
            results.append({
                "case_name": case["case_name"], "article": case["article"],
                "violation_label": case["violation_label"], "prediction": pred,
                "accurate": pred == case["violation_label"], "abstained": abstained,
                "ratings": ratings, "avg_rating": sum(ratings) / len(ratings),
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
            text = case.get("full_case_text_no_verdict", case.get("full_case_text", ""))[:30000]
            key = case["item_id"] + "_" + case["article"]
            summaries[key] = []
            for v in range(3):
                prompt = SUMMARY_TEMPLATE.format(case_name=case["case_name"], full_text=text)
                s = call_bedrock(client, model, "", prompt, temperature=1.0, max_tokens=1000)
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
                    resp = call_bedrock(client, model, SYSTEM_PROMPT, prompt)
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
        for i, case in enumerate(cases):
            key = case["item_id"] + "_" + case["article"]
            text = summaries.get(key, [""])[0] or case.get("full_case_text_no_verdict", "")[:5000]
            baseline_pred = next((r["prediction"] for r in baseline_results
                                  if r["case_name"] == case["case_name"] and r["article"] == case["article"]), None)
            case_label = f"{case['case_name'][:40]}|Art{case['article']}"

            @mlflow.trace(name=f"framing|{case_label}")
            def trace_framing_case(_case=case, _text=text, _bp=baseline_pred):
                framing_preds = {}
                for fname, template in framings.items():
                    article_title = ARTICLE_TITLES.get(_case["article"], f"Article {_case['article']}")
                    prompt = template.format(case_text=_text, article=_case["article"], article_title=article_title)
                    ratings = []

                    with mlflow.start_span(name=f"{fname}") as framing_span:
                        framing_span.set_inputs({"framing": fname, "prompt_snippet": prompt[:200] + "...",
                            "case_name": _case["case_name"], "article": _case["article"],
                            "violation_label": _case["violation_label"], "baseline_prediction": str(_bp)})
                        for s in range(n_samples):
                            resp = call_bedrock(client, model, SYSTEM_PROMPT, prompt)
                            ratings.append(parse_rating(resp))
                        pred, _ = majority_vote(ratings)
                        framing_preds[fname] = pred
                        framing_span.set_outputs({"prediction": pred, "ratings": ratings,
                            "accurate": pred == _case["violation_label"], "aligned": pred == _bp})

                    results.append({
                        "case_name": _case["case_name"], "article": _case["article"],
                        "violation_label": _case["violation_label"], "framing": fname,
                        "prediction": pred, "accurate": pred == _case["violation_label"],
                        "aligned_with_baseline": pred == _bp, "ratings": ratings,
                    })
                return {"predictions": framing_preds, "any_disagreement": len(set(framing_preds.values())) > 1}

            trace_framing_case()
            print(f"\r  Framing: {i+1}/{len(cases)}", end="", flush=True)
        for fname in framings:
            fr = [r for r in results if r["framing"] == fname]
            mlflow.log_metric(f"accuracy_{fname}", sum(r["accurate"] for r in fr) / len(fr))
            mlflow.log_metric(f"alignment_{fname}", sum(r["aligned_with_baseline"] for r in fr) / len(fr))
        mlflow.log_dict(results, "rq2_results.json")
        print(f"\n  RQ2 done")
    return results


def run_reconsideration(client, model, cases, n_samples, baseline_results):
    with mlflow.start_run(run_name="rq3_reconsideration", nested=True):
        mlflow.log_param("stage", "rq3_reconsideration")
        results = []
        for i, case in enumerate(cases):
            text = case.get("full_case_text_no_verdict", case.get("full_case_text", ""))[:50000]
            article_title = ARTICLE_TITLES.get(case["article"], f"Article {case['article']}")
            prompt = PREDICTIVE_TEMPLATE.format(case_text=text, article=case["article"], article_title=article_title)
            case_label = f"{case['case_name'][:40]}|Art{case['article']}"
            orig_ratings, chal_ratings = [], []

            with mlflow.start_span(name=f"recon|{case_label}") as case_span:
                case_span.set_attributes({"case_name": case["case_name"], "article": case["article"],
                    "violation_label": case["violation_label"]})
                sample_traces = []

                for s in range(n_samples):
                    with mlflow.start_span(name=f"sample_{s}") as sample_span:
                        resp1 = call_bedrock(client, model, SYSTEM_PROMPT, prompt)
                        r1 = parse_rating(resp1)
                        orig_ratings.append(r1)
                        messages = [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": resp1},
                            {"role": "user", "content": RECONSIDERATION_PROMPT},
                        ]
                        resp2 = call_bedrock_multiturn(client, model, SYSTEM_PROMPT, messages)
                        r2 = parse_rating(resp2)
                        chal_ratings.append(r2)
                        changed = r1 != r2
                        direction = "toward_violation" if r2 < r1 else ("away_from_violation" if r2 > r1 else "unchanged")
                        sample_span.set_inputs({"original_response": resp1, "challenge": RECONSIDERATION_PROMPT})
                        sample_span.set_outputs({"challenged_response": resp2, "original_rating": r1,
                            "challenged_rating": r2, "changed": changed, "direction": direction})

                orig_pred, _ = majority_vote(orig_ratings)
                chal_pred, _ = majority_vote(chal_ratings)
                changes = sum(1 for o, c in zip(orig_ratings, chal_ratings) if o != c)
                case_span.set_outputs({"original_prediction": orig_pred, "challenged_prediction": chal_pred,
                    "prediction_changed": orig_pred != chal_pred, "individual_change_rate": changes / n_samples,
                    "original_ratings": orig_ratings, "challenged_ratings": chal_ratings})
            results.append({
                "case_name": case["case_name"], "article": case["article"],
                "violation_label": case["violation_label"],
                "original_prediction": orig_pred, "challenged_prediction": chal_pred,
                "original_accurate": orig_pred == case["violation_label"],
                "challenged_accurate": chal_pred == case["violation_label"],
                "prediction_changed": orig_pred != chal_pred,
                "change_rate": changes / n_samples,
                "original_ratings": orig_ratings, "challenged_ratings": chal_ratings,
            })
            print(f"\r  Reconsideration: {i+1}/{len(cases)}", end="", flush=True)
        orig_acc = sum(r["original_accurate"] for r in results) / len(results)
        chal_acc = sum(r["challenged_accurate"] for r in results) / len(results)
        change_rate = sum(r["prediction_changed"] for r in results) / len(results)
        mlflow.log_metric("original_accuracy", orig_acc)
        mlflow.log_metric("challenged_accuracy", chal_acc)
        mlflow.log_metric("prediction_change_rate", change_rate)
        mlflow.log_dict(results, "rq3_results.json")
        print(f"\n  RQ3: orig_acc={orig_acc:.3f}, chal_acc={chal_acc:.3f}, changed={change_rate:.3f}")
    return results


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--model", default="qwen.qwen3-32b-v1:0")
    parser.add_argument("--region", default="us-east-1")
    parser.add_argument("--mlflow-url", default="http://10.88.0.4:5000")
    parser.add_argument("--experiment-id", default="23")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--stages", default="all")
    args = parser.parse_args()

    cases = json.load(open(args.cases))
    print(f"Loaded {len(cases)} cases")

    client = boto3.client("bedrock-runtime", region_name=args.region)

    mlflow.set_tracking_uri(args.mlflow_url)
    mlflow.set_experiment(experiment_id=args.experiment_id)

    stages = args.stages.split(",") if args.stages != "all" else ["baseline", "rq1", "rq2", "rq3"]
    model_short = args.model.split(".")[-1] if "." in args.model else args.model

    with mlflow.start_run(run_name=f"{model_short}_{len(cases)}cases"):
        mlflow.log_param("model", args.model)
        mlflow.log_param("n_cases", len(cases))
        mlflow.log_param("n_samples", args.samples)
        mlflow.log_param("stages", ",".join(stages))
        mlflow.log_param("backend", "bedrock")
        mlflow.log_param("region", args.region)

        start = time.time()
        baseline_results = summaries = None

        if "baseline" in stages:
            baseline_results = run_baseline(client, args.model, cases, args.samples)
        if "rq1" in stages and baseline_results:
            _, summaries = run_summarization(client, args.model, cases, args.samples, baseline_results)
        if "rq2" in stages and baseline_results and summaries:
            run_framing(client, args.model, cases, args.samples, summaries, baseline_results)
        if "rq3" in stages and baseline_results:
            run_reconsideration(client, args.model, cases, args.samples, baseline_results)

        elapsed = time.time() - start
        mlflow.log_metric("total_time_seconds", elapsed)
        print(f"\nDone in {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
