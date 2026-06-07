#!/usr/bin/env python3
"""Thin wrapper: run perturbation study against a vLLM endpoint via OpenAI-compatible API."""
import argparse, json, os, sys, time
from collections import Counter
import mlflow, openai

# Reuse all logic from bedrock runner
sys.path.insert(0, os.path.dirname(__file__))
from run_perturbation_bedrock import (
    SYSTEM_PROMPT, ARTICLE_TITLES, PREDICTIVE_TEMPLATE, NORMATIVE_TEMPLATE,
    FACTUAL_TEMPLATE, SUMMARY_TEMPLATE, RECONSIDERATION_PROMPT,
    parse_rating, majority_vote,
)


def call_vllm(client, model, system, user, temperature=1.0, max_tokens=100):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=temperature, max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"


def call_vllm_multiturn(client, model, system, messages, temperature=1.0, max_tokens=100):
    try:
        msgs = [{"role": "system", "content": system}] + messages
        resp = client.chat.completions.create(
            model=model, messages=msgs, temperature=temperature, max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"ERROR: {e}"


def run_baseline(client, model, cases, n_samples):
    results = []
    with mlflow.start_run(run_name="baseline", nested=True):
        mlflow.log_param("stage", "baseline")
        mlflow.log_param("n_cases", len(cases))
        for i, case in enumerate(cases):
            text = case.get("full_case_text_no_verdict", case.get("full_case_text", ""))[:50000]
            at = ARTICLE_TITLES.get(case["article"], f"Article {case['article']}")
            prompt = PREDICTIVE_TEMPLATE.format(case_text=text, article=case["article"], article_title=at)
            ratings = [parse_rating(call_vllm(client, model, SYSTEM_PROMPT, prompt)) for _ in range(n_samples)]
            pred, abstained = majority_vote(ratings)
            results.append({"case_name": case["case_name"], "article": case["article"],
                "violation_label": case["violation_label"], "prediction": pred,
                "accurate": pred == case["violation_label"], "abstained": abstained,
                "ratings": ratings, "avg_rating": sum(ratings)/len(ratings)})
            print(f"\r  Baseline: {i+1}/{len(cases)} | acc={sum(r['accurate'] for r in results)/len(results):.2f}", end="", flush=True)
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
            summaries[key] = [call_vllm(client, model, "", SUMMARY_TEMPLATE.format(case_name=case["case_name"], full_text=text), temperature=1.0, max_tokens=1000) for _ in range(3)]
            print(f"\r  Summaries: {i+1}/{len(cases)}", end="", flush=True)
        print()
        mlflow.log_dict(summaries, "summaries.json")
        results = []
        for i, case in enumerate(cases):
            key = case["item_id"] + "_" + case["article"]
            bp = next((r["prediction"] for r in baseline_results if r["case_name"]==case["case_name"] and r["article"]==case["article"]), None)
            for v in range(3):
                at = ARTICLE_TITLES.get(case["article"], f"Article {case['article']}")
                prompt = PREDICTIVE_TEMPLATE.format(case_text=summaries[key][v], article=case["article"], article_title=at)
                ratings = [parse_rating(call_vllm(client, model, SYSTEM_PROMPT, prompt)) for _ in range(n_samples)]
                pred, _ = majority_vote(ratings)
                results.append({"case_name": case["case_name"], "article": case["article"],
                    "violation_label": case["violation_label"], "summary_version": v,
                    "prediction": pred, "accurate": pred==case["violation_label"], "aligned": pred==bp, "ratings": ratings})
            print(f"\r  Summary eval: {i+1}/{len(cases)}", end="", flush=True)
        accuracy = sum(r["accurate"] for r in results) / len(results)
        alignment = sum(r["aligned"] for r in results) / len(results)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("alignment_rate", alignment)
        mlflow.log_dict(results, "rq1_results.json")
        print(f"\n  RQ1: acc={accuracy:.3f}, alignment={alignment:.3f}")
    return results, summaries


def run_framing(client, model, cases, n_samples, summaries, baseline_results):
    with mlflow.start_run(run_name="rq2_framing", nested=True):
        mlflow.log_param("stage", "rq2_framing")
        framings = {"predictive": PREDICTIVE_TEMPLATE, "normative": NORMATIVE_TEMPLATE, "factual": FACTUAL_TEMPLATE}
        results = []
        for i, case in enumerate(cases):
            key = case["item_id"] + "_" + case["article"]
            text = summaries.get(key, [""])[0] or case.get("full_case_text_no_verdict", "")[:5000]
            bp = next((r["prediction"] for r in baseline_results if r["case_name"]==case["case_name"] and r["article"]==case["article"]), None)
            case_label = f"{case['case_name'][:40]}|Art{case['article']}"

            with mlflow.start_span(name=f"framing|{case_label}") as case_span:
                case_span.set_attributes({"case_name": case["case_name"], "article": case["article"],
                    "violation_label": case["violation_label"], "baseline_prediction": str(bp)})
                framing_preds = {}
                for fname, tmpl in framings.items():
                    at = ARTICLE_TITLES.get(case["article"], f"Article {case['article']}")
                    prompt = tmpl.format(case_text=text, article=case["article"], article_title=at)
                    with mlflow.start_span(name=f"{fname}") as fs:
                        fs.set_inputs({"framing": fname, "prompt_snippet": prompt[:200] + "..."})
                        ratings = [parse_rating(call_vllm(client, model, SYSTEM_PROMPT, prompt)) for _ in range(n_samples)]
                        pred, _ = majority_vote(ratings)
                        framing_preds[fname] = pred
                        fs.set_outputs({"prediction": pred, "ratings": ratings,
                            "accurate": pred==case["violation_label"], "aligned": pred==bp})
                    results.append({"case_name": case["case_name"], "article": case["article"],
                        "violation_label": case["violation_label"], "framing": fname,
                        "prediction": pred, "accurate": pred==case["violation_label"], "aligned_with_baseline": pred==bp, "ratings": ratings})
                case_span.set_outputs({"predictions": framing_preds, "any_disagreement": len(set(framing_preds.values()))>1})
            print(f"\r  Framing: {i+1}/{len(cases)}", end="", flush=True)
        for fn in framings:
            fr = [r for r in results if r["framing"]==fn]
            mlflow.log_metric(f"accuracy_{fn}", sum(r["accurate"] for r in fr)/len(fr))
            mlflow.log_metric(f"alignment_{fn}", sum(r["aligned_with_baseline"] for r in fr)/len(fr))
        mlflow.log_dict(results, "rq2_results.json")
        print(f"\n  RQ2 done")
    return results


def run_reconsideration(client, model, cases, n_samples, baseline_results):
    with mlflow.start_run(run_name="rq3_reconsideration", nested=True):
        mlflow.log_param("stage", "rq3_reconsideration")
        results = []
        for i, case in enumerate(cases):
            text = case.get("full_case_text_no_verdict", case.get("full_case_text", ""))[:50000]
            at = ARTICLE_TITLES.get(case["article"], f"Article {case['article']}")
            prompt = PREDICTIVE_TEMPLATE.format(case_text=text, article=case["article"], article_title=at)
            case_label = f"{case['case_name'][:40]}|Art{case['article']}"
            orig, chal = [], []

            with mlflow.start_span(name=f"recon|{case_label}") as case_span:
                case_span.set_attributes({"case_name": case["case_name"], "article": case["article"],
                    "violation_label": case["violation_label"]})
                for s in range(n_samples):
                    with mlflow.start_span(name=f"sample_{s}") as ss:
                        r1_text = call_vllm(client, model, SYSTEM_PROMPT, prompt)
                        r1 = parse_rating(r1_text)
                        orig.append(r1)
                        msgs = [{"role":"user","content":prompt},{"role":"assistant","content":r1_text},{"role":"user","content":RECONSIDERATION_PROMPT}]
                        r2_text = call_vllm_multiturn(client, model, SYSTEM_PROMPT, msgs)
                        r2 = parse_rating(r2_text)
                        chal.append(r2)
                        direction = "toward_violation" if r2 < r1 else ("away_from_violation" if r2 > r1 else "unchanged")
                        ss.set_inputs({"original_response": r1_text, "challenge": RECONSIDERATION_PROMPT})
                        ss.set_outputs({"challenged_response": r2_text, "original_rating": r1,
                            "challenged_rating": r2, "changed": r1!=r2, "direction": direction})
                op, _ = majority_vote(orig)
                cp, _ = majority_vote(chal)
                changes = sum(1 for o,c in zip(orig,chal) if o!=c)
                case_span.set_outputs({"original_prediction": op, "challenged_prediction": cp,
                    "prediction_changed": op!=cp, "individual_change_rate": changes/n_samples})
            results.append({"case_name": case["case_name"], "article": case["article"],
                "violation_label": case["violation_label"], "original_prediction": op,
                "challenged_prediction": cp, "original_accurate": op==case["violation_label"],
                "challenged_accurate": cp==case["violation_label"], "prediction_changed": op!=cp,
                "change_rate": changes/n_samples, "original_ratings": orig, "challenged_ratings": chal})
            print(f"\r  Reconsideration: {i+1}/{len(cases)}", end="", flush=True)
        oa = sum(r["original_accurate"] for r in results)/len(results)
        ca = sum(r["challenged_accurate"] for r in results)/len(results)
        cr = sum(r["prediction_changed"] for r in results)/len(results)
        mlflow.log_metric("original_accuracy", oa)
        mlflow.log_metric("challenged_accuracy", ca)
        mlflow.log_metric("prediction_change_rate", cr)
        mlflow.log_dict(results, "rq3_results.json")
        print(f"\n  RQ3: orig_acc={oa:.3f}, chal_acc={ca:.3f}, changed={cr:.3f}")
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--vllm-url", default="http://10.88.0.3:8000/v1")
    parser.add_argument("--model", default="meta-llama/Llama-3.3-70B-Instruct")
    parser.add_argument("--mlflow-url", default="http://10.88.0.4:5000")
    parser.add_argument("--experiment-id", default="23")
    parser.add_argument("--samples", type=int, default=10)
    parser.add_argument("--stages", default="all")
    args = parser.parse_args()

    cases = json.load(open(args.cases))
    print(f"Loaded {len(cases)} cases")

    client = openai.OpenAI(base_url=args.vllm_url, api_key="dummy")
    mlflow.set_tracking_uri(args.mlflow_url)
    mlflow.set_experiment(experiment_id=args.experiment_id)

    stages = args.stages.split(",") if args.stages != "all" else ["baseline","rq1","rq2","rq3"]
    model_short = args.model.split("/")[-1]

    with mlflow.start_run(run_name=f"{model_short}_{len(cases)}cases"):
        mlflow.log_param("model", args.model)
        mlflow.log_param("n_cases", len(cases))
        mlflow.log_param("n_samples", args.samples)
        mlflow.log_param("stages", ",".join(stages))
        mlflow.log_param("backend", "vllm-brev")
        start = time.time()
        bl = sm = None
        if "baseline" in stages: bl = run_baseline(client, args.model, cases, args.samples)
        if "rq1" in stages and bl: _, sm = run_summarization(client, args.model, cases, args.samples, bl)
        if "rq2" in stages and bl and sm: run_framing(client, args.model, cases, args.samples, sm, bl)
        if "rq3" in stages and bl: run_reconsideration(client, args.model, cases, args.samples, bl)
        elapsed = time.time() - start
        mlflow.log_metric("total_time_seconds", elapsed)
        print(f"\nDone in {elapsed/60:.1f} minutes")


if __name__ == "__main__":
    main()
