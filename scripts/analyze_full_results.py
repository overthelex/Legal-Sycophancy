#!/usr/bin/env python3
"""
Analyze full-scale perturbation experiment results.

Loads all experiment results from data/experiments/full_scale/,
computes per-model, per-country, per-article metrics, generates
summary tables, and outputs CSVs for LaTeX import.

Usage:
  python scripts/analyze_full_results.py
  python scripts/analyze_full_results.py --pilot-dir data/experiments/pilot/
  python scripts/analyze_full_results.py --output-dir results/tables/
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import binom

# ---------------------------------------------------------------------------
# Model registry (mirrors run_full_experiment.py)
# ---------------------------------------------------------------------------

MODELS = {
    "opus": "Claude Opus 4.8",
    "sonnet": "Claude Sonnet 4.6",
    "nova": "Amazon Nova Pro",
    "qwen": "Qwen3-32B",
    "llama": "Llama 3.3 70B",
}

ARTICLE_TITLES = {
    "1": "Protection of property",
    "2": "Right to life",
    "3": "Prohibition of torture",
    "5": "Right to liberty and security",
    "6": "Right to a fair trial",
    "7": "No punishment without law",
    "8": "Right to respect for private and family life",
    "9": "Freedom of thought, conscience and religion",
    "10": "Freedom of expression",
    "11": "Freedom of assembly and association",
    "13": "Right to an effective remedy",
    "14": "Prohibition of discrimination",
    "34": "Individual applications",
    "41": "Just satisfaction",
    "P1-1": "Protection of property",
}

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RESULTS_DIR = REPO_ROOT / "data" / "experiments" / "full_scale"


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_results(results_dir: Path) -> dict:
    """
    Load all available results.

    Returns:
        {model_key: {rq: list_of_result_dicts}}
    """
    all_results = {}
    for model_key in MODELS:
        model_dir = results_dir / model_key
        if not model_dir.exists():
            continue
        model_results = {}
        for rq in ["baseline", "rq1", "rq2", "rq3"]:
            rq_file = model_dir / f"{rq}_results.json"
            if rq_file.exists():
                with open(rq_file) as f:
                    model_results[rq] = json.load(f)
        if model_results:
            all_results[model_key] = model_results
    return all_results


def load_cases(cases_path: Path) -> list:
    """Load stratified sample for metadata."""
    with open(cases_path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Baseline analysis
# ---------------------------------------------------------------------------

def analyze_baseline(results: list) -> dict:
    """Compute baseline metrics."""
    n = len(results)
    if n == 0:
        return {}

    accurate = sum(1 for r in results if r["accurate"])
    abstained = sum(1 for r in results if r.get("abstained", False))
    violations = [r for r in results if r["violation_label"] == "violation"]
    no_violations = [r for r in results if r["violation_label"] == "no_violation"]

    metrics = {
        "n_cases": n,
        "accuracy": accurate / n,
        "abstention_rate": abstained / n,
        "avg_rating": np.mean([r["avg_rating"] for r in results]),
    }

    if violations:
        metrics["accuracy_violation"] = sum(
            1 for r in violations if r["accurate"]
        ) / len(violations)
    if no_violations:
        metrics["accuracy_no_violation"] = sum(
            1 for r in no_violations if r["accurate"]
        ) / len(no_violations)

    return metrics


def baseline_by_country(results: list) -> pd.DataFrame:
    """Break down baseline metrics by respondent country."""
    by_country = defaultdict(list)
    for r in results:
        country = r.get("respondent", "UNKNOWN")
        by_country[country].append(r)

    rows = []
    for country in sorted(by_country.keys()):
        recs = by_country[country]
        n = len(recs)
        acc = sum(1 for r in recs if r["accurate"]) / n if n else 0
        abst = sum(1 for r in recs if r.get("abstained", False)) / n if n else 0
        avg_r = np.mean([r["avg_rating"] for r in recs])
        rows.append({
            "country": country,
            "n_cases": n,
            "accuracy": acc,
            "abstention_rate": abst,
            "avg_rating": avg_r,
        })
    return pd.DataFrame(rows)


def baseline_by_article(results: list) -> pd.DataFrame:
    """Break down baseline metrics by article."""
    by_article = defaultdict(list)
    for r in results:
        by_article[r["article"]].append(r)

    rows = []
    for article in sorted(by_article.keys(), key=lambda x: (len(x), x)):
        recs = by_article[article]
        n = len(recs)
        acc = sum(1 for r in recs if r["accurate"]) / n if n else 0
        rows.append({
            "article": article,
            "article_title": ARTICLE_TITLES.get(article, f"Art. {article}"),
            "n_cases": n,
            "accuracy": acc,
            "avg_rating": np.mean([r["avg_rating"] for r in recs]),
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RQ1 analysis -- summarization
# ---------------------------------------------------------------------------

def analyze_rq1(results: list) -> dict:
    """Compute RQ1 (summarization) metrics."""
    n = len(results)
    if n == 0:
        return {}

    return {
        "n_evaluations": n,
        "accuracy": sum(1 for r in results if r["accurate"]) / n,
        "alignment_rate": sum(1 for r in results if r["aligned"]) / n,
    }


def rq1_by_version(results: list) -> pd.DataFrame:
    """Break down RQ1 metrics by summary version."""
    by_version = defaultdict(list)
    for r in results:
        by_version[r["summary_version"]].append(r)

    rows = []
    for version in sorted(by_version.keys()):
        recs = by_version[version]
        n = len(recs)
        rows.append({
            "summary_version": version,
            "n_evaluations": n,
            "accuracy": sum(1 for r in recs if r["accurate"]) / n,
            "alignment_rate": sum(1 for r in recs if r["aligned"]) / n,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RQ2 analysis -- framing
# ---------------------------------------------------------------------------

def analyze_rq2(results: list) -> dict:
    """Compute RQ2 (framing) metrics per framing type."""
    by_framing = defaultdict(list)
    for r in results:
        by_framing[r["framing"]].append(r)

    metrics = {}
    for fname, recs in by_framing.items():
        n = len(recs)
        metrics[fname] = {
            "n_evaluations": n,
            "accuracy": sum(1 for r in recs if r["accurate"]) / n,
            "alignment_rate": sum(
                1 for r in recs if r["aligned_with_baseline"]
            ) / n,
        }
    return metrics


def rq2_framing_table(results: list) -> pd.DataFrame:
    """Generate framing comparison table."""
    rows = []
    by_framing = defaultdict(list)
    for r in results:
        by_framing[r["framing"]].append(r)

    for fname in ["predictive", "normative", "factual"]:
        if fname not in by_framing:
            continue
        recs = by_framing[fname]
        n = len(recs)
        rows.append({
            "framing": fname,
            "n_evaluations": n,
            "accuracy": sum(1 for r in recs if r["accurate"]) / n,
            "alignment_with_baseline": sum(
                1 for r in recs if r["aligned_with_baseline"]
            ) / n,
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# RQ3 analysis -- reconsideration
# ---------------------------------------------------------------------------

def analyze_rq3(results: list) -> dict:
    """Compute RQ3 (reconsideration) metrics."""
    n = len(results)
    if n == 0:
        return {}

    return {
        "n_cases": n,
        "original_accuracy": sum(1 for r in results if r["original_accurate"]) / n,
        "challenged_accuracy": sum(1 for r in results if r["challenged_accurate"]) / n,
        "prediction_change_rate": sum(1 for r in results if r["prediction_changed"]) / n,
        "avg_individual_change_rate": np.mean([r["change_rate"] for r in results]),
    }


def rq3_direction_table(results: list) -> pd.DataFrame:
    """Analyze direction of changes in RQ3."""
    rows = []
    for r in results:
        orig_ratings = r.get("original_ratings", [])
        chal_ratings = r.get("challenged_ratings", [])
        for o, c in zip(orig_ratings, chal_ratings):
            if o != c:
                direction = "toward_violation" if c < o else "away_from_violation"
                rows.append({
                    "case_name": r["case_name"],
                    "article": r["article"],
                    "violation_label": r["violation_label"],
                    "original_rating": o,
                    "challenged_rating": c,
                    "direction": direction,
                    "magnitude": abs(c - o),
                })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# McNemar test (for statistical significance)
# ---------------------------------------------------------------------------

def mcnemar_test_exact(n01: int, n10: int) -> float:
    """Two-tailed exact McNemar test."""
    n = n01 + n10
    if n == 0:
        return 1.0
    p_value = 2 * min(
        binom.cdf(min(n01, n10), n, 0.5),
        1 - binom.cdf(min(n01, n10) - 1, n, 0.5),
    )
    return min(p_value, 1.0)


# ---------------------------------------------------------------------------
# Cross-model summary table
# ---------------------------------------------------------------------------

def build_summary_table(all_results: dict) -> pd.DataFrame:
    """Build the main cross-model summary table (Table 1 in the paper)."""
    rows = []
    for mk in MODELS:
        if mk not in all_results:
            continue
        model_name = MODELS[mk]
        mr = all_results[mk]

        row = {"model": model_name}

        # Baseline
        if "baseline" in mr:
            bl = analyze_baseline(mr["baseline"])
            row["baseline_accuracy"] = bl.get("accuracy")
            row["baseline_abstention"] = bl.get("abstention_rate")
            row["baseline_avg_rating"] = bl.get("avg_rating")
        else:
            row["baseline_accuracy"] = None
            row["baseline_abstention"] = None
            row["baseline_avg_rating"] = None

        # RQ1
        if "rq1" in mr:
            rq1 = analyze_rq1(mr["rq1"])
            row["rq1_accuracy"] = rq1.get("accuracy")
            row["rq1_alignment"] = rq1.get("alignment_rate")
        else:
            row["rq1_accuracy"] = None
            row["rq1_alignment"] = None

        # RQ2
        if "rq2" in mr:
            rq2 = analyze_rq2(mr["rq2"])
            for fname in ["predictive", "normative", "factual"]:
                if fname in rq2:
                    row[f"rq2_{fname}_accuracy"] = rq2[fname]["accuracy"]
                    row[f"rq2_{fname}_alignment"] = rq2[fname]["alignment_rate"]
        else:
            for fname in ["predictive", "normative", "factual"]:
                row[f"rq2_{fname}_accuracy"] = None
                row[f"rq2_{fname}_alignment"] = None

        # RQ3
        if "rq3" in mr:
            rq3 = analyze_rq3(mr["rq3"])
            row["rq3_orig_accuracy"] = rq3.get("original_accuracy")
            row["rq3_chal_accuracy"] = rq3.get("challenged_accuracy")
            row["rq3_change_rate"] = rq3.get("prediction_change_rate")
            row["rq3_avg_individual_change"] = rq3.get("avg_individual_change_rate")
        else:
            row["rq3_orig_accuracy"] = None
            row["rq3_chal_accuracy"] = None
            row["rq3_change_rate"] = None
            row["rq3_avg_individual_change"] = None

        rows.append(row)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Per-country cross-model table
# ---------------------------------------------------------------------------

def build_country_model_table(all_results: dict, cases: list) -> pd.DataFrame:
    """Build accuracy-by-country table across models (for baseline)."""
    # Attach country info to results
    case_country = {}
    for c in cases:
        key = (c["case_name"], c["article"])
        case_country[key] = c.get("respondent", "UNKNOWN")

    rows = []
    countries_seen = set()

    for mk in MODELS:
        if mk not in all_results or "baseline" not in all_results[mk]:
            continue
        by_country = defaultdict(list)
        for r in all_results[mk]["baseline"]:
            country = case_country.get((r["case_name"], r["article"]), "UNKNOWN")
            r["respondent"] = country
            by_country[country].append(r)
            countries_seen.add(country)

        for country, recs in by_country.items():
            n = len(recs)
            rows.append({
                "model": MODELS[mk],
                "country": country,
                "n_cases": n,
                "accuracy": sum(1 for r in recs if r["accurate"]) / n,
                "abstention_rate": sum(1 for r in recs if r.get("abstained", False)) / n,
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Pilot comparison
# ---------------------------------------------------------------------------

def compare_with_pilot(full_results: dict, pilot_dir: Path) -> Optional[pd.DataFrame]:
    """Compare full-scale with pilot results if available."""
    pilot_results = load_results(pilot_dir)
    if not pilot_results:
        return None

    rows = []
    for mk in MODELS:
        if mk not in full_results or mk not in pilot_results:
            continue

        row = {"model": MODELS[mk]}

        for stage in ["baseline"]:
            if stage in full_results[mk] and stage in pilot_results[mk]:
                full_bl = analyze_baseline(full_results[mk][stage])
                pilot_bl = analyze_baseline(pilot_results[mk][stage])
                row["pilot_accuracy"] = pilot_bl.get("accuracy")
                row["full_accuracy"] = full_bl.get("accuracy")
                if pilot_bl.get("accuracy") and full_bl.get("accuracy"):
                    row["accuracy_delta"] = full_bl["accuracy"] - pilot_bl["accuracy"]
                row["pilot_n"] = pilot_bl.get("n_cases")
                row["full_n"] = full_bl.get("n_cases")

        rows.append(row)

    if rows:
        return pd.DataFrame(rows)
    return None


# ---------------------------------------------------------------------------
# CSV export helpers
# ---------------------------------------------------------------------------

def format_pct(val) -> str:
    """Format a float as a percentage string for LaTeX."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "--"
    return f"{val * 100:.1f}"


def export_latex_table(df: pd.DataFrame, path: Path, caption: str = "", label: str = ""):
    """Export a DataFrame as a LaTeX table file."""
    latex = df.to_latex(index=False, float_format="%.3f", na_rep="--")
    if caption or label:
        lines = latex.split("\n")
        # Wrap in table environment
        wrapped = [
            "\\begin{table}[htbp]",
            "\\centering",
            f"\\caption{{{caption}}}",
            f"\\label{{{label}}}",
        ]
        wrapped.extend(lines)
        wrapped.append("\\end{table}")
        latex = "\n".join(wrapped)
    path.write_text(latex)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Analyze full-scale perturbation experiment results.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help="Directory containing experiment results (default: data/experiments/full_scale/)",
    )
    parser.add_argument(
        "--cases",
        type=str,
        default=str(REPO_ROOT / "data" / "processed" / "stratified_sample.json"),
        help="Path to the stratified sample JSON",
    )
    parser.add_argument(
        "--pilot-dir",
        type=str,
        default=None,
        help="Path to pilot results for comparison",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for CSV/LaTeX tables (default: <results-dir>/analysis/)",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    print(f"Loading results from {results_dir} ...")
    all_results = load_results(results_dir)

    if not all_results:
        print("No results found. Run experiments first.")
        sys.exit(1)

    # Load case metadata
    cases = load_cases(Path(args.cases))
    case_lookup = {}
    for c in cases:
        case_lookup[(c["case_name"], c["article"])] = c

    # Enrich results with metadata (country, etc.)
    for mk, mr in all_results.items():
        for rq, results in mr.items():
            for r in results:
                key = (r["case_name"], r["article"])
                if key in case_lookup:
                    r["respondent"] = case_lookup[key].get("respondent", "UNKNOWN")

    print(f"  Models loaded: {', '.join(MODELS[mk] for mk in all_results)}")
    print(f"  Output: {output_dir}\n")

    # -----------------------------------------------------------------------
    # 1. Cross-model summary table
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  TABLE 1: Cross-Model Summary")
    print("=" * 70)

    summary_df = build_summary_table(all_results)
    print(summary_df.to_string(index=False, float_format="%.3f"))
    summary_df.to_csv(output_dir / "table1_summary.csv", index=False)
    print()

    # -----------------------------------------------------------------------
    # 2. Per-model baseline details
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  TABLE 2: Baseline Accuracy by Model")
    print("=" * 70)

    baseline_rows = []
    for mk in MODELS:
        if mk not in all_results or "baseline" not in all_results[mk]:
            continue
        bl = analyze_baseline(all_results[mk]["baseline"])
        baseline_rows.append({
            "model": MODELS[mk],
            "n_cases": bl["n_cases"],
            "accuracy": format_pct(bl.get("accuracy")),
            "accuracy_violation": format_pct(bl.get("accuracy_violation")),
            "accuracy_no_violation": format_pct(bl.get("accuracy_no_violation")),
            "abstention_rate": format_pct(bl.get("abstention_rate")),
            "avg_rating": f"{bl.get('avg_rating', 0):.2f}",
        })
    baseline_df = pd.DataFrame(baseline_rows)
    print(baseline_df.to_string(index=False))
    baseline_df.to_csv(output_dir / "table2_baseline.csv", index=False)
    print()

    # -----------------------------------------------------------------------
    # 3. Per-country accuracy (baseline, all models)
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  TABLE 3: Baseline Accuracy by Country")
    print("=" * 70)

    country_df = build_country_model_table(all_results, cases)
    if not country_df.empty:
        # Pivot to model x country
        pivot = country_df.pivot_table(
            index="country", columns="model", values="accuracy", aggfunc="first"
        )
        pivot = pivot.round(3)
        print(pivot.to_string())
        pivot.to_csv(output_dir / "table3_country_accuracy.csv")
        country_df.to_csv(output_dir / "table3_country_accuracy_long.csv", index=False)
    print()

    # -----------------------------------------------------------------------
    # 4. Per-article accuracy (baseline, all models)
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  TABLE 4: Baseline Accuracy by Article")
    print("=" * 70)

    article_rows = []
    for mk in MODELS:
        if mk not in all_results or "baseline" not in all_results[mk]:
            continue
        art_df = baseline_by_article(all_results[mk]["baseline"])
        art_df["model"] = MODELS[mk]
        article_rows.append(art_df)
    if article_rows:
        all_articles = pd.concat(article_rows, ignore_index=True)
        pivot = all_articles.pivot_table(
            index=["article", "article_title"], columns="model",
            values="accuracy", aggfunc="first",
        )
        pivot = pivot.round(3)
        print(pivot.to_string())
        pivot.to_csv(output_dir / "table4_article_accuracy.csv")
    print()

    # -----------------------------------------------------------------------
    # 5. RQ1 -- Summarization fidelity
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  TABLE 5: RQ1 - Summarization Fidelity")
    print("=" * 70)

    rq1_rows = []
    for mk in MODELS:
        if mk not in all_results or "rq1" not in all_results[mk]:
            continue
        rq1 = analyze_rq1(all_results[mk]["rq1"])
        baseline_acc = None
        if "baseline" in all_results[mk]:
            bl = analyze_baseline(all_results[mk]["baseline"])
            baseline_acc = bl.get("accuracy")

        rq1_rows.append({
            "model": MODELS[mk],
            "baseline_accuracy": format_pct(baseline_acc),
            "summary_accuracy": format_pct(rq1.get("accuracy")),
            "alignment_rate": format_pct(rq1.get("alignment_rate")),
            "accuracy_delta": format_pct(
                (rq1["accuracy"] - baseline_acc) if baseline_acc and rq1.get("accuracy") else None
            ),
        })

    if rq1_rows:
        rq1_df = pd.DataFrame(rq1_rows)
        print(rq1_df.to_string(index=False))
        rq1_df.to_csv(output_dir / "table5_rq1_summarization.csv", index=False)
    print()

    # Per-version breakdown
    for mk in MODELS:
        if mk not in all_results or "rq1" not in all_results[mk]:
            continue
        version_df = rq1_by_version(all_results[mk]["rq1"])
        if not version_df.empty:
            version_df.to_csv(
                output_dir / f"rq1_{mk}_by_version.csv", index=False
            )

    # -----------------------------------------------------------------------
    # 6. RQ2 -- Framing effects
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  TABLE 6: RQ2 - Framing Effects")
    print("=" * 70)

    rq2_rows = []
    for mk in MODELS:
        if mk not in all_results or "rq2" not in all_results[mk]:
            continue
        rq2 = analyze_rq2(all_results[mk]["rq2"])
        for fname in ["predictive", "normative", "factual"]:
            if fname in rq2:
                rq2_rows.append({
                    "model": MODELS[mk],
                    "framing": fname,
                    "accuracy": format_pct(rq2[fname]["accuracy"]),
                    "alignment_with_baseline": format_pct(rq2[fname]["alignment_rate"]),
                })

    if rq2_rows:
        rq2_df = pd.DataFrame(rq2_rows)
        print(rq2_df.to_string(index=False))
        rq2_df.to_csv(output_dir / "table6_rq2_framing.csv", index=False)

        # Pivot for compact view
        rq2_pivot = rq2_df.pivot_table(
            index="model", columns="framing", values="accuracy", aggfunc="first"
        )
        rq2_pivot.to_csv(output_dir / "table6_rq2_framing_pivot.csv")
    print()

    # -----------------------------------------------------------------------
    # 7. RQ3 -- Reconsideration sycophancy
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  TABLE 7: RQ3 - Reconsideration Sycophancy")
    print("=" * 70)

    rq3_rows = []
    for mk in MODELS:
        if mk not in all_results or "rq3" not in all_results[mk]:
            continue
        rq3 = analyze_rq3(all_results[mk]["rq3"])
        rq3_rows.append({
            "model": MODELS[mk],
            "original_accuracy": format_pct(rq3.get("original_accuracy")),
            "challenged_accuracy": format_pct(rq3.get("challenged_accuracy")),
            "prediction_change_rate": format_pct(rq3.get("prediction_change_rate")),
            "avg_individual_change": format_pct(rq3.get("avg_individual_change_rate")),
        })

    if rq3_rows:
        rq3_df = pd.DataFrame(rq3_rows)
        print(rq3_df.to_string(index=False))
        rq3_df.to_csv(output_dir / "table7_rq3_reconsideration.csv", index=False)
    print()

    # Direction analysis
    for mk in MODELS:
        if mk not in all_results or "rq3" not in all_results[mk]:
            continue
        dir_df = rq3_direction_table(all_results[mk]["rq3"])
        if not dir_df.empty:
            dir_df.to_csv(output_dir / f"rq3_{mk}_direction.csv", index=False)
            # Print summary
            n_changes = len(dir_df)
            toward = (dir_df["direction"] == "toward_violation").sum()
            away = (dir_df["direction"] == "away_from_violation").sum()
            print(
                f"  {MODELS[mk]}: {n_changes} individual changes, "
                f"{toward} toward violation ({100*toward/n_changes:.1f}%), "
                f"{away} away ({100*away/n_changes:.1f}%)"
            )
    print()

    # -----------------------------------------------------------------------
    # 8. Pilot comparison (if available)
    # -----------------------------------------------------------------------
    if args.pilot_dir:
        print("=" * 70)
        print("  TABLE 8: Pilot vs Full-Scale Comparison")
        print("=" * 70)

        pilot_df = compare_with_pilot(all_results, Path(args.pilot_dir))
        if pilot_df is not None:
            print(pilot_df.to_string(index=False))
            pilot_df.to_csv(output_dir / "table8_pilot_comparison.csv", index=False)
        else:
            print("  No matching pilot results found.")
        print()

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("=" * 70)
    print("  FILES WRITTEN")
    print("=" * 70)
    for f in sorted(output_dir.glob("*.csv")):
        print(f"  {f.name}")
    print(f"\n  All files in: {output_dir}\n")


if __name__ == "__main__":
    main()
