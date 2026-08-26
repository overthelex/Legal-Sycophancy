#!/usr/bin/env python3
"""Summarise a perturbation run and correct for multiplicity across it.

Every (model, arm, variant) is compared against that model's own baseline with an
exact McNemar test, and the whole family is then BH-corrected together. Correcting
per arm would defeat the purpose: the multiplicity comes from running four arms over
eight models, not from any one of them.

Reports, per comparison: accuracy and balanced accuracy, abstention rate, unparsed
calls, mean confidence, and the flip rate split by direction. The split matters --
an aggregate flip rate cannot distinguish drift toward "violation" from drift away
from it, and on the pilot those were 17-2 and 28-4 one way.

    python scripts/analyse_perturbation_run.py --run-dir data/experiments/full_scale
"""

import argparse, csv, json, os, sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))
from stats import balanced_accuracy, benjamini_hochberg, mcnemar_exact   # noqa: E402

ARMS = {"rq1": "summary_version", "rq2": "framing", "rq3": None}


def load(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def describe(rows):
    """The per-condition numbers we agreed to report alongside accuracy."""
    n = len(rows)
    confidences = [r["avg_rating"] for r in rows if r.get("avg_rating") is not None]
    directions = Counter(r["flip_direction"] for r in rows if r.get("flip_direction"))
    return {
        "n": n,
        "accuracy": sum(1 for r in rows if r.get("accurate")) / n if n else None,
        "balanced_accuracy": balanced_accuracy(rows) if n else None,
        "abstention_rate": sum(1 for r in rows if r.get("abstained")) / n if n else None,
        "n_unparsed": sum(r.get("n_unparsed", 0) for r in rows),
        "mean_confidence": sum(confidences) / len(confidences) if confidences else None,
        "flip_rate": sum(directions.values()) / n if n else None,
        "flips_to_violation": directions.get("no_violation->violation", 0),
        "flips_to_no_violation": directions.get("violation->no_violation", 0),
        # A judgment that moves into the 40-60 band has not reversed, it has
        # stopped committing. The 1-5 scale could not express this at all: it
        # returned only two values in 555 samples, so every change looked like a
        # reversal. On the pilot this is the most common thing summaries cause.
        "flips_to_abstention": sum(v for k, v in directions.items()
                                   if k.endswith("->abstention")),
        "flips_out_of_abstention": sum(v for k, v in directions.items()
                                       if k.startswith("abstention->")),
    }


def paired(baseline, rows, key):
    """Discordant counts of correctness against the baseline for the same unit."""
    ref = {(r["item_id"], r["article"]): r for r in baseline}
    n01 = n10 = 0
    for r in rows:
        b = ref.get((r["item_id"], r["article"]))
        if b is None:
            continue
        if b.get("accurate") and not r.get("accurate"):
            n10 += 1
        elif not b.get("accurate") and r.get("accurate"):
            n01 += 1
    return n01, n10


def main():
    p = argparse.ArgumentParser(description="Summarise and BH-correct a perturbation run")
    p.add_argument("--run-dir", required=True, help="directory of per-model result folders")
    p.add_argument("--out", default="perturbation_summary.csv")
    args = p.parse_args()

    comparisons = []
    for model_dir in sorted(os.listdir(args.run_dir)):
        base_path = os.path.join(args.run_dir, model_dir, "baseline_results.json")
        baseline = load(base_path)
        if not baseline:
            print(f"  {model_dir}: no baseline, skipped")
            continue
        row = {"model": model_dir, "arm": "baseline", "variant": "", **describe(baseline)}
        row["mcnemar_n"], row["mcnemar_p"] = "", ""
        comparisons.append(row)

        for arm, variant_key in ARMS.items():
            rows = load(os.path.join(args.run_dir, model_dir, f"{arm}_results.json"))
            if not rows:
                print(f"  {model_dir}: {arm} missing")
                continue
            groups = defaultdict(list)
            for r in rows:
                groups[str(r.get(variant_key, "")) if variant_key else ""].append(r)
            for variant, subset in sorted(groups.items()):
                if arm == "rq3":
                    # RQ3's own control is the first answer, not the baseline arm.
                    # Its rows name the two answers separately, so map them onto the
                    # field names the shared helpers read -- otherwise balanced
                    # accuracy reaches for `prediction` and finds nothing. That went
                    # unnoticed once because the slice it ran on held a single class,
                    # and balanced accuracy returns early before touching the field.
                    for r in subset:
                        r["prediction"] = r.get("challenged_prediction")
                        r["abstained"] = r.get("challenged_abstained")
                        r["accurate"] = r["prediction"] == r["violation_label"]
                    ref = [{**r,
                            "prediction": r.get("original_prediction"),
                            "abstained": r.get("original_abstained"),
                            "accurate": r.get("original_prediction") == r["violation_label"]}
                           for r in subset]
                    n01, n10 = paired(ref, subset, variant)
                else:
                    n01, n10 = paired(baseline, subset, variant)
                n, pval = mcnemar_exact(n01, n10)
                comparisons.append({
                    "model": model_dir, "arm": arm, "variant": variant,
                    **describe(subset),
                    "mcnemar_n": n, "mcnemar_p": pval,
                    "worse_than_reference": n10, "better_than_reference": n01,
                })

    tested = [c for c in comparisons if c["mcnemar_p"] != ""]
    for c, q in zip(tested, benjamini_hochberg([c["mcnemar_p"] for c in tested])):
        c["mcnemar_q"] = q
        c["significant_bh"] = q < 0.05
    for c in comparisons:
        c.setdefault("mcnemar_q", "")
        c.setdefault("significant_bh", "")

    if not comparisons:
        sys.exit("No results found. Check --run-dir.")
    fields = list(dict.fromkeys(k for c in comparisons for k in c))
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(comparisons)

    print(f"\nWrote {args.out}: {len(comparisons)} conditions, {len(tested)} tests\n")
    raw = sum(1 for c in tested if c["mcnemar_p"] < 0.05)
    kept = sum(1 for c in tested if c["significant_bh"])
    print(f"Significant at 0.05: {raw} raw, {kept} after BH across the family of {len(tested)}")
    for c in tested:
        if c["significant_bh"]:
            print(f"  {c['model']:<28s} {c['arm']}/{c['variant']:<12s} "
                  f"q={c['mcnemar_q']:.4f}  acc={c['accuracy']:.3f} "
                  f"bal={c['balanced_accuracy'] or float('nan'):.3f}  "
                  f"flips {c['flips_to_violation']}->viol / {c['flips_to_no_violation']}->no")


if __name__ == "__main__":
    main()
