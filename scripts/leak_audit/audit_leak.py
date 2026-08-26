"""Audit the published datasets for retained Court reasoning.

Reads the released parquet exports straight from the Hub, so the numbers can be
reproduced by anyone without local state:

    python scripts/leak_audit/audit_leak.py
    python scripts/leak_audit/audit_leak.py --datasets echr-livehrb-static-2k

Reports the leak rate per dataset and its skew along every axis the paper draws
a comparison on. The skew matters more than the headline rate: a defect spread
evenly is a constant, one concentrated in the rare class is a confound.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from leakdef import leaking  # noqa: E402

HUB = ("https://huggingface.co/datasets/overthelex/{name}"
       "/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet")

DATASETS = {
    "echr-verdict-free": "verdict_free_text",
    "echr-ukr-verdict-free": "verdict_free_text",
    "echr-livehrb-static-2k": "verdict_free_text",
    "echr-livehrb-temporal-2k": "verdict_free_text",
    "echr-livehrb-stateswap": "case_text_rendered",
}

AXES = ["group", "arm", "bin", "violation_label", "importance",
        "respondent", "respondent_original", "article_full"]
SKEW_THRESHOLD = 8.0   # percentage points between the extreme buckets
MIN_BUCKET = 25


def audit(name, column, min_bucket=MIN_BUCKET):
    frame = pd.read_parquet(HUB.format(name=name))
    frame["leak"] = frame[column].map(leaking)
    print(f"\n=== {name}: {len(frame):,} rows, {frame['leak'].sum():,} leaking "
          f"({100 * frame['leak'].mean():.1f}%)")

    if "decision_date" in frame.columns:
        frame["decision_year"] = frame["decision_date"].astype(str).str[:4]

    for axis in AXES + ["decision_year"]:
        if axis not in frame.columns:
            continue
        table = frame.groupby(axis)["leak"].agg(["sum", "count", "mean"])
        table = table[table["count"] >= min_bucket].sort_values("mean", ascending=False)
        if len(table) < 2:
            continue
        spread = 100 * (table["mean"].max() - table["mean"].min())
        marker = "  <-- SKEWED" if spread >= SKEW_THRESHOLD else ""
        print(f"  by {axis}: spread {spread:.1f}pp{marker}")
        shown = pd.concat([table.head(3), table.tail(1)]) if len(table) > 4 else table
        for key, row in shown.iterrows():
            print(f"     {str(key)[:32]:32} {int(row['sum']):5}/{int(row['count']):5}"
                  f" = {100 * row['mean']:5.1f}%")
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS),
                        help="subset of dataset names to audit")
    parser.add_argument("--min-bucket", type=int, default=MIN_BUCKET,
                        help="ignore axis buckets smaller than this")
    args = parser.parse_args()

    for name in args.datasets:
        if name not in DATASETS:
            print(f"unknown dataset {name}; known: {', '.join(DATASETS)}")
            continue
        audit(name, DATASETS[name], args.min_bucket)


if __name__ == "__main__":
    main()
