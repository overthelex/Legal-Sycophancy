"""Turn the published 1k evaluation set into the cases JSON the runners expect.

    python scripts/build_livehrb_input.py --out data/processed/livehrb_1k.json
    python scripts/build_livehrb_input.py --limit 20 --out data/processed/dry_run.json

The set is instance-level: one row per case-article pair, 1,212 instances over 976
judgments, keyed on `article_full` so that Article 1 of Protocol 1 stays distinct
from Convention Article 1. Instances of the same judgment share `item_id`, which is
what lets the runners summarise, extract and score each record once and reuse it.

Procedural provisions (Articles 34, 38, 41, 46) are carried through with their flag
rather than dropped, matching the release convention: they are retained but bucketed
out of the substantive headline metric.
"""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

HUB = ("https://huggingface.co/datasets/overthelex/echr-livehrb-temporal-1k"
       "/resolve/refs%2Fconvert%2Fparquet/default/train/0000.parquet")


def build(source, limit=None, substantive_only=False, seed=42):
    frame = pd.read_parquet(source)
    if substantive_only:
        frame = frame[~frame["is_procedural"]]
    if limit:
        # sample whole judgments, not instances, so a case is never split across the
        # boundary -- its summary is computed once and shared by its instances
        ids = frame["item_id"].drop_duplicates().sample(
            n=min(limit, frame["item_id"].nunique()), random_state=seed)
        frame = frame[frame["item_id"].isin(ids)]

    cases = [{
        "item_id": row["item_id"],
        "case_name": row["case_name"],
        "article": row["article_full"],          # runners key prompts and buckets on this
        "article_legacy": row["article"],
        "violation_label": row["violation_label"],
        "full_case_text_no_verdict": row["verdict_free_text"],
        "respondent": row["respondent"],
        "decision_date": row["decision_date"],
        "year": int(row["year"]),
        "importance": row["importance"],
        "is_procedural": bool(row["is_procedural"]),
        "retention_percentage": float(row["retention_percentage"]),
    } for _, row in frame.iterrows()]
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default=HUB, help="parquet path or URL")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int, help="sample this many judgments (for a dry run)")
    parser.add_argument("--substantive-only", action="store_true",
                        help="drop procedural provisions instead of flagging them")
    args = parser.parse_args()

    cases = build(args.source, args.limit, args.substantive_only)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(cases, ensure_ascii=False))

    labels = Counter(c["violation_label"] for c in cases)
    years = Counter(c["year"] for c in cases)
    print(f"{len(cases)} instances over {len({c['item_id'] for c in cases})} judgments -> {args.out}")
    print(f"  labels: {dict(labels)} "
          f"(no_violation {100 * labels['no_violation'] / len(cases):.1f}%)")
    print(f"  years: {min(years)}-{max(years)}, {len(years)} bins")
    print(f"  articles: {len({c['article'] for c in cases})}, "
          f"procedural instances: {sum(c['is_procedural'] for c in cases)}")


if __name__ == "__main__":
    main()
