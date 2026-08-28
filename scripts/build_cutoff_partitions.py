#!/usr/bin/env python3
"""
Build pre/post-cutoff matched partitions per model for the contamination
audit (RQ1).

For each model m with training cutoff c_m (configs/model_cutoffs.json),
evaluation cases are split into:

    D_pre(m)  = cases decided before c_m   (potentially in training data)
    D_post(m) = cases decided after  c_m   (guaranteed unseen)

The contamination gap is  Delta_m = Acc_m(D_pre) - Acc_m(D_post)  on MATCHED
samples: each post-cutoff case is greedily paired with an unused pre-cutoff
case sharing (respondent, article, violation_label), so the two partitions
have identical distributions over the matching key. Unmatched cases are
reported and excluded from the matched analysis.

Models with a null cutoff in the registry are refused -- verify the provider
documentation first, never guess.

Usage:
  python scripts/build_cutoff_partitions.py \
      --dataset data/processed/stratified_sample.json \
      --dates data/processed/decision_dates.json \
      --cutoffs configs/model_cutoffs.json \
      --models gpt-4o llama-3.3-70b \
      --output data/processed/partitions/

  # All models with verified cutoffs:
  python scripts/build_cutoff_partitions.py \
      --dataset ... --dates ... --cutoffs configs/model_cutoffs.json --all \
      --output data/processed/partitions/
"""

import argparse
import json
import os
import random
import sys
from collections import defaultdict


def load_json(path):
    with open(path) as f:
        return json.load(f)


def match_key(rec: dict) -> tuple:
    return (rec.get("respondent", ""), rec.get("article", ""),
            rec.get("violation_label", ""))


def build_partition(records: list, dates: dict, cutoff: str, seed: int) -> dict:
    """Split records on cutoff date and greedily 1:1 match post->pre on
    (respondent, article, violation_label)."""
    pre, post, undated = [], [], []
    for rec in records:
        d = dates.get(rec.get("item_id", ""), {}).get("decision_date", "")
        if not d:
            undated.append(rec["item_id"])
            continue
        (pre if d < cutoff else post).append(rec)

    pre_by_key = defaultdict(list)
    for rec in pre:
        pre_by_key[match_key(rec)].append(rec)

    rng = random.Random(seed)
    for bucket in pre_by_key.values():
        rng.shuffle(bucket)

    matched_pairs, unmatched_post = [], []
    post_shuffled = list(post)
    rng.shuffle(post_shuffled)
    for rec in post_shuffled:
        bucket = pre_by_key.get(match_key(rec))
        if bucket:
            matched_pairs.append({"post": rec["item_id"] + "|" + rec["article"],
                                  "pre": bucket.pop()["item_id"] + "|" + rec["article"],
                                  "key": list(match_key(rec))})
        else:
            unmatched_post.append(rec["item_id"])

    return {
        "cutoff": cutoff,
        "n_pre_total": len(pre),
        "n_post_total": len(post),
        "n_undated": len(undated),
        "n_matched_pairs": len(matched_pairs),
        "n_unmatched_post": len(unmatched_post),
        "matched_pairs": matched_pairs,
        "unmatched_post_item_ids": unmatched_post,
        "undated_item_ids": undated,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True, help="Dataset JSON (list of records)")
    ap.add_argument("--dates", required=True,
                    help="decision_dates.json from backfill_decision_dates.py")
    ap.add_argument("--cutoffs", default="configs/model_cutoffs.json")
    ap.add_argument("--models", nargs="*", default=[],
                    help="Model keys from the cutoff registry")
    ap.add_argument("--all", action="store_true",
                    help="Partition for every model with a verified (non-null) cutoff")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", required=True, help="Output directory")
    args = ap.parse_args()

    records = load_json(args.dataset)
    dates = load_json(args.dates)
    registry = load_json(args.cutoffs)["models"]

    if args.all:
        models = [m for m, v in registry.items() if v.get("cutoff")]
        skipped = [m for m, v in registry.items() if not v.get("cutoff")]
        if skipped:
            print(f"Skipping models with unverified cutoffs: {', '.join(skipped)}")
    else:
        models = args.models
        if not models:
            ap.error("Provide --models or --all")

    os.makedirs(args.output, exist_ok=True)
    for m in models:
        if m not in registry:
            print(f"ERROR: '{m}' not in cutoff registry {args.cutoffs}", file=sys.stderr)
            return 1
        cutoff = registry[m].get("cutoff")
        if not cutoff:
            print(f"ERROR: cutoff for '{m}' is null -- verify provider docs and "
                  f"fill configs/model_cutoffs.json (source field is mandatory)",
                  file=sys.stderr)
            return 1

        part = build_partition(records, dates, cutoff, args.seed)
        part["model"] = m
        part["cutoff_source"] = registry[m].get("source", "")
        out_path = os.path.join(args.output, f"partition_{m.replace('/', '_')}.json")
        with open(out_path, "w") as f:
            json.dump(part, f, indent=1)
        print(f"{m}: cutoff {cutoff} -> pre {part['n_pre_total']}, "
              f"post {part['n_post_total']}, matched pairs {part['n_matched_pairs']}, "
              f"unmatched post {part['n_unmatched_post']}, undated {part['n_undated']} "
              f"-> {out_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
