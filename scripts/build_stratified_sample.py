#!/usr/bin/env python3
"""
Build a balanced stratified sample from the multi-jurisdictional ECtHR
verdict-free dataset.

Stratification strategy:
  1. Rank respondent states by number of case-article pairs
  2. Select top N countries
  3. For each country, sample K *cases* (not pairs) uniformly at random,
     then include all article-pairs belonging to those cases
  4. If a country has fewer than K cases, take all of them

Usage:
  # From HuggingFace dataset
  python scripts/build_stratified_sample.py \
    --output data/processed/stratified_sample.json \
    --cases-per-country 20 --top-n-countries 20

  # From local JSON
  python scripts/build_stratified_sample.py \
    --input data/processed/echr_full.json \
    --output data/processed/stratified_sample.json

  # Custom parameters
  python scripts/build_stratified_sample.py \
    --output data/processed/stratified_sample_10x30.json \
    --cases-per-country 30 --top-n-countries 10 --seed 123
"""

import argparse
import json
import os
import random
import sys
from collections import Counter, defaultdict


def load_from_huggingface() -> list:
    """Load the dataset from HuggingFace Hub."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package required for HuggingFace loading.", file=sys.stderr)
        print("Install with: pip install datasets", file=sys.stderr)
        sys.exit(1)

    print("Loading dataset from HuggingFace: overthelex/echr-verdict-free ...")
    ds = load_dataset("overthelex/echr-verdict-free", split="train")
    records = [dict(row) for row in ds]
    print(f"  Loaded {len(records)} records from HuggingFace.")
    return records


def load_from_json(path: str) -> list:
    """Load dataset from a local JSON file."""
    print(f"Loading dataset from {path} ...")
    with open(path) as f:
        records = json.load(f)
    print(f"  Loaded {len(records)} records from JSON.")
    return records


def build_stratified_sample(
    records: list,
    cases_per_country: int,
    top_n_countries: int,
    seed: int,
) -> list:
    """
    Build a stratified sample:
      - Rank countries by pair count, take top N
      - For each country, sample K unique cases, include all their pairs
    """
    rng = random.Random(seed)

    # Group by respondent state
    country_pairs = defaultdict(list)
    for rec in records:
        country = rec.get("respondent", "UNKNOWN")
        country_pairs[country].append(rec)

    # Rank by pair count
    country_ranking = sorted(
        country_pairs.keys(),
        key=lambda c: len(country_pairs[c]),
        reverse=True,
    )
    selected_countries = country_ranking[:top_n_countries]

    # Sample cases per country
    sampled = []

    for country in selected_countries:
        pairs = country_pairs[country]

        # Group pairs by case (item_id)
        cases = defaultdict(list)
        for p in pairs:
            cases[p["item_id"]].append(p)

        case_ids = sorted(cases.keys())  # sort for determinism before sampling
        n_available = len(case_ids)
        n_sample = min(cases_per_country, n_available)

        chosen_ids = rng.sample(case_ids, n_sample)

        for cid in sorted(chosen_ids):  # deterministic output order
            sampled.extend(cases[cid])

    return sampled


def print_stats(records: list, label: str = "Dataset") -> None:
    """Print summary statistics for a set of records."""
    if not records:
        print(f"\n{label}: 0 records")
        return

    total_pairs = len(records)
    unique_cases = len({r["item_id"] for r in records})
    articles = Counter(r["article"] for r in records)
    countries = Counter(r.get("respondent", "UNKNOWN") for r in records)
    violations = sum(1 for r in records if r.get("violation_label") == "violation")
    no_violations = sum(1 for r in records if r.get("violation_label") == "no_violation")

    print(f"\n{'=' * 65}")
    print(f"  {label}")
    print(f"{'=' * 65}")
    print(f"  Total case-article pairs : {total_pairs:,}")
    print(f"  Unique cases             : {unique_cases:,}")
    print(f"  Countries                : {len(countries):,}")
    print(f"  Unique articles          : {len(articles):,}")
    print(f"  Violations               : {violations:,} ({100 * violations / total_pairs:.1f}%)")
    print(f"  No violations            : {no_violations:,} ({100 * no_violations / total_pairs:.1f}%)")

    # Per-country breakdown
    print(f"\n  {'Country':<25} {'Pairs':>6} {'Cases':>6} {'Viol%':>6}")
    print(f"  {'-' * 25} {'-' * 6} {'-' * 6} {'-' * 6}")

    country_cases = defaultdict(set)
    country_violations = defaultdict(int)
    country_pair_count = Counter()

    for r in records:
        c = r.get("respondent", "UNKNOWN")
        country_cases[c].add(r["item_id"])
        country_pair_count[c] += 1
        if r.get("violation_label") == "violation":
            country_violations[c] += 1

    for country in sorted(
        country_pair_count.keys(),
        key=lambda x: country_pair_count[x],
        reverse=True,
    ):
        n_pairs = country_pair_count[country]
        n_cases = len(country_cases[country])
        viol_rate = 100 * country_violations[country] / n_pairs if n_pairs else 0
        print(f"  {country:<25} {n_pairs:>6} {n_cases:>6} {viol_rate:>5.1f}%")

    # Article distribution
    print(f"\n  Article distribution:")
    for art, count in articles.most_common(15):
        print(f"    Art. {art:<6} : {count:>5} ({100 * count / total_pairs:.1f}%)")
    if len(articles) > 15:
        print(f"    ... and {len(articles) - 15} more articles")


def main():
    parser = argparse.ArgumentParser(
        description="Build a balanced stratified sample from the ECtHR verdict-free dataset.",
    )
    parser.add_argument(
        "--input",
        type=str,
        default=None,
        help="Path to input JSON file. If not provided, loads from HuggingFace.",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output path for the stratified sample JSON.",
    )
    parser.add_argument(
        "--cases-per-country",
        type=int,
        default=20,
        help="Number of cases to sample per country (default: 20).",
    )
    parser.add_argument(
        "--top-n-countries",
        type=int,
        default=20,
        help="Number of top respondent states to include (default: 20).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    args = parser.parse_args()

    # Load data
    if args.input:
        records = load_from_json(args.input)
    else:
        records = load_from_huggingface()

    if not records:
        print("ERROR: No records loaded.", file=sys.stderr)
        sys.exit(1)

    # Show source stats
    print_stats(records, label="Source dataset")

    # Build sample
    print(
        f"\nBuilding stratified sample: top {args.top_n_countries} countries, "
        f"{args.cases_per_country} cases each, seed={args.seed}"
    )

    sample = build_stratified_sample(
        records,
        cases_per_country=args.cases_per_country,
        top_n_countries=args.top_n_countries,
        seed=args.seed,
    )

    # Show sample stats
    print_stats(sample, label="Stratified sample")

    # Write output
    with open(args.output, "w") as f:
        json.dump(sample, f, ensure_ascii=False, indent=2)

    file_size = os.path.getsize(args.output)
    print(f"\nWritten {len(sample)} records to {args.output}")
    print(f"File size: {file_size / (1024 * 1024):.1f} MB")


if __name__ == "__main__":
    main()
