"""
Step 0 — Build the contamination input file from the v1.1 leak-scrubbed 2K set.

Loads the LiveHumanRightsBench static-2k dataset (from the Hugging Face Hub by
default, or from a local JSON/Parquet file) and rewrites it into the case format
that the contamination runners expect (the `echr_cases_final_clean.json` schema),
while carrying through the new stratification axes (respondent, group, decision
date, importance) for downstream group/temporal analysis.

The only real transform is a field rename:
    verdict_free_text  ->  full_case_text  (+ full_case_text_no_verdict, same value)
Everything else is passthrough. Labels are already 'violation' / 'no_violation'.

Usage
-----
# Pull the static-2k set straight from the Hub (default):
python scripts/build_contamination_input.py

# Full judgments only (drop the ~17% HUDOC Information Notes, item_id '002-*'):
python scripts/build_contamination_input.py --full-judgments-only

# From a local file instead of the Hub:
python scripts/build_contamination_input.py --source path/to/static_2k.json
"""

import argparse
import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_HF_DATASET = "overthelex/echr-livehrb-static-2k"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "echr_livehrb_static_2k.json"

# Fields we carry through verbatim if present on the source row.
PASSTHROUGH_FIELDS = [
    "item_id", "case_name", "respondent", "article", "article_full",
    "violation_label", "verdict_removal_method", "original_length",
    "verdict_free_length", "retention_percentage", "decision_date",
    "application_number", "importance", "ecli", "group",
]

VALID_LABELS = {"violation", "no_violation"}

# Article-key normalization. The legacy `article` field collapses protocol
# prefixes (article='1' is Article 1 of Protocol 1 / property, not Convention
# Art 1; '4' conflates Convention Art 4 with Protocol 4). Since v1.2 the datasets
# carry a protocol-aware `article_full` (P1-1, P4-2, P7-4, P12-1, …) that resolves
# all of these — we key on it when present. ARTICLE_KEY_REMAP is a fallback only
# for older sources that predate article_full.
ARTICLE_KEY_REMAP = {"1": "P1-1"}


def load_rows(source: str):
    """Return a list of dict rows from either the Hub or a local file."""
    local = Path(source)
    if local.exists():
        print(f"Loading local file: {local}")
        if local.suffix == ".json":
            with open(local) as f:
                return json.load(f)
        elif local.suffix in (".parquet", ".pq"):
            import pandas as pd
            return pd.read_parquet(local).to_dict(orient="records")
        else:
            print(f"ERROR: unsupported local file type: {local.suffix}")
            sys.exit(1)

    # Otherwise treat `source` as a Hugging Face dataset id.
    print(f"Loading Hugging Face dataset: {source}")
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: `datasets` not installed. Run: pip install datasets")
        sys.exit(1)
    ds = load_dataset(source)
    split = "train" if "train" in ds else list(ds.keys())[0]
    print(f"  Using split: {split} ({len(ds[split])} rows)")
    return [dict(r) for r in ds[split]]


def convert(rows, full_judgments_only: bool):
    out = []
    skipped_genre = 0
    skipped_bad = 0
    remapped = 0
    for r in rows:
        item_id = str(r.get("item_id", ""))

        # HUDOC genre filter: 001-* = full judgment, 002-* = Information Note.
        if full_judgments_only and not item_id.startswith("001-"):
            skipped_genre += 1
            continue

        text = r.get("verdict_free_text") or ""
        label = r.get("violation_label")
        if not text or label not in VALID_LABELS:
            skipped_bad += 1
            continue

        rec = {k: r[k] for k in PASSTHROUGH_FIELDS if k in r}
        # Prefer the protocol-aware article_full (v1.2+); it resolves every
        # collapsed key, not just '1'. Fall back to the interim ARTICLE_KEY_REMAP
        # only when article_full is absent (older sources).
        af = str(r.get("article_full") or "").strip()
        if af:
            if af != str(rec.get("article", "")):
                remapped += 1
            rec["article"] = af
        elif rec.get("article") in ARTICLE_KEY_REMAP:
            rec["article"] = ARTICLE_KEY_REMAP[rec["article"]]
            remapped += 1
        # The rename: contamination runners read full_case_text / *_no_verdict.
        rec["full_case_text"] = text
        rec["full_case_text_no_verdict"] = text
        out.append(rec)

    if remapped:
        print(f"  Set {remapped} article keys to their protocol-aware code "
              f"(article_full where present, else ARTICLE_KEY_REMAP).")
    if skipped_genre:
        print(f"  Dropped {skipped_genre} Information-Note (002-*) rows "
              f"[--full-judgments-only].")
    if skipped_bad:
        print(f"  Dropped {skipped_bad} rows with missing text or bad label.")
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Build contamination input from the v1.1 static-2k set."
    )
    parser.add_argument(
        "--source", default=DEFAULT_HF_DATASET,
        help=f"HF dataset id or local .json/.parquet path "
             f"(default: {DEFAULT_HF_DATASET})",
    )
    parser.add_argument(
        "--output", type=Path, default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--full-judgments-only", action="store_true",
        help="Keep only full judgments (item_id 001-*); drop 002-* Info Notes.",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("STEP 0: BUILD CONTAMINATION INPUT (v1.1 static-2k)")
    print("=" * 80)
    print(f"Source : {args.source}")
    print(f"Output : {args.output}")
    print(f"Genre  : {'full judgments only' if args.full_judgments_only else 'all (judgments + info notes)'}")
    print("=" * 80)

    rows = load_rows(args.source)
    print(f"\nLoaded {len(rows)} source rows.")

    cases = convert(rows, args.full_judgments_only)
    if not cases:
        print("\nERROR: no usable cases after conversion.")
        sys.exit(1)

    # Quick composition report — useful sanity check before you spend API calls.
    from collections import Counter
    labels = Counter(c["violation_label"] for c in cases)
    groups = Counter(c.get("group", "?") for c in cases)
    articles = Counter(c["article"] for c in cases)
    print(f"\nConverted {len(cases)} cases.")
    print(f"  Labels   : {dict(labels)}")
    print(f"  Groups   : {dict(groups)}")
    print(f"  Articles : {dict(sorted(articles.items(), key=lambda kv: -kv[1]))}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(cases, f, indent=2, ensure_ascii=False)
    print(f"\nSaved {len(cases)} cases to {args.output}")
    print("Next: python experiments/generate_mft.py")


if __name__ == "__main__":
    main()
