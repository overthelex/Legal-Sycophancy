#!/usr/bin/env python3
"""
HUDOC Live Refresh Orchestrator for LiveHumanRightsBench

Automated pipeline that:
  1. Determines the latest decision_date in the current dataset
  2. Downloads new judgments from HUDOC since that date
  3. Runs the verdict-leakage removal pipeline on new judgments
  4. Appends clean results to the existing dataset
  5. Optionally pushes the updated dataset to HuggingFace

Designed to run as a cron job for continuous "live" benchmark updates.

Usage:
  # Basic refresh from existing dataset
  python scripts/hudoc_live_refresh.py \
    --dataset data/processed/echr_cases_ukr_eng_final.json \
    --output data/processed/echr_live_updated.json

  # Refresh from HuggingFace dataset
  python scripts/hudoc_live_refresh.py \
    --hf-dataset overthelex/echr-verdict-free \
    --output data/processed/echr_live_updated.json

  # Refresh with full pipeline (LLM verification)
  python scripts/hudoc_live_refresh.py \
    --dataset data/processed/echr_cases_ukr_eng_final.json \
    --output data/processed/echr_live_updated.json \
    --full-pipeline

  # Refresh and push to HuggingFace
  python scripts/hudoc_live_refresh.py \
    --dataset data/processed/echr_cases_ukr_eng_final.json \
    --output data/processed/echr_live_updated.json \
    --push-to-hf overthelex/echr-verdict-free

  # Filter to a specific country
  python scripts/hudoc_live_refresh.py \
    --dataset data/processed/echr_cases_ukr_eng_final.json \
    --country UKR \
    --output data/processed/echr_ukr_live.json

  # Cron-friendly: quiet output, auto-detect dates
  python scripts/hudoc_live_refresh.py \
    --dataset data/processed/echr_live.json \
    --output data/processed/echr_live.json \
    --quiet
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta
from pathlib import Path


# ── Constants ──────────────────────────────────────────────────────────────────

SCRIPTS_DIR = Path(__file__).parent.resolve()
SCRAPER_SCRIPT = SCRIPTS_DIR / "hudoc_scraper.py"
REMOVAL_SCRIPT = SCRIPTS_DIR / "verdict_leakage_removal.py"


# ── Dataset helpers ────────────────────────────────────────────────────────────

def load_dataset(path: str) -> list[dict]:
    """Load existing dataset from a JSON file."""
    if not os.path.exists(path):
        print(f"  Dataset file not found: {path}")
        return []
    with open(path) as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} existing records from {path}")
    return data


def load_from_huggingface(dataset_name: str) -> list[dict]:
    """Load dataset from HuggingFace Hub."""
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package required for HuggingFace loading.", file=sys.stderr)
        print("Install with: pip install datasets", file=sys.stderr)
        sys.exit(1)

    print(f"  Loading dataset from HuggingFace: {dataset_name} ...")
    ds = load_dataset(dataset_name, split="train")
    records = [dict(row) for row in ds]
    print(f"  Loaded {len(records)} records from HuggingFace.")
    return records


def get_latest_date(records: list[dict]) -> str:
    """
    Find the latest decision_date in the dataset.
    Returns date string in YYYY-MM-DD format, or empty string if none found.
    """
    dates = []
    for r in records:
        date_val = (r.get("decision_date") or r.get("date")
                    or r.get("kp_date") or r.get("judgment_date") or "")
        date_str = str(date_val)[:10] if date_val else ""
        if date_str and len(date_str) == 10:
            try:
                datetime.strptime(date_str, "%Y-%m-%d")
                dates.append(date_str)
            except ValueError:
                continue
    return max(dates) if dates else ""


def get_existing_item_ids(records: list[dict]) -> set[str]:
    """Get set of item_ids already in the dataset."""
    return {r.get("item_id", "") for r in records if r.get("item_id")}


def deduplicate_records(existing: list[dict], new_records: list[dict]) -> list[dict]:
    """
    Merge new records into existing, deduplicating by (item_id, article) pair.
    New records take precedence over existing ones (fresher data).
    """
    seen = set()
    merged = []

    # Index existing records
    for r in existing:
        key = (r.get("item_id", ""), r.get("article", ""))
        if key not in seen:
            seen.add(key)
            merged.append(r)

    # Add new records, replacing existing duplicates
    added = 0
    replaced = 0
    for r in new_records:
        key = (r.get("item_id", ""), r.get("article", ""))
        if key in seen:
            # Replace: find and update the existing record
            for i, existing_r in enumerate(merged):
                if (existing_r.get("item_id"), existing_r.get("article")) == key:
                    merged[i] = r
                    replaced += 1
                    break
        else:
            seen.add(key)
            merged.append(r)
            added += 1

    print(f"  Deduplication: {added} added, {replaced} replaced, {len(merged)} total")
    return merged


# ── Pipeline orchestration ─────────────────────────────────────────────────────

def run_scraper(
    since: str,
    until: str = "",
    country: str = "",
    output_path: str = "",
    resume: bool = True,
    delay: float = 1.0,
) -> list[dict]:
    """
    Run the HUDOC scraper and return the downloaded records.
    """
    cmd = [
        sys.executable, str(SCRAPER_SCRIPT),
        "--since", since,
        "--output", output_path,
        "--delay", str(delay),
    ]
    if until:
        cmd.extend(["--until", until])
    if country:
        cmd.extend(["--country", country])
    if resume:
        cmd.append("--resume")

    print(f"\n{'='*60}")
    print(f"Step 1: Downloading new judgments from HUDOC")
    print(f"{'='*60}")
    print(f"  Since: {since}")
    print(f"  Until: {until or 'now'}")
    print(f"  Country: {country or 'all'}")
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\n  ERROR: Scraper exited with code {result.returncode}", file=sys.stderr)
        return []

    # Load the scraper output
    if os.path.exists(output_path):
        with open(output_path) as f:
            records = json.load(f)
        return records
    return []


def run_verdict_removal(
    input_path: str,
    output_path: str,
    stage1_only: bool = True,
    workers: int = 5,
    cutoff_date: str = "",
    end_date: str = "",
) -> list[dict]:
    """
    Run the verdict-leakage removal pipeline on the given input file.
    """
    cmd = [
        sys.executable, str(REMOVAL_SCRIPT),
        "--source", "json",
        "--input", input_path,
        "--output", output_path,
        "--workers", str(workers),
    ]
    if stage1_only:
        cmd.append("--stage1-only")
    if cutoff_date:
        cmd.extend(["--cutoff-date", cutoff_date])
    if end_date:
        cmd.extend(["--end-date", end_date])

    print(f"\n{'='*60}")
    print(f"Step 2: Running verdict-leakage removal")
    print(f"{'='*60}")
    print(f"  Mode: {'stage1 only (pattern-based)' if stage1_only else 'full pipeline (pattern + LLM)'}")
    print(f"  Input: {input_path}")
    print(f"  Output: {output_path}")
    print(f"  Command: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False, text=True)

    if result.returncode != 0:
        print(f"\n  ERROR: Verdict removal exited with code {result.returncode}", file=sys.stderr)
        return []

    if os.path.exists(output_path):
        with open(output_path) as f:
            records = json.load(f)
        return records
    return []


def push_to_huggingface(records: list[dict], repo_id: str, commit_message: str = ""):
    """
    Push the updated dataset to HuggingFace Hub.
    """
    try:
        from datasets import Dataset
        from huggingface_hub import HfApi
    except ImportError:
        print("ERROR: 'datasets' and 'huggingface_hub' packages required for HuggingFace push.", file=sys.stderr)
        print("Install with: pip install datasets huggingface_hub", file=sys.stderr)
        return False

    print(f"\n{'='*60}")
    print(f"Step 4: Pushing to HuggingFace")
    print(f"{'='*60}")
    print(f"  Repository: {repo_id}")
    print(f"  Records: {len(records)}")

    if not commit_message:
        today = datetime.now().strftime("%Y-%m-%d")
        dates = sorted(set(r.get("decision_date", "") for r in records if r.get("decision_date")))
        date_range = f"{dates[0]} to {dates[-1]}" if dates else "unknown"
        commit_message = f"Live refresh {today}: {len(records)} records, dates {date_range}"

    # Prepare dataset - remove full_case_text to keep dataset size manageable
    # (the HF dataset should have the verdict-free text, not the original)
    columns_to_keep = [
        "item_id", "case_name", "article", "violation_label",
        "full_case_text_no_verdict", "decision_date", "respondent",
        "verdict_removal_method", "original_length", "verdict_free_length",
        "retention_percentage",
    ]

    clean_records = []
    for r in records:
        clean = {k: r.get(k, "") for k in columns_to_keep}
        # Rename for consistency
        if "full_case_text_no_verdict" in clean and clean["full_case_text_no_verdict"]:
            clean["case_text"] = clean.pop("full_case_text_no_verdict")
        else:
            clean["case_text"] = r.get("case_text", r.get("full_case_text", ""))
            clean.pop("full_case_text_no_verdict", None)
        clean_records.append(clean)

    try:
        ds = Dataset.from_list(clean_records)
        ds.push_to_hub(repo_id, commit_message=commit_message)
        print(f"  Pushed successfully: {commit_message}")
        return True
    except Exception as e:
        print(f"  ERROR pushing to HuggingFace: {e}", file=sys.stderr)
        return False


# ── Main orchestrator ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Live refresh orchestrator for LiveHumanRightsBench",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script automates the full "live" loop:
  1. Check the latest decision date in the current dataset
  2. Download new HUDOC judgments since that date
  3. Run verdict-leakage removal on new judgments
  4. Merge clean results into the existing dataset
  5. Optionally push to HuggingFace

For cron usage:
  # Run daily at 06:00 UTC
  0 6 * * * cd /path/to/Legal-Sycophancy && python scripts/hudoc_live_refresh.py \\
    --dataset data/processed/echr_live.json \\
    --output data/processed/echr_live.json \\
    --quiet >> logs/live_refresh.log 2>&1
""",
    )

    # Input sources
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--dataset", type=str,
        help="Path to existing dataset JSON file",
    )
    input_group.add_argument(
        "--hf-dataset", type=str,
        help="HuggingFace dataset name (e.g., overthelex/echr-verdict-free)",
    )

    # Output
    parser.add_argument(
        "--output", required=True,
        help="Output path for the updated dataset",
    )

    # Filtering
    parser.add_argument(
        "--country", default="",
        help="Filter by respondent state (e.g., UKR)",
    )
    parser.add_argument(
        "--since-override", default="",
        help="Override auto-detected start date (YYYY-MM-DD). Useful for initial seeding.",
    )

    # Pipeline options
    parser.add_argument(
        "--full-pipeline", action="store_true",
        help="Run full verdict removal (pattern + LLM). Default is stage1 only.",
    )
    parser.add_argument(
        "--workers", type=int, default=5,
        help="Number of parallel workers for LLM verification (default: 5)",
    )
    parser.add_argument(
        "--delay", type=float, default=1.0,
        help="Delay between HUDOC requests in seconds (default: 1.0)",
    )

    # HuggingFace push
    parser.add_argument(
        "--push-to-hf", default="",
        help="Push updated dataset to this HuggingFace repo (e.g., overthelex/echr-verdict-free)",
    )

    # Misc
    parser.add_argument(
        "--quiet", action="store_true",
        help="Minimal output (for cron jobs)",
    )
    parser.add_argument(
        "--keep-temp", action="store_true",
        help="Keep temporary files (raw downloads, intermediate results)",
    )

    args = parser.parse_args()

    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if not args.quiet:
        print(f"{'='*60}")
        print(f"  LiveHumanRightsBench - HUDOC Live Refresh")
        print(f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*60}")

    # ── Step 0: Load existing dataset ──────────────────────────────────────

    if args.dataset:
        existing_records = load_dataset(args.dataset)
    else:
        existing_records = load_from_huggingface(args.hf_dataset)

    # ── Step 0.5: Determine date range ─────────────────────────────────────

    if args.since_override:
        since_date = args.since_override
        print(f"  Using override start date: {since_date}")
    elif existing_records:
        latest = get_latest_date(existing_records)
        if latest:
            # Start from the day after the latest known date to avoid re-downloading
            # the same cases. But subtract 1 day as a safety margin for HUDOC
            # indexing lag.
            try:
                latest_dt = datetime.strptime(latest, "%Y-%m-%d")
                since_dt = latest_dt - timedelta(days=1)
                since_date = since_dt.strftime("%Y-%m-%d")
            except ValueError:
                since_date = latest
            print(f"  Latest date in dataset: {latest}")
            print(f"  Fetching since: {since_date} (1-day overlap for safety)")
        else:
            print("  WARNING: No dates found in existing dataset. Using 2020-01-01 as default.", file=sys.stderr)
            since_date = "2020-01-01"
    else:
        print("  No existing dataset. Using 2020-01-01 as default start date.")
        since_date = "2020-01-01"

    # Check if there's anything new to fetch
    today = datetime.now().strftime("%Y-%m-%d")
    if since_date >= today:
        print(f"\n  Dataset is already up to date (latest: {since_date}, today: {today})")
        print(f"  Nothing to do.")
        return

    # ── Step 1: Download new judgments ──────────────────────────────────────

    # Use temp directory for intermediate files
    temp_dir = tempfile.mkdtemp(prefix="hudoc_refresh_")
    raw_path = os.path.join(temp_dir, f"hudoc_raw_{timestamp}.json")
    clean_path = os.path.join(temp_dir, f"hudoc_clean_{timestamp}.json")

    scraped_records = run_scraper(
        since=since_date,
        country=args.country,
        output_path=raw_path,
        resume=True,
        delay=args.delay,
    )

    if not scraped_records:
        print("\n  No new judgments found. Dataset is up to date.")
        if not args.keep_temp:
            _cleanup_temp(temp_dir)
        return

    # Filter out records we already have
    existing_ids = get_existing_item_ids(existing_records)
    truly_new = [r for r in scraped_records if r.get("item_id") not in existing_ids]

    if not truly_new:
        print(f"\n  All {len(scraped_records)} scraped records already exist in dataset.")
        print(f"  Dataset is up to date.")
        if not args.keep_temp:
            _cleanup_temp(temp_dir)
        return

    print(f"\n  New records to process: {truly_new_count(truly_new, existing_ids)}")

    # Save the truly new records for verdict removal
    new_raw_path = os.path.join(temp_dir, f"hudoc_new_{timestamp}.json")
    with open(new_raw_path, "w") as f:
        json.dump(truly_new, f, ensure_ascii=False)

    # ── Step 2: Run verdict-leakage removal ────────────────────────────────

    clean_records = run_verdict_removal(
        input_path=new_raw_path,
        output_path=clean_path,
        stage1_only=not args.full_pipeline,
        workers=args.workers,
    )

    if not clean_records:
        print("\n  WARNING: Verdict removal produced no clean records.", file=sys.stderr)
        if not args.keep_temp:
            _cleanup_temp(temp_dir)
        return

    # ── Step 3: Merge with existing dataset ────────────────────────────────

    print(f"\n{'='*60}")
    print(f"Step 3: Merging datasets")
    print(f"{'='*60}")
    print(f"  Existing records: {len(existing_records)}")
    print(f"  New clean records: {len(clean_records)}")

    merged = deduplicate_records(existing_records, clean_records)

    # Sort by decision_date descending
    merged.sort(
        key=lambda r: r.get("decision_date", "") or "0000-00-00",
        reverse=True,
    )

    # Save merged dataset
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    print(f"  Saved {len(merged)} records to {args.output}")

    # ── Step 4 (optional): Push to HuggingFace ─────────────────────────────

    if args.push_to_hf:
        push_to_huggingface(merged, args.push_to_hf)

    # ── Cleanup ────────────────────────────────────────────────────────────

    if not args.keep_temp:
        _cleanup_temp(temp_dir)

    # ── Summary ────────────────────────────────────────────────────────────

    elapsed = time.time() - start_time

    print(f"\n{'='*60}")
    print(f"  Live Refresh Complete")
    print(f"{'='*60}")
    print(f"  Elapsed: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"  Existing records: {len(existing_records)}")
    print(f"  New records added: {len(merged) - len(existing_records)}")
    print(f"  Total records: {len(merged)}")

    # Date range
    dates = sorted(set(r.get("decision_date", "") for r in merged if r.get("decision_date")))
    if dates:
        print(f"  Date range: {dates[0]} to {dates[-1]}")

    # Per-country breakdown (top 10)
    from collections import Counter
    countries = Counter(r.get("respondent", "?") for r in merged)
    labels = Counter(r.get("violation_label", "?") for r in merged)
    print(f"  Violation labels: {dict(labels)}")
    if len(countries) <= 10:
        print(f"  Countries: {dict(countries.most_common())}")
    else:
        top10 = countries.most_common(10)
        print(f"  Top 10 countries: {dict(top10)}")
        print(f"  Total countries: {len(countries)}")

    print(f"\n  Output: {args.output}")


def truly_new_count(records: list[dict], existing_ids: set[str]) -> int:
    """Count records whose item_id is not in existing_ids."""
    return sum(1 for r in records if r.get("item_id") not in existing_ids)


def _cleanup_temp(temp_dir: str):
    """Remove temporary directory and its contents."""
    import shutil
    try:
        shutil.rmtree(temp_dir)
    except OSError:
        pass


if __name__ == "__main__":
    main()
