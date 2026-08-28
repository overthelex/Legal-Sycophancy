#!/usr/bin/env python3
"""
Backfill decision dates and application numbers from HUDOC metadata.

The published datasets (overthelex/echr-verdict-free) carry item_id but not
decision_date / application_number. Both are required for the contamination
audit (pre/post-cutoff partitions) and the facts-ablated variants.

This script sweeps the HUDOC search API (metadata only, no document bodies),
collects {item_id -> decision_date, application_number, importance, ecli}
for all English judgments, and writes a lookup map. The sweep is paged by
date range, so it is resumable and HUDOC-friendly.

Usage:
  # Full sweep (all English judgments, ~30K metadata rows, no text download)
  python scripts/backfill_decision_dates.py --output data/processed/decision_dates.json

  # Restrict to a date range
  python scripts/backfill_decision_dates.py --since 2020-01-01 --output data/processed/decision_dates.json

  # Resume an interrupted sweep
  python scripts/backfill_decision_dates.py --output data/processed/decision_dates.json --resume

  # Verify coverage against a dataset file
  python scripts/backfill_decision_dates.py --check data/processed/stratified_sample.json \
      --output data/processed/decision_dates.json
"""

import argparse
import json
import os
import sys
import time

import requests

# Reuse the HUDOC client from the scraper
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hudoc_scraper import (  # noqa: E402
    build_search_query,
    search_hudoc,
    parse_search_result,
    PAGE_SIZE,
)


def load_existing(path: str) -> dict:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


# HUDOC's search API stops returning results past offset ~10000 (Lucene
# deep-pagination cap), so ranges with more matches are swept in
# per-year windows.
HUDOC_OFFSET_CAP = 9500
EARLIEST_YEAR = 1959


def sweep_range(session, since: str, until: str, collected: dict,
                output: str, checkpoint_every: int = 2000) -> int:
    """Page through one date range (must be under the offset cap)."""
    query = build_search_query(since=since or "", until=until or "",
                               language="ENG", doc_collection="JUDGMENTS")
    first = search_hudoc(session, query, start=0, length=1)
    total = first.get("resultcount", 0)
    if total == 0:
        return 0

    start = 0
    since_checkpoint = 0
    while start < total:
        page = search_hudoc(session, query, start=start, length=PAGE_SIZE)
        results = page.get("results", [])
        if not results:
            break
        for r in results:
            meta = parse_search_result(r)
            item_id = meta.get("item_id")
            if not item_id:
                continue
            collected[item_id] = {
                "decision_date": meta.get("decision_date", ""),
                "application_number": meta.get("application_number", ""),
                "importance": meta.get("importance", ""),
                "ecli": meta.get("ecli", ""),
            }
        start += len(results)
        since_checkpoint += len(results)
        print(f"  [{since or '*'} .. {until or '*'}] {start}/{total} "
              f"(collected {len(collected)} total)")
        if since_checkpoint >= checkpoint_every:
            save(collected, output)
            since_checkpoint = 0
        time.sleep(0.5)

    return total


def sweep_metadata(since: str, until: str, existing: dict, output: str) -> dict:
    """Sweep HUDOC metadata, splitting into per-year windows when the range
    exceeds the deep-pagination cap."""
    session = requests.Session()
    query = build_search_query(since=since or "", until=until or "",
                               language="ENG", doc_collection="JUDGMENTS")
    first = search_hudoc(session, query, start=0, length=1)
    total = first.get("resultcount", 0)
    print(f"HUDOC reports {total} English judgments in range "
          f"[{since or '*'} .. {until or '*'}]")

    collected = dict(existing)

    if total <= HUDOC_OFFSET_CAP:
        sweep_range(session, since, until, collected, output)
        return collected

    start_year = int(since[:4]) if since else EARLIEST_YEAR
    end_year = int(until[:4]) if until else time.gmtime().tm_year
    print(f"Range exceeds HUDOC offset cap ({HUDOC_OFFSET_CAP}); "
          f"sweeping per-year windows {start_year}..{end_year}")
    for year in range(start_year, end_year + 1):
        y_since = max(since or "", f"{year}-01-01")
        y_until = f"{year + 1}-01-01"
        if until:
            y_until = min(until, y_until)
        n = sweep_range(session, y_since, y_until, collected, output)
        print(f"  year {year}: {n} judgments")
        save(collected, output)

    return collected


def fill_missing(item_ids: list, collected: dict, output: str) -> None:
    """Targeted per-item queries for ids the bulk sweep did not cover
    (e.g. 002-* legal summaries, non-JUDGMENTS doc types)."""
    session = requests.Session()
    for i, item_id in enumerate(item_ids, 1):
        query = f'contentsitename:ECHR AND (itemid:"{item_id}")'
        # HUDOC does PREFIX matching on itemid (itemid:"002-112" ranks
        # 002-11250 first) -- fetch a page and keep the exact id only.
        # This poisoned 50 dates across published sets before 2026-07-03.
        page = search_hudoc(session, query, start=0, length=100)
        results = [r for r in page.get("results", [])
                   if r.get("columns", {}).get("itemid") == item_id]
        if results:
            meta = parse_search_result(results[0])
            collected[item_id] = {
                "decision_date": meta.get("decision_date", ""),
                "application_number": meta.get("application_number", ""),
                "importance": meta.get("importance", ""),
                "ecli": meta.get("ecli", ""),
            }
            print(f"  [{i}/{len(item_ids)}] {item_id} -> {meta.get('decision_date', '?')}")
        else:
            print(f"  [{i}/{len(item_ids)}] {item_id} -> NOT FOUND")
        if i % 25 == 0:
            save(collected, output)
        time.sleep(0.4)
    save(collected, output)


def save(data: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


def check_coverage(dataset_path: str, dates: dict) -> None:
    with open(dataset_path) as f:
        records = json.load(f)
    item_ids = {r["item_id"] for r in records}
    missing = sorted(i for i in item_ids
                     if i not in dates or not dates[i].get("decision_date"))
    print(f"Coverage check against {dataset_path}:")
    print(f"  dataset item_ids: {len(item_ids)}")
    print(f"  with decision_date: {len(item_ids) - len(missing)}")
    print(f"  missing: {len(missing)}")
    if missing:
        print("  first missing:", missing[:10])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", required=True, help="Output JSON map path")
    ap.add_argument("--since", default=None, help="Start date YYYY-MM-DD")
    ap.add_argument("--until", default=None, help="End date YYYY-MM-DD")
    ap.add_argument("--resume", action="store_true",
                    help="Merge into existing output instead of starting fresh")
    ap.add_argument("--check", default=None,
                    help="Dataset JSON to verify coverage against (skips sweep if used alone with existing output)")
    ap.add_argument("--check-only", action="store_true",
                    help="Only run the coverage check, no sweep")
    ap.add_argument("--fill-missing", default=None,
                    help="Dataset JSON; targeted per-item queries for item_ids "
                         "missing from the existing output map")
    args = ap.parse_args()

    existing = load_existing(args.output) if (args.resume or args.check_only
                                              or args.fill_missing) else {}

    if args.fill_missing:
        with open(args.fill_missing) as f:
            records = json.load(f)
        missing = sorted({r["item_id"] for r in records}
                         - {i for i, v in existing.items() if v.get("decision_date")})
        print(f"{len(missing)} item_ids missing dates; querying individually")
        fill_missing(missing, existing, args.output)
        if args.check:
            check_coverage(args.check, existing)
        return 0

    if not args.check_only:
        collected = sweep_metadata(args.since, args.until, existing, args.output)
        save(collected, args.output)
        print(f"Saved {len(collected)} item_id -> metadata entries to {args.output}")
    else:
        collected = existing
        print(f"Loaded {len(collected)} entries from {args.output}")

    if args.check:
        check_coverage(args.check, collected)

    return 0


if __name__ == "__main__":
    sys.exit(main())
