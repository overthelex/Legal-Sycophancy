#!/usr/bin/env python3
"""
Backfill the protocol-aware `article_full` column onto published datasets.

The published sets (overthelex/echr-verdict-free, echr-livehrb-static-2k, ...)
carry a lossy `article` field with protocol prefixes collapsed ("Article 1 of
Protocol No. 1" -> "1"). They do NOT carry the raw `conclusion` string (it was
removed as a verdict leak), so the full code cannot be recovered from the row
alone. This script re-fetches the HUDOC `conclusion` column per item_id, parses
it with the fixed parser (hudoc_scraper.parse_conclusion_to_pairs -> article_full),
and writes an enriched dataset with a non-breaking `article_full` column added.

`article` is left untouched for continuity. Rows whose full code cannot be
resolved unambiguously fall back to the legacy `article` value (never fabricated)
and are counted in the coverage report.

Usage:
  # test on a small batch first (recommended before a full run)
  python scripts/backfill_article_full.py \
    --dataset data/echr-livehrb-static-2k.parquet \
    --output  data/echr-livehrb-static-2k.article_full.parquet \
    --limit 50

  # full run with resumable HUDOC cache
  python scripts/backfill_article_full.py \
    --dataset data/echr-livehrb-static-2k.parquet \
    --output  data/echr-livehrb-static-2k.article_full.parquet \
    --map-output data/conclusion_map.json --resume

  # enrich and push back to HuggingFace
  python scripts/backfill_article_full.py \
    --hf-dataset overthelex/echr-livehrb-static-2k \
    --output data/static-2k-v12.parquet \
    --map-output data/conclusion_map.json --resume \
    --push-to-hf overthelex/echr-livehrb-static-2k

  # coverage report only (no HUDOC calls), against an existing map
  python scripts/backfill_article_full.py \
    --dataset data/echr-livehrb-static-2k.parquet \
    --map-output data/conclusion_map.json --check-only
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

try:
    import pandas as pd
except ImportError:
    print("ERROR: pandas is required (pip install pandas pyarrow)", file=sys.stderr)
    sys.exit(1)

try:
    import requests
except ImportError:
    print("ERROR: requests is required (pip install requests)", file=sys.stderr)
    sys.exit(1)

# Reuse the single source of truth for HUDOC access and the fixed parser.
SCRIPTS_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPTS_DIR))
from hudoc_scraper import (  # noqa: E402
    build_search_query,
    search_hudoc,
    parse_search_result,
    parse_conclusion_to_pairs,
)

HUDOC_OFFSET_CAP = 10000   # HUDOC stops serving results past this offset
BULK_PAGE = 500            # max page size for the bulk search endpoint


# ── Dataset I/O ──────────────────────────────────────────────────────────────

def load_dataset(dataset: str, hf_dataset: str) -> pd.DataFrame:
    """Load a local parquet/json or an HF dataset into a DataFrame."""
    if hf_dataset:
        from datasets import load_dataset as hf_load
        ds = hf_load(hf_dataset, split="train")
        return ds.to_pandas()
    if dataset.endswith(".parquet"):
        return pd.read_parquet(dataset)
    if dataset.endswith(".json"):
        return pd.read_json(dataset)
    raise ValueError(f"Unsupported dataset format: {dataset}")


def load_map(path: str) -> dict:
    if path and os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_map(data: dict, path: str) -> None:
    if not path:
        return
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=1)
    os.replace(tmp, path)


# ── HUDOC sweep ──────────────────────────────────────────────────────────────

def fetch_conclusion_pairs(session, item_id: str) -> list:
    """Exact per-item HUDOC query -> parsed (article, article_full, label) pairs.

    HUDOC prefix-matches itemid (itemid:"002-112" ranks 002-11250 first), so we
    keep only the exact match -- the same guard that poisoned 50 dates before.
    """
    query = f'contentsitename:ECHR AND (itemid:"{item_id}")'
    page = search_hudoc(session, query, start=0, length=100)
    exact = [r for r in page.get("results", [])
             if r.get("columns", {}).get("itemid") == item_id]
    if not exact:
        return []
    meta = parse_search_result(exact[0])
    pairs = parse_conclusion_to_pairs(
        conclusion=meta.get("conclusion", ""),
        item_id=item_id,
        case_name=meta.get("case_name", ""),
        decision_date=meta.get("decision_date", ""),
        respondent=meta.get("respondent", ""),
        full_text="",  # not needed; keeps parse from emitting an "unknown" row
    )
    return [{"article": p["article"],
             "article_full": p["article_full"],
             "violation_label": p["violation_label"]}
            for p in pairs if p.get("article")]


def _pairs_from_meta(meta: dict) -> list:
    return [{"article": p["article"],
             "article_full": p["article_full"],
             "violation_label": p["violation_label"]}
            for p in parse_conclusion_to_pairs(
                conclusion=meta.get("conclusion", ""),
                item_id=meta.get("item_id", ""),
                case_name=meta.get("case_name", ""),
                decision_date=meta.get("decision_date", ""),
                respondent=meta.get("respondent", ""),
                full_text="")
            if p.get("article")]


def bulk_sweep(target_ids: set, cache: dict, map_path: str, delay: float,
               since_year: int, until_year: int) -> tuple:
    """Paged bulk sweep over per-year windows (JUDGMENTS collection).

    Pulls itemid+conclusion ~500 at a time and keeps only target ids, so ~30-80
    requests replace tens of thousands of per-item queries. Per-year windows keep
    each result set under the 10K deep-pagination cap. 002-* Information Notes are
    not in JUDGMENTS and stay in `remaining` for the per-item fallback.
    """
    session = requests.Session()
    remaining = set(target_ids) - set(cache)
    print(f"bulk sweep: {len(remaining)} target ids to find, years "
          f"{since_year}..{until_year}")
    for year in range(since_year, until_year + 1):
        if not remaining:
            break
        query = build_search_query(since=f"{year}-01-01", until=f"{year + 1}-01-01",
                                   language="ENG", doc_collection="JUDGMENTS")
        start, hits = 0, 0
        while True:
            resp = search_hudoc(session, query, start=start, length=BULK_PAGE)
            results = resp.get("results", [])
            if not results:
                break
            for r in results:
                meta = parse_search_result(r)
                iid = meta.get("item_id")
                if iid in remaining:
                    cache[iid] = _pairs_from_meta(meta)
                    remaining.discard(iid)
                    hits += 1
            start += len(results)
            if start >= HUDOC_OFFSET_CAP or len(results) < BULK_PAGE:
                break
            time.sleep(delay)
        if hits:
            print(f"  {year}: +{hits} (remaining {len(remaining)})")
            save_map(cache, map_path)
        time.sleep(delay)
    save_map(cache, map_path)
    return cache, remaining


def sweep(item_ids: list, cache: dict, map_path: str, delay: float,
          checkpoint_every: int = 50) -> dict:
    session = requests.Session()
    todo = [i for i in item_ids if i not in cache]
    print(f"{len(item_ids)} unique item_ids; {len(todo)} to fetch "
          f"({len(item_ids) - len(todo)} cached)")
    for n, item_id in enumerate(todo, 1):
        try:
            cache[item_id] = fetch_conclusion_pairs(session, item_id)
            tag = ",".join(sorted({p["article_full"] for p in cache[item_id]})) or "NONE"
        except Exception as e:  # network hiccup: leave uncached, resumable
            print(f"  [{n}/{len(todo)}] {item_id} -> ERROR {e}", file=sys.stderr)
            continue
        if n % 20 == 0 or n == len(todo):
            print(f"  [{n}/{len(todo)}] {item_id} -> {tag}")
        if n % checkpoint_every == 0:
            save_map(cache, map_path)
        time.sleep(delay)
    save_map(cache, map_path)
    return cache


# ── Enrichment ───────────────────────────────────────────────────────────────

def build_lookup(cache: dict) -> dict:
    """(item_id, article, label) -> set(article_full)."""
    lut = {}
    for item_id, pairs in cache.items():
        for p in pairs:
            key = (item_id, str(p["article"]), p["violation_label"])
            lut.setdefault(key, set()).add(p["article_full"])
    return lut


def enrich(df: pd.DataFrame, cache: dict) -> tuple:
    lut = build_lookup(cache)
    filled = protocol = fallback = ambiguous = 0
    out = []
    for _, row in df.iterrows():
        item_id = str(row.get("item_id", ""))
        article = str(row.get("article", ""))
        label = str(row.get("violation_label", ""))
        cands = lut.get((item_id, article, label))
        if cands is None:
            # try without the label (some rows differ only by conclusion phrasing)
            cands = set().union(*[v for k, v in lut.items()
                                  if k[0] == item_id and k[1] == article]) or None
        if cands and len(cands) == 1:
            code = next(iter(cands))
            filled += 1
            if code.startswith("P"):
                protocol += 1
        elif cands and len(cands) > 1:
            code = article          # ambiguous within the case -> keep legacy
            ambiguous += 1
        else:
            code = article          # unresolved -> keep legacy (never fabricate)
            fallback += 1
        out.append(code)
    df = df.copy()
    df["article_full"] = out
    stats = {"rows": len(df), "resolved": filled, "protocol_coded": protocol,
             "ambiguous_kept_legacy": ambiguous, "unresolved_kept_legacy": fallback}
    return df, stats


def push_to_hf(df: pd.DataFrame, repo: str) -> None:
    from datasets import Dataset
    print(f"Pushing {len(df)} rows to {repo} (split=train) ...")
    Dataset.from_pandas(df, preserve_index=False).push_to_hub(repo, split="train")
    print("  pushed.")


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--dataset", help="Local parquet/json to enrich")
    src.add_argument("--hf-dataset", help="HF dataset repo to enrich")
    ap.add_argument("--output", help="Output path (.parquet or .json)")
    ap.add_argument("--map-output", default=None,
                    help="JSON cache of item_id -> conclusion pairs (resume)")
    ap.add_argument("--resume", action="store_true",
                    help="Reuse an existing --map-output cache")
    ap.add_argument("--check-only", action="store_true",
                    help="Coverage report only; no HUDOC calls")
    ap.add_argument("--limit", type=int, default=0,
                    help="Fetch at most N new item_ids (small test batch)")
    ap.add_argument("--bulk", action="store_true",
                    help="Bulk paged sweep by year window (fast), then per-item "
                         "fallback for uncovered ids (e.g. 002-* Info Notes)")
    ap.add_argument("--since-year", type=int, default=1959,
                    help="First year for the bulk sweep window (default 1959)")
    ap.add_argument("--delay", type=float, default=1.0,
                    help="Seconds between HUDOC requests (politeness)")
    ap.add_argument("--push-to-hf", default=None, help="HF repo to push enriched set")
    args = ap.parse_args()

    df = load_dataset(args.dataset or "", args.hf_dataset or "")
    if "item_id" not in df.columns or "article" not in df.columns:
        print("ERROR: dataset needs item_id and article columns", file=sys.stderr)
        return 1
    print(f"Loaded {len(df)} rows.")

    cache = load_map(args.map_output) if (args.resume or args.check_only) else {}

    if not args.check_only:
        item_ids = list(dict.fromkeys(str(i) for i in df["item_id"]))
        if args.bulk:
            until_year = time.gmtime().tm_year
            cache, remaining = bulk_sweep(set(item_ids), cache, args.map_output,
                                          args.delay, args.since_year, until_year)
            if remaining:
                print(f"per-item fallback for {len(remaining)} uncovered ids "
                      f"(002-* Info Notes etc.)")
                cache = sweep(sorted(remaining), cache, args.map_output, args.delay)
        else:
            if args.limit:
                uncached = [i for i in item_ids if i not in cache][:args.limit]
                item_ids = [i for i in item_ids if i in cache] + uncached
                print(f"--limit {args.limit}: fetching up to {len(uncached)} new item_ids")
            cache = sweep(item_ids, cache, args.map_output, args.delay)

    enriched, stats = enrich(df, cache)
    print("\nCoverage:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    dist = enriched["article_full"].astype(str).value_counts()
    print("\narticle_full top values:")
    print(dist.head(20).to_string())

    if args.output:
        if args.output.endswith(".parquet"):
            enriched.to_parquet(args.output, index=False)
        else:
            enriched.to_json(args.output, orient="records", indent=1)
        print(f"\nWrote {args.output}")

    if args.push_to_hf:
        push_to_hf(enriched, args.push_to_hf)

    return 0


if __name__ == "__main__":
    sys.exit(main())
