#!/usr/bin/env python3
"""
HUDOC Bulk Downloader for ECtHR Judgments

Downloads judgment texts from the HUDOC public API (hudoc.echr.coe.int),
extracts metadata, and outputs JSON compatible with the verdict_leakage_removal
pipeline.

Usage:
  # Fetch all English judgments since 2025-01-01
  python scripts/hudoc_scraper.py --since 2025-01-01 --output data/raw/hudoc_2025.json

  # Fetch Ukrainian cases only
  python scripts/hudoc_scraper.py --since 2024-01-01 --country UKR --output data/raw/hudoc_ukr_2024.json

  # Fetch a specific date range
  python scripts/hudoc_scraper.py --since 2024-01-01 --until 2025-01-01 --output data/raw/hudoc_2024.json

  # Resume a previous download (skip already-downloaded item_ids)
  python scripts/hudoc_scraper.py --since 2024-01-01 --output data/raw/hudoc_2024.json --resume

  # Dry run (metadata only, no full text download)
  python scripts/hudoc_scraper.py --since 2025-01-01 --dry-run --output data/raw/hudoc_meta.json

API reference:
  Search endpoint: https://hudoc.echr.coe.int/app/query/results
  Document body:   https://hudoc.echr.coe.int/app/conversion/docx/html/body?library=ECHR&id={item_id}
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, asdict, field
from datetime import datetime, date
from html.parser import HTMLParser
from typing import Optional
from urllib.parse import quote

try:
    import requests
except ImportError:
    print("ERROR: 'requests' package is required. Install with: pip install requests", file=sys.stderr)
    sys.exit(1)


# ── Configuration ──────────────────────────────────────────────────────────────

HUDOC_SEARCH_URL = "https://hudoc.echr.coe.int/app/query/results"
HUDOC_DOC_BODY_URL = "https://hudoc.echr.coe.int/app/conversion/docx/html/body"

# Fields to retrieve from the search API
SEARCH_FIELDS = [
    "itemid", "docname", "appno", "article", "conclusion",
    "kpdate", "respondent", "respondentOrderEng",
    "languageisocode", "importance", "ecli",
    "doctypebranch", "separateopinion",
]

PAGE_SIZE = 50  # HUDOC returns up to 500 per page; use 50 for politeness
REQUEST_DELAY = 1.0  # seconds between requests (be polite)
MAX_RETRIES = 3
RETRY_BACKOFF = 5.0  # seconds, multiplied by attempt number

# User-Agent header for polite scraping
HEADERS = {
    "User-Agent": "LiveHumanRightsBench/1.0 (academic research; https://github.com/overthelex/Legal-Sycophancy)",
    "Accept": "application/json",
}


# ── HTML to text conversion ───────────────────────────────────────────────────

class HTMLTextExtractor(HTMLParser):
    """Simple HTML-to-text converter that preserves paragraph structure."""

    def __init__(self):
        super().__init__()
        self._text_parts = []
        self._skip = False

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip = True
        elif tag in ("p", "div", "br", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr"):
            self._text_parts.append("\n")

    def handle_endtag(self, tag):
        if tag in ("script", "style"):
            self._skip = False
        elif tag in ("p", "div", "h1", "h2", "h3", "h4", "h5", "h6", "li", "tr", "table"):
            self._text_parts.append("\n")
        elif tag == "td":
            self._text_parts.append("  ")

    def handle_data(self, data):
        if not self._skip:
            self._text_parts.append(data)

    def get_text(self) -> str:
        raw = "".join(self._text_parts)
        # Collapse multiple blank lines into two newlines
        raw = re.sub(r"\n{3,}", "\n\n", raw)
        # Collapse multiple spaces within lines
        lines = [re.sub(r"[ \t]+", " ", line.strip()) for line in raw.split("\n")]
        return "\n".join(lines).strip()


def html_to_text(html: str) -> str:
    """Convert HTML to clean plain text."""
    parser = HTMLTextExtractor()
    parser.feed(html)
    return parser.get_text()


# ── HUDOC API client ──────────────────────────────────────────────────────────

def build_search_query(
    since: str = "",
    until: str = "",
    country: str = "",
    language: str = "ENG",
    doc_collection: str = "JUDGMENTS",
) -> str:
    """
    Build a HUDOC search query string.

    The query uses HUDOC's custom query language:
      - contentsitename:ECHR -- search the ECHR library
      - documentcollectionid:"JUDGMENTS" -- only judgments
      - languageisocode:"ENG" -- English language
      - respondent:"UKR" -- respondent state
      - kpdate range filters
    """
    parts = [
        'contentsitename:ECHR',
        '(NOT (doctype:PR OR doctype:HFCOMOLD OR doctype:HECOMOLD))',
    ]

    if language:
        parts.append(f'((languageisocode:"{language}"))')

    if doc_collection:
        parts.append(f'((documentcollectionid:"{doc_collection}"))')

    if country:
        parts.append(f'((respondent:"{country.upper()}"))')

    # Date range filter using kpdate (Lucene range syntax)
    if since or until:
        range_start = f"{since}T00:00:00" if since else "*"
        range_end = f"{until}T00:00:00" if until else "*"
        parts.append(f'((kpdate:[{range_start} TO {range_end}]))')

    return " AND ".join(parts)


def search_hudoc(
    session: requests.Session,
    query: str,
    start: int = 0,
    length: int = PAGE_SIZE,
) -> dict:
    """
    Execute a search query against the HUDOC API.

    Returns the raw JSON response dict with keys:
      - resultcount: total number of matching results
      - results: list of result items
    """
    params = {
        "query": query,
        "select": ",".join(SEARCH_FIELDS),
        "sort": "kpdate Descending",
        "start": str(start),
        "length": str(length),
    }

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(HUDOC_SEARCH_URL, params=params, headers=HEADERS, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES:
                raise
            wait = RETRY_BACKOFF * attempt
            print(f"  [retry {attempt}/{MAX_RETRIES}] Search request failed: {e}. Waiting {wait}s...")
            time.sleep(wait)

    return {"resultcount": 0, "results": []}


def fetch_document_text(session: requests.Session, item_id: str) -> Optional[str]:
    """
    Fetch the full text of a HUDOC document by its item_id.

    Downloads the HTML body and converts to plain text.
    Returns None on failure.
    """
    url = f"{HUDOC_DOC_BODY_URL}?library=ECHR&id={item_id}"

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = session.get(url, headers=HEADERS, timeout=120)
            resp.raise_for_status()
            text = html_to_text(resp.text)
            if text and len(text) > 100:
                return text
            return None
        except requests.exceptions.RequestException as e:
            if attempt == MAX_RETRIES:
                print(f"  [FAIL] Could not fetch document {item_id}: {e}")
                return None
            wait = RETRY_BACKOFF * attempt
            print(f"  [retry {attempt}/{MAX_RETRIES}] Doc fetch failed for {item_id}: {e}. Waiting {wait}s...")
            time.sleep(wait)

    return None


# ── Result parsing ─────────────────────────────────────────────────────────────

def parse_search_result(result: dict) -> dict:
    """
    Parse a single HUDOC search result into a flat metadata dict.

    Input: {"columns": {"itemid": "001-...", "docname": "CASE OF ...", ...}}
    Output: flat dict with normalized field names
    """
    cols = result.get("columns", {})

    # Parse kpdate
    kpdate_raw = cols.get("kpdate", "")
    decision_date = ""
    if kpdate_raw:
        try:
            decision_date = kpdate_raw[:10]  # "2026-06-09T00:00:00" -> "2026-06-09"
        except (ValueError, IndexError):
            decision_date = ""

    # Parse articles (semicolon-separated)
    articles_raw = cols.get("article", "")
    articles = [a.strip() for a in articles_raw.split(";") if a.strip()] if articles_raw else []

    # Parse application numbers
    appno = cols.get("appno", "")

    return {
        "item_id": cols.get("itemid", ""),
        "case_name": cols.get("docname", ""),
        "application_number": appno,
        "decision_date": decision_date,
        "respondent": cols.get("respondent", ""),
        "articles": articles,
        "conclusion": cols.get("conclusion", ""),
        "language": cols.get("languageisocode", ""),
        "importance": cols.get("importance", ""),
        "ecli": cols.get("ecli", ""),
        "doc_type_branch": cols.get("doctypebranch", ""),
        "separate_opinion": cols.get("separateopinion", ""),
    }


def parse_article_code(part: str) -> tuple[Optional[str], Optional[str]]:
    """
    Extract the article from a conclusion fragment, returning
    (article, article_full).

    - article       : legacy collapsed code (bare number, kept for continuity)
    - article_full  : protocol-aware code (e.g. "P1-1", "P4-2"), never lossy

    "Article 1 of Protocol No. 1"  -> ("1",  "P1-1")   (property, NOT Convention Art 1)
    "Article 2 of Protocol No. 4"  -> ("2",  "P4-2")
    "P1-1"                         -> ("1",  "P1-1")
    "Article 6"                    -> ("6",  "6")
    """
    # "Article <art> of Protocol No. <proto>"
    m = re.search(r'Article\s+(\d+)\s+of\s+Protocol\s+(?:No\.?\s*)?(\d+)',
                  part, re.IGNORECASE)
    if m:
        art, proto = m.group(1), m.group(2)
        return art, f"P{proto}-{art}"
    # Already-coded thesaurus form "P<proto>-<art>"
    m = re.search(r'\bP(\d+)-(\d+)\b', part)
    if m:
        proto, art = m.group(1), m.group(2)
        return art, f"P{proto}-{art}"
    # Plain Convention article "Article <art>"
    m = re.search(r'Article\s+(\d+)', part, re.IGNORECASE)
    if m:
        return m.group(1), m.group(1)
    return None, None


def parse_conclusion_to_pairs(conclusion: str, item_id: str, case_name: str,
                                decision_date: str, respondent: str,
                                full_text: str) -> list[dict]:
    """
    Parse the HUDOC conclusion string into case-article pairs,
    matching the format expected by verdict_leakage_removal.py.

    Each pair has the full CaseRecord-compatible fields:
      item_id, case_name, article, article_full, violation_label,
      full_case_text, decision_date
    """
    results = []
    if not conclusion:
        # No conclusion available - still include the case as "unknown"
        results.append({
            "item_id": item_id,
            "case_name": case_name,
            "article": "",
            "article_full": "",
            "violation_label": "unknown",
            "full_case_text": full_text,
            "decision_date": decision_date,
            "respondent": respondent,
        })
        return results

    parts = re.split(r'[;]', conclusion)
    for part in parts:
        part = part.strip()
        article, article_full = parse_article_code(part)
        if not article:
            continue

        if re.search(r'no\s+violation', part, re.IGNORECASE):
            label = "no_violation"
        elif re.search(r'violation', part, re.IGNORECASE):
            label = "violation"
        else:
            continue

        results.append({
            "item_id": item_id,
            "case_name": case_name,
            "article": article,
            "article_full": article_full,
            "violation_label": label,
            "full_case_text": full_text,
            "decision_date": decision_date,
            "respondent": respondent,
        })

    # If no article-violation pairs were parsed, include as unknown
    if not results and full_text:
        results.append({
            "item_id": item_id,
            "case_name": case_name,
            "article": "",
            "article_full": "",
            "violation_label": "unknown",
            "full_case_text": full_text,
            "decision_date": decision_date,
            "respondent": respondent,
        })

    return results


# ── Main scraper ───────────────────────────────────────────────────────────────

def load_existing_item_ids(output_path: str) -> set[str]:
    """Load item_ids from an existing output file for resume capability."""
    if not os.path.exists(output_path):
        return set()
    try:
        with open(output_path) as f:
            data = json.load(f)
        ids = {item.get("item_id", "") for item in data}
        return ids
    except (json.JSONDecodeError, IOError):
        return set()


def scrape_hudoc(
    since: str = "",
    until: str = "",
    country: str = "",
    language: str = "ENG",
    output_path: str = "data/raw/hudoc_cases.json",
    resume: bool = False,
    dry_run: bool = False,
    page_size: int = PAGE_SIZE,
    delay: float = REQUEST_DELAY,
) -> list[dict]:
    """
    Main scraping function.

    1. Query HUDOC search API to get metadata for matching judgments
    2. Download full text for each judgment
    3. Parse conclusions into case-article pairs
    4. Output JSON compatible with verdict_leakage_removal.py

    Returns list of case-article pair dicts.
    """
    session = requests.Session()

    # Build search query
    query = build_search_query(
        since=since, until=until, country=country, language=language,
    )

    # Load existing IDs for resume
    existing_ids = set()
    existing_records = []
    if resume:
        existing_ids = load_existing_item_ids(output_path)
        if existing_ids:
            print(f"Resume mode: found {len(existing_ids)} already-downloaded item_ids")
            try:
                with open(output_path) as f:
                    existing_records = json.load(f)
            except (json.JSONDecodeError, IOError):
                existing_records = []

    # Phase 1: Search for matching judgments
    print(f"\nSearching HUDOC...")
    print(f"  Query: {query}")
    print(f"  Date range: {since or 'any'} to {until or 'any'}")
    print(f"  Country: {country or 'all'}")
    print(f"  Language: {language}")

    first_page = search_hudoc(session, query, start=0, length=page_size)
    total_count = first_page.get("resultcount", 0)
    print(f"  Total matching judgments: {total_count:,}")

    if total_count == 0:
        print("  No judgments found. Exiting.")
        return []

    # Collect all metadata
    all_metadata = []
    results = first_page.get("results", [])
    for r in results:
        meta = parse_search_result(r)
        if meta["item_id"]:
            all_metadata.append(meta)

    # Paginate through remaining results
    fetched = len(results)
    while fetched < total_count:
        time.sleep(delay)
        page = search_hudoc(session, query, start=fetched, length=page_size)
        results = page.get("results", [])
        if not results:
            break
        for r in results:
            meta = parse_search_result(r)
            if meta["item_id"]:
                all_metadata.append(meta)
        fetched += len(results)
        print(f"\r  Fetched metadata: {fetched:,}/{total_count:,}", end="", flush=True)

    print(f"\n  Collected metadata for {len(all_metadata)} judgments")

    # Filter out already-downloaded
    if existing_ids:
        before = len(all_metadata)
        all_metadata = [m for m in all_metadata if m["item_id"] not in existing_ids]
        skipped = before - len(all_metadata)
        print(f"  Skipping {skipped} already-downloaded judgments")
        print(f"  New judgments to download: {len(all_metadata)}")

    if dry_run:
        print("\n  [DRY RUN] Skipping full text download. Outputting metadata only.")
        # Output metadata without full text
        output_records = []
        for meta in all_metadata:
            output_records.append({
                "item_id": meta["item_id"],
                "case_name": meta["case_name"],
                "article": ";".join(meta["articles"]),
                "violation_label": "",
                "full_case_text": "",
                "decision_date": meta["decision_date"],
                "respondent": meta["respondent"],
                "application_number": meta["application_number"],
                "conclusion": meta["conclusion"],
                "ecli": meta["ecli"],
            })
        return existing_records + output_records

    # Phase 2: Download full text for each judgment
    print(f"\nDownloading full texts ({len(all_metadata)} documents)...")
    all_case_pairs = list(existing_records)
    downloaded = 0
    failed = 0
    skipped_short = 0

    for i, meta in enumerate(all_metadata):
        item_id = meta["item_id"]

        # Rate limiting
        if i > 0:
            time.sleep(delay)

        # Download full text
        full_text = fetch_document_text(session, item_id)
        downloaded += 1

        if not full_text:
            failed += 1
            print(f"\r  Progress: {i+1}/{len(all_metadata)} | Downloaded: {downloaded} | Failed: {failed}", end="", flush=True)
            continue

        if len(full_text) < 500:
            skipped_short += 1
            print(f"\r  Progress: {i+1}/{len(all_metadata)} | Downloaded: {downloaded} | Short: {skipped_short}", end="", flush=True)
            continue

        # Parse into case-article pairs
        pairs = parse_conclusion_to_pairs(
            conclusion=meta["conclusion"],
            item_id=item_id,
            case_name=meta["case_name"],
            decision_date=meta["decision_date"],
            respondent=meta["respondent"],
            full_text=full_text,
        )
        all_case_pairs.extend(pairs)

        if (i + 1) % 10 == 0 or (i + 1) == len(all_metadata):
            print(f"\r  Progress: {i+1}/{len(all_metadata)} | Pairs: {len(all_case_pairs)} | Failed: {failed} | Short: {skipped_short}  ", end="", flush=True)

        # Checkpoint: save every 100 documents
        if (i + 1) % 100 == 0:
            _save_checkpoint(all_case_pairs, output_path)

    print()

    # Final stats
    new_pairs = len(all_case_pairs) - len(existing_records)
    print(f"\n  Download complete:")
    print(f"    Total documents attempted: {len(all_metadata)}")
    print(f"    Failed downloads: {failed}")
    print(f"    Skipped (too short): {skipped_short}")
    print(f"    New case-article pairs: {new_pairs}")
    print(f"    Total case-article pairs: {len(all_case_pairs)}")

    return all_case_pairs


def _save_checkpoint(records: list[dict], output_path: str):
    """Save intermediate checkpoint to disk."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    checkpoint_path = output_path + ".checkpoint"
    with open(checkpoint_path, "w") as f:
        json.dump(records, f, ensure_ascii=False)
    os.replace(checkpoint_path, output_path)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Download ECtHR judgments from HUDOC for the LiveHumanRightsBench pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --since 2025-01-01 --output data/raw/hudoc_2025.json
  %(prog)s --since 2024-01-01 --until 2025-01-01 --country UKR --output data/raw/hudoc_ukr_2024.json
  %(prog)s --since 2025-06-01 --dry-run --output data/raw/hudoc_meta.json
  %(prog)s --since 2025-01-01 --output data/raw/hudoc_2025.json --resume
""",
    )
    parser.add_argument(
        "--since", required=True,
        help="Fetch judgments from this date onwards (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--until", default="",
        help="Fetch judgments before this date (YYYY-MM-DD, exclusive)",
    )
    parser.add_argument(
        "--country", default="",
        help="Respondent state ISO code (e.g., UKR, GBR, FRA). Empty = all countries",
    )
    parser.add_argument(
        "--language", default="ENG",
        help="Language ISO code (default: ENG)",
    )
    parser.add_argument(
        "--output", default="data/raw/hudoc_cases.json",
        help="Output JSON file path",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume: skip item_ids already present in the output file",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Metadata only - do not download full text",
    )
    parser.add_argument(
        "--page-size", type=int, default=PAGE_SIZE,
        help=f"Number of results per search page (default: {PAGE_SIZE})",
    )
    parser.add_argument(
        "--delay", type=float, default=REQUEST_DELAY,
        help=f"Delay between requests in seconds (default: {REQUEST_DELAY})",
    )

    args = parser.parse_args()

    # Validate dates
    for date_arg, name in [(args.since, "--since"), (args.until, "--until")]:
        if date_arg:
            try:
                datetime.strptime(date_arg, "%Y-%m-%d")
            except ValueError:
                print(f"ERROR: {name} must be in YYYY-MM-DD format, got: {date_arg}", file=sys.stderr)
                sys.exit(1)

    start_time = time.time()

    records = scrape_hudoc(
        since=args.since,
        until=args.until,
        country=args.country,
        language=args.language,
        output_path=args.output,
        resume=args.resume,
        dry_run=args.dry_run,
        page_size=args.page_size,
        delay=args.delay,
    )

    if records:
        # Save final output
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)

        elapsed = time.time() - start_time
        print(f"\nSaved {len(records)} case-article pairs to {args.output}")
        print(f"Elapsed time: {elapsed:.0f}s ({elapsed/60:.1f}m)")

        # Summary stats
        from collections import Counter
        countries = Counter(r.get("respondent", "?") for r in records)
        labels = Counter(r.get("violation_label", "?") for r in records)
        dates = sorted(set(r.get("decision_date", "") for r in records if r.get("decision_date")))

        print(f"\nSummary:")
        print(f"  Unique cases: {len(set(r['item_id'] for r in records))}")
        print(f"  Case-article pairs: {len(records)}")
        if dates:
            print(f"  Date range: {dates[0]} to {dates[-1]}")
        print(f"  Violation labels: {dict(labels)}")
        if len(countries) <= 20:
            print(f"  Countries: {dict(countries.most_common())}")
        else:
            print(f"  Countries: {len(countries)} ({', '.join(c for c, _ in countries.most_common(10))}, ...)")
    else:
        print("\nNo records to save.")


if __name__ == "__main__":
    main()
