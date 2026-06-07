#!/usr/bin/env python3
"""
Verdict-Leakage Removal Pipeline for ECtHR Cases

Three-stage pipeline as described in the paper:
  1. Pattern-based truncation -- remove text after markers introducing the Court's assessment
  2. Dual LLM verification -- two models independently check for remaining verdict leakage
  3. Repair or exclude -- fix leaking cases or drop those that are too short

Usage:
  # From HUDOC database (prod)
  python scripts/verdict_leakage_removal.py \
    --source db \
    --db "dbname=secondlayer_prod user=secondlayer host=172.18.0.13" \
    --country UKR \
    --output data/processed/echr_cases_ukr_clean.json \
    --workers 10

  # From existing JSON (HUDOC export)
  python scripts/verdict_leakage_removal.py \
    --source json \
    --input data/raw/echr_cases_ukr.json \
    --output data/processed/echr_cases_ukr_clean.json

  # Dry run (stage 1 only, no LLM calls)
  python scripts/verdict_leakage_removal.py \
    --source db --country UKR --stage1-only --output /dev/stdout
"""

import argparse
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import Optional

try:
    import openai
except ImportError:
    openai = None

try:
    import boto3
except ImportError:
    boto3 = None

# ── Stage 1: Pattern-based truncation ────────────────────────────────────────

# Markers that introduce the Court's own assessment/conclusions.
# Text at or after these markers is removed.
COURT_ASSESSMENT_MARKERS = [
    # Section headers (uppercase, often standalone)
    r"^[IVX]+\.\s*THE COURT['’]?S ASSESSMENT",
    r"^[IVX]+\.\s*MERITS",
    r"^[IVX]+\.\s*THE MERITS",
    r"^THE COURT['’]?S ASSESSMENT",
    r"^MERITS$",
    # Operative provisions and final sections
    r"^FOR THESE REASONS",
    r"^OPERATIVE PROVISIONS",
    # Article 41 (just satisfaction) -- always post-verdict
    r"^[IVX]+\.\s*APPLICATION OF ARTICLE 41",
    r"^APPLICATION OF ARTICLE 41",
    r"^ARTICLE 41",
    # Common assessment openers within paragraphs
    r"The Court will now assess",
    r"The Court will first examine",
    r"The Court must therefore determine",
    r"The Court will examine the merits",
    r"The Court notes at the outset that",
    r"The Court observes that the parties",
    r"Turning to the merits",
    r"As to the merits",
]

# Conclusory patterns that directly reveal the verdict
CONCLUSORY_PATTERNS = [
    r"(?:finds|concludes|holds)\s+that\s+there\s+has\s+(?:been|not been)\s+(?:a\s+)?violation",
    r"(?:finds|concludes|holds)\s+that\s+there\s+has\s+(?:been|not been)\s+(?:a\s+)?breach",
    r"there\s+has\s+(?:been|not been)\s+(?:a\s+)?violation\s+of\s+Article",
    r"there\s+has\s+(?:been|not been)\s+(?:a\s+)?breach\s+of\s+Article",
    r"no\s+violation\s+of\s+Article",
    r"violated?\s+Article",
    r"Accordingly,?\s+the\s+Court\s+(?:finds|concludes|holds)",
    r"It\s+follows\s+that\s+there\s+has\s+(?:been|not been)",
    r"In\s+the\s+light\s+of\s+the\s+above.*(?:finds|concludes)",
    r"The\s+Court\s+therefore\s+concludes",
    r"dismisses?\s+the\s+(?:application|complaint)",
    r"Holds\s+that",
    r"Decides\s+that",
]

COMPILED_MARKERS = [re.compile(m, re.IGNORECASE | re.MULTILINE) for m in COURT_ASSESSMENT_MARKERS]
COMPILED_CONCLUSORY = [re.compile(p, re.IGNORECASE) for p in CONCLUSORY_PATTERNS]

MIN_VERDICT_FREE_LENGTH = 500


@dataclass
class CaseRecord:
    item_id: str
    case_name: str
    article: str
    violation_label: str
    full_case_text: str
    full_case_text_no_verdict: str = ""
    verdict_removal_method: str = ""
    original_length: int = 0
    verdict_free_length: int = 0
    retention_percentage: float = 0.0
    leakage_check_primary: str = ""
    leakage_check_secondary: str = ""
    excluded: bool = False
    exclusion_reason: str = ""


def stage1_truncate(text: str) -> tuple[str, str]:
    """
    Pattern-based truncation: find the earliest Court assessment marker
    and truncate everything from that point onward.

    Returns (truncated_text, method_used).
    """
    earliest_pos = len(text)
    method = "no_truncation"

    for pattern in COMPILED_MARKERS:
        match = pattern.search(text)
        if match and match.start() < earliest_pos:
            earliest_pos = match.start()
            method = f"court_assessment_start"

    if earliest_pos < len(text):
        truncated = text[:earliest_pos].rstrip()
        return truncated, method

    # Fallback: look for conclusory patterns and truncate before the sentence
    for pattern in COMPILED_CONCLUSORY:
        match = pattern.search(text)
        if match:
            sentence_start = text.rfind('.', 0, match.start())
            if sentence_start > 0:
                truncated = text[:sentence_start + 1].rstrip()
                return truncated, "aggressive_fix: conclusory_pattern"

    return text, "no_truncation"


# ── Stage 2: LLM-based verification ─────────────────────────────────────────

LEAKAGE_CHECK_PROMPT = """You are a legal expert reviewing a case document from the European Court of Human Rights (ECtHR).

The following text is supposed to contain ONLY the factual background, procedural history, parties' arguments, cited legal standards, and relevant domestic law. It should NOT contain:
- The Court's own legal reasoning or assessment
- The Court's findings or conclusions
- Whether the Court found a violation or not
- Operative provisions or dispositif
- Article 41 (just satisfaction) analysis

Review the text carefully and determine:
1. Does this text reveal the Court's verdict (violation or no violation)?
2. If yes, quote the specific sentence(s) that leak the verdict.

Text to review:
---
{text}
---

Respond in this exact JSON format:
{{"leaks_verdict": true/false, "leaking_sentences": ["sentence1", "sentence2"] or [], "confidence": "high"/"medium"/"low"}}

Respond with ONLY the JSON, no other text."""


def check_leakage_bedrock(bedrock_client, model_id: str, text: str, max_chars: int = 50000) -> dict:
    """Check a single case for verdict leakage using AWS Bedrock."""
    truncated_text = text[:max_chars] if len(text) > max_chars else text
    prompt = LEAKAGE_CHECK_PROMPT.format(text=truncated_text)
    try:
        response = bedrock_client.converse(
            modelId=model_id,
            system=[{"text": "You are a legal document reviewer. Respond only in valid JSON."}],
            messages=[{"role": "user", "content": [{"text": prompt}]}],
            inferenceConfig={"temperature": 0.0, "maxTokens": 500},
        )
        content = response["output"]["message"]["content"][0]["text"].strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content)
    except Exception as e:
        return {"leaks_verdict": None, "error": str(e), "leaking_sentences": []}


def check_leakage_sync(client, model: str, text: str, max_chars: int = 50000) -> dict:
    """Check a single case for verdict leakage using OpenAI-compatible API."""
    truncated_text = text[:max_chars] if len(text) > max_chars else text
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a legal document reviewer. Respond only in valid JSON."},
                {"role": "user", "content": LEAKAGE_CHECK_PROMPT.format(text=truncated_text)},
            ],
            temperature=0.0,
            max_tokens=500,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content)
    except Exception as e:
        return {"leaks_verdict": None, "error": str(e), "leaking_sentences": []}


def stage2_verify(case: CaseRecord, verifiers: list[tuple], use_bedrock: bool = False) -> CaseRecord:
    """
    Multi-model LLM verification: N models independently check for verdict leakage.
    If ANY model flags leakage, attempt repair.

    verifiers: list of (client, model_id, label) tuples
    """
    text = case.full_case_text_no_verdict
    if not text or case.excluded:
        return case

    check_fn = check_leakage_bedrock if use_bedrock else check_leakage_sync

    results = {}
    all_leaking_sentences = []
    any_leaks = False

    for client, model_id, label in verifiers:
        result = check_fn(client, model_id, text)
        results[label] = result
        if result.get("leaks_verdict", False):
            any_leaks = True
            all_leaking_sentences.extend(result.get("leaking_sentences", []))

    case.leakage_check_primary = json.dumps(results.get("primary", results.get(list(results.keys())[0], {})))
    case.leakage_check_secondary = json.dumps({k: v for k, v in results.items() if k != "primary"})

    if any_leaks:
        case = stage3_repair(case, all_leaking_sentences)

    return case


# ── Stage 3: Repair or exclude ───────────────────────────────────────────────

def stage3_repair(case: CaseRecord, leaking_sentences: list[str]) -> CaseRecord:
    """
    Attempt to repair a leaking case by removing identified leaking sentences.
    If the result is too short, exclude the case.
    """
    text = case.full_case_text_no_verdict

    for sentence in leaking_sentences:
        if not sentence:
            continue
        # Try exact removal
        escaped = re.escape(sentence.strip())
        new_text = re.sub(escaped, "", text, count=1)
        if new_text != text:
            text = new_text
            case.verdict_removal_method += f"; surgical_fix: removed leaking sentence"
            continue

        # Fuzzy: find and truncate before the sentence
        idx = text.lower().find(sentence.lower()[:50])
        if idx > 0:
            sentence_start = text.rfind('.', 0, idx)
            if sentence_start > 0:
                text = text[:sentence_start + 1].rstrip()
                case.verdict_removal_method += f"; surgical_fix: truncated before leaking sentence"

    # Also scan for any remaining conclusory patterns
    for pattern in COMPILED_CONCLUSORY:
        match = pattern.search(text)
        if match:
            sentence_start = text.rfind('.', 0, match.start())
            if sentence_start > 0:
                text = text[:sentence_start + 1].rstrip()
                case.verdict_removal_method += "; conclusory_pattern_cleanup"

    text = text.strip()

    if len(text) < MIN_VERDICT_FREE_LENGTH:
        case.excluded = True
        case.exclusion_reason = "too_short_after_repair"
    else:
        case.full_case_text_no_verdict = text
        case.verdict_free_length = len(text)
        case.retention_percentage = len(text) / case.original_length * 100 if case.original_length else 0

    return case


# ── Data loading ─────────────────────────────────────────────────────────────

def load_from_db(db_url: str, country: str = "", cutoff_date: str = "", lang: str = "ENG") -> list[CaseRecord]:
    """Load ECtHR cases from PostgreSQL echr_cases table."""
    import psycopg2

    conn = psycopg2.connect(db_url)
    with conn.cursor() as cur:
        conditions = ["full_text IS NOT NULL", "length(full_text) > 500", "conclusion IS NOT NULL"]
        params = []

        if country:
            conditions.append("respondent = %s")
            params.append(country)
        if cutoff_date:
            conditions.append("kp_date >= %s")
            params.append(cutoff_date)
        if lang:
            conditions.append("language_iso = %s")
            params.append(lang)

        # Exclude translations
        conditions.append("doc_name NOT ILIKE %s")
        params.append("%Translation%")

        query = f"SELECT item_id, doc_name, conclusion, full_text, respondent FROM echr_cases WHERE {' AND '.join(conditions)} ORDER BY item_id"
        cur.execute(query, params if params else None)
        rows = cur.fetchall()
    conn.close()

    cases = []
    for row in rows:
        item_id, doc_name, conclusion, full_text = row[0], row[1], row[2], row[3]
        respondent = row[4] if len(row) > 4 else ""
        articles_violations = parse_conclusion(conclusion or "")
        for article, label in articles_violations:
            cases.append(CaseRecord(
                item_id=item_id,
                case_name=doc_name or "",
                article=article,
                violation_label=label,
                full_case_text=full_text,
                original_length=len(full_text),
            ))

    return cases


def parse_conclusion(conclusion: str) -> list[tuple[str, str]]:
    """
    Parse ECtHR conclusion field into (article, violation_label) pairs.

    Examples:
      "Violation of Article 3" -> [("3", "violation")]
      "No violation of Article 8; Violation of Article 3" -> [("8", "no_violation"), ("3", "violation")]
    """
    results = []
    if not conclusion:
        return results

    parts = re.split(r'[;]', conclusion)
    for part in parts:
        part = part.strip()
        article_match = re.search(r'Article\s+([\dP][\d-]*)', part, re.IGNORECASE)
        if not article_match:
            continue
        article = article_match.group(1)

        if re.search(r'no\s+violation', part, re.IGNORECASE):
            label = "no_violation"
        elif re.search(r'violation', part, re.IGNORECASE):
            label = "violation"
        else:
            continue

        results.append((article, label))

    return results


def load_from_json(path: str) -> list[CaseRecord]:
    """Load cases from a JSON file."""
    with open(path) as f:
        data = json.load(f)

    cases = []
    for item in data:
        cases.append(CaseRecord(
            item_id=item.get("item_id", ""),
            case_name=item.get("case_name", ""),
            article=item.get("article", ""),
            violation_label=item.get("violation_label", ""),
            full_case_text=item.get("full_case_text", item.get("case_text", "")),
            original_length=len(item.get("full_case_text", item.get("case_text", ""))),
        ))
    return cases


# ── Main pipeline ────────────────────────────────────────────────────────────

def run_pipeline(cases: list[CaseRecord], args) -> list[CaseRecord]:
    """Run the full 3-stage pipeline."""

    print(f"\n{'='*60}")
    print(f"Stage 1: Pattern-based truncation ({len(cases)} cases)")
    print(f"{'='*60}")

    for case in cases:
        truncated, method = stage1_truncate(case.full_case_text)
        case.full_case_text_no_verdict = truncated
        case.verdict_removal_method = method
        case.verdict_free_length = len(truncated)
        case.retention_percentage = len(truncated) / case.original_length * 100 if case.original_length else 0

    # Stats
    truncated_count = sum(1 for c in cases if c.verdict_removal_method != "no_truncation")
    avg_retention = sum(c.retention_percentage for c in cases) / len(cases) if cases else 0
    print(f"  Truncated: {truncated_count}/{len(cases)}")
    print(f"  Average retention: {avg_retention:.1f}%")

    # Exclude cases that became too short
    for case in cases:
        if case.verdict_free_length < MIN_VERDICT_FREE_LENGTH:
            case.excluded = True
            case.exclusion_reason = "too_short_after_truncation"

    excluded = sum(1 for c in cases if c.excluded)
    print(f"  Excluded (too short): {excluded}")

    if args.stage1_only:
        print("\n  --stage1-only: skipping LLM verification")
        return cases

    # Stage 2+3: LLM verification and repair
    # Only verify untruncated cases (truncated ones are already clean)
    needs_verification = [c for c in cases if not c.excluded and c.verdict_removal_method == "no_truncation"]
    already_clean = [c for c in cases if not c.excluded and c.verdict_removal_method != "no_truncation"]
    print(f"\n{'='*60}")
    print(f"Stage 2+3: LLM verification & repair ({len(needs_verification)} untruncated cases)")
    print(f"  Already truncated (skipping): {len(already_clean)}")
    print(f"{'='*60}")

    if not needs_verification:
        print("  No cases need LLM verification.")
        return cases

    use_bedrock = args.backend == "bedrock"

    if use_bedrock:
        if not boto3:
            print("  ERROR: boto3 not installed. pip install boto3")
            return cases
        bedrock = boto3.client("bedrock-runtime", region_name=args.aws_region)
        verifiers = [(bedrock, m.strip(), f"v{i}") for i, m in enumerate(args.models.split(","))]
    else:
        api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENROUTER_API_KEY", "")
        if not api_key:
            print("  WARNING: No API key found (OPENAI_API_KEY or OPENROUTER_API_KEY)")
            print("  Skipping LLM verification. Set API key to enable.")
            return cases
        use_openrouter = bool(os.environ.get("OPENROUTER_API_KEY"))
        base_url = "https://openrouter.ai/api/v1" if use_openrouter else None
        client_kwargs = {"api_key": api_key}
        if base_url:
            client_kwargs["base_url"] = base_url
        client = openai.OpenAI(**client_kwargs)
        verifiers = [(client, m.strip(), f"v{i}") for i, m in enumerate(args.models.split(","))]

    # Set first as "primary" label
    verifiers[0] = (verifiers[0][0], verifiers[0][1], "primary")

    print(f"  Backend: {'bedrock' if use_bedrock else 'openai/openrouter'}")
    print(f"  Verifiers ({len(verifiers)}):")
    for client, model_id, label in verifiers:
        print(f"    [{label}] {model_id}")
    print(f"  Workers: {args.workers}")

    done = 0

    def process_case(case):
        return stage2_verify(case, verifiers, use_bedrock)

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_case, c): c for c in needs_verification}
        for future in as_completed(futures):
            result = future.result()
            idx = cases.index(futures[future])
            cases[idx] = result
            done += 1
            if done % 10 == 0 or done == len(needs_verification):
                flagged = sum(1 for c in cases if c.leakage_check_primary and
                              json.loads(c.leakage_check_primary).get("leaks_verdict"))
                print(f"\r  Verified: {done}/{len(needs_verification)} | Flagged: {flagged}", end="", flush=True)

    print()

    final_excluded = sum(1 for c in cases if c.excluded)
    final_count = len(cases) - final_excluded
    print(f"\n  Final dataset: {final_count} cases ({final_excluded} excluded)")

    return cases


def main():
    parser = argparse.ArgumentParser(description="Verdict-leakage removal pipeline for ECtHR cases")
    parser.add_argument("--source", choices=["db", "json"], default="db")
    parser.add_argument("--db", default="dbname=secondlayer_prod user=secondlayer password=secondlayer host=172.18.0.13")
    parser.add_argument("--country", default="", help="Respondent country code (empty = all countries)")
    parser.add_argument("--lang", default="ENG", help="Language ISO code filter (ENG, FRE, etc.)")
    parser.add_argument("--cutoff-date", default="", help="Only include cases after this date")
    parser.add_argument("--input", help="Input JSON file (when --source json)")
    parser.add_argument("--output", default="data/processed/echr_cases_clean.json")
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--backend", choices=["openai", "bedrock"], default="bedrock", help="LLM backend")
    parser.add_argument("--aws-region", default="us-east-1", help="AWS region for Bedrock")
    parser.add_argument("--models", default="us.anthropic.claude-sonnet-4-6-20250514-v1:0,us.anthropic.claude-haiku-4-5-20251001-v1:0,amazon.nova-pro-v1:0,qwen.qwen3-32b-v1:0",
                        help="Comma-separated verifier model IDs")
    parser.add_argument("--stage1-only", action="store_true", help="Only run pattern truncation (no LLM calls)")
    parser.add_argument("--min-length", type=int, default=500, help="Minimum verdict-free text length")
    args = parser.parse_args()

    global MIN_VERDICT_FREE_LENGTH
    MIN_VERDICT_FREE_LENGTH = args.min_length

    # Load cases
    if args.source == "db":
        print(f"Loading cases from database (country={args.country or 'ALL'}, lang={args.lang}, cutoff={args.cutoff_date or 'none'})...")
        cases = load_from_db(args.db, args.country, args.cutoff_date, args.lang)
    else:
        print(f"Loading cases from {args.input}...")
        cases = load_from_json(args.input)

    print(f"Loaded {len(cases)} case-article pairs")

    if not cases:
        print("No cases to process.")
        return

    # Run pipeline
    cases = run_pipeline(cases, args)

    # Save results
    output_cases = [asdict(c) for c in cases if not c.excluded]
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(output_cases, f, indent=2, ensure_ascii=False)

    print(f"\nSaved {len(output_cases)} clean cases to {args.output}")

    # Summary
    from collections import Counter
    labels = Counter(c.violation_label for c in cases if not c.excluded)
    articles = Counter(c.article for c in cases if not c.excluded)
    methods = Counter(c.verdict_removal_method.split(";")[0].strip() for c in cases if not c.excluded)
    print(f"\nViolation labels: {dict(labels)}")
    print(f"Articles: {dict(articles)}")
    print(f"Removal methods: {dict(methods)}")


if __name__ == "__main__":
    main()
