#!/usr/bin/env python3
"""Measure whether a summary keeps the facts the Court relied on.

    python scripts/build_atomic_coverage.py \
        --full-texts data/processed/annot_full_texts.csv \
        --summaries data/processed/summaries_grok46.json \
        --variant abstractive \
        --model google/gemini-3.5-flash \
        --api-key-env OPENROUTER_API_KEY \
        --out data/experiments/coverage_atomic

Run it once per summary variant; the two runs share nothing but the source, so the
abstractive and extractive numbers are comparable claim for claim.

The instrument is judge-independent: coverage is a property of a (source, summary)
pair and never touches what the eight models answered, so it is paid once no matter
how wide the roster gets. The one hard constraint on `--model` is that it must not be
the summariser, for the same reason a judge must not read its own writing. Being in
the judge roster is harmless here, since the metric never sees a judge's verdict.

Claims are extracted per paragraph so each keeps the number it came from, which is
what makes the headline possible: the Court's assessment back-references the
paragraphs it rested on, so "relied upon" is read off the judgment rather than
decided by us. That is the part of Yu Fan's objection that can be answered.
"""

import argparse
import json
import os
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import csv

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "experiments"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from atomic import (EXTRACT_TEMPLATE, VERIFY_TEMPLATE, coverage,  # noqa: E402
                    number_claims, parse_claims, parse_verdicts)
from checkpoint import Checkpoint                                  # noqa: E402
from build_annotation import BACKREF, LAW, fact_paragraphs, split_paragraphs, \
    is_heading_only, assessment_region                             # noqa: E402
from summaries import load_summaries                               # noqa: E402

csv.field_size_limit(10 ** 9)

ATTEMPTS = 4
MAX_TOKENS = 4000
# Verifying claim by claim would be 50-odd calls per judgment; batching them into one
# call per (judgment, variant) is what makes this affordable. The cost of batching is
# that a truncated reply silently scores its tail unsupported, which `parse_verdicts`
# refuses by returning None and which is retried here rather than recorded.
BATCH = 40


def complete(client, model, prompt, max_tokens=MAX_TOKENS):
    limit = max_tokens
    for attempt in range(ATTEMPTS):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0,
                max_completion_tokens=limit,
            )
            content = (resp.choices[0].message.content or "").strip()
            if content:
                return content
            limit *= 2          # an empty body at a tight ceiling is billed in full
        except Exception as exc:                      # noqa: BLE001
            if attempt == ATTEMPTS - 1:
                return "ERROR: %s" % exc
            time.sleep(min(2 ** attempt, 30) + random.uniform(0, 1))
    return "ERROR: empty after %d attempts" % ATTEMPTS


def relied_upon_numbers(full_text):
    """Paragraph numbers the Court's own assessment back-references."""
    marks = list(LAW.finditer(full_text))
    if not marks:
        return set()
    cut = marks[-1].start()
    numbers = set()
    for _, body in ((n, b) for n, b in split_paragraphs(assessment_region(full_text, cut))
                    if not is_heading_only(b)):
        for match in BACKREF.finditer(body):
            nums = [g for g in match.groups() if g]
            if len(nums) == 2 and any(t in match.group(0) for t in ("-", "–", " to ")):
                numbers.update(str(x) for x in range(int(nums[0]), int(nums[1]) + 1))
            else:
                numbers.update(nums)
    return numbers


def sample_paragraphs(facts, relied, per_side, seed):
    """Up to `per_side` relied-upon paragraphs and as many the Court never cites.

    Balanced on purpose. A relied-upon coverage rate on its own cannot distinguish a
    summariser that drops relied-upon facts from one that drops everything equally,
    and it is the difference between the two groups that answers the objection.
    """
    rng = random.Random("%s|%d" % (seed, per_side))
    cited = sorted((n for n in facts if n in relied), key=lambda n: int(n) if n.isdigit() else 0)
    other = sorted((n for n in facts if n not in relied), key=lambda n: int(n) if n.isdigit() else 0)
    rng.shuffle(cited)
    rng.shuffle(other)
    keep = cited[:per_side] + other[:per_side]
    return sorted(keep, key=lambda n: int(n) if n.isdigit() else 0)


def claims_for(client, model, full_text, item_id, ckpt, per_side, max_claims):
    """Atomic claims of the sampled fact paragraphs, tagged with relied-upon."""
    marks = list(LAW.finditer(full_text))
    if not marks:
        return []
    facts = fact_paragraphs(full_text[:marks[-1].start()])
    relied = relied_upon_numbers(full_text)
    chosen = sample_paragraphs(facts, relied, per_side, item_id)
    out = []
    for number in chosen:
        body = facts[number]
        key = ckpt.key("extract", item_id, number, "")
        row = ckpt.get(key)
        if row is None:
            reply = complete(client, model, EXTRACT_TEMPLATE.format(
                paragraph=body, max_claims=max_claims))
            row = {"claims": parse_claims(reply)}
            ckpt.record(key, {"item_id": item_id, "paragraph": number, **row})
        for claim in row["claims"]:
            out.append({"item_id": item_id, "paragraph": number, "claim": claim,
                        "relied_upon": number in relied})
    return out


def verify(client, model, summary, claims):
    """Support decisions for one judgment's claims, in batches."""
    verdicts = []
    for start in range(0, len(claims), BATCH):
        chunk = [c["claim"] for c in claims[start:start + BATCH]]
        answer = None
        for _ in range(2):                     # one retry: a short reply is a failure
            reply = complete(client, model, VERIFY_TEMPLATE.format(
                summary=summary, claims=number_claims(chunk), n=len(chunk)))
            answer = parse_verdicts(reply, len(chunk))
            if answer is not None:
                break
        if answer is None:
            return None
        verdicts.extend(answer)
    return verdicts


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--full-texts", required=True)
    p.add_argument("--summaries", required=True)
    p.add_argument("--variant", required=True, help="a label, e.g. abstractive or extractive")
    p.add_argument("--model", default="google/gemini-3.5-flash",
                   help="extractor and verifier; must not be the summariser")
    p.add_argument("--summarizer", default="x-ai/grok-4.6",
                   help="named only so the run refuses to grade its own writing")
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--out", required=True)
    p.add_argument("--limit", type=int, help="pilot on N judgments")
    p.add_argument("--workers", type=int, default=12)
    p.add_argument("--per-side", type=int, default=8,
                   help="relied-upon paragraphs per judgment, and as many uncited ones")
    p.add_argument("--max-claims", type=int, default=6, help="claims per paragraph")
    args = p.parse_args()

    if args.model == args.summarizer:
        sys.exit("--model is the summariser; coverage would be self-assessed")

    key = os.environ.get(args.api_key_env)
    if not key:
        sys.exit("%s is not set" % args.api_key_env)
    client = OpenAI(base_url=args.base_url, api_key=key)

    summaries, meta = load_summaries(args.summaries)
    rows = [r for r in csv.DictReader(open(args.full_texts)) if r.get("full_text")]
    rows = [r for r in rows if r["item_id"] in summaries]
    if args.limit:
        rows = rows[:args.limit]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    extract_ckpt = Checkpoint(out_dir / "claims.jsonl")
    result_ckpt = Checkpoint(out_dir / ("%s.jsonl" % args.variant))

    print("%d judgments, model %s, variant %s" % (len(rows), args.model, args.variant))

    def one(row):
        item_id = row["item_id"]
        claims = claims_for(client, args.model, row["full_text"], item_id, extract_ckpt,
                            args.per_side, args.max_claims)
        if not claims:
            return item_id, None
        verdicts = verify(client, args.model, summaries[item_id], claims)
        if verdicts is None:
            return item_id, None
        for claim, supported in zip(claims, verdicts):
            claim["supported"] = supported
        return item_id, claims

    failures = Counter()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, r): r for r in rows
                   if not result_ckpt.done(result_ckpt.key(args.variant, r["item_id"], "", ""))}
        for done in as_completed(futures):
            item_id, claims = done.result()
            if claims is None:
                failures[item_id] += 1
                continue
            overall, relied, n_all, n_relied = coverage(claims)
            result_ckpt.record(result_ckpt.key(args.variant, item_id, "", ""), {
                "item_id": item_id, "variant": args.variant,
                "coverage": overall, "coverage_relied_upon": relied,
                "n_claims": n_all, "n_relied_upon": n_relied,
                "claims": claims,
            })

    scored = result_ckpt.rows()
    pooled = [c for row in scored for c in row["claims"]]
    overall, relied, n_all, n_relied = coverage(pooled)
    print("\n  judgments scored: %d, failed: %d" % (len(scored), len(failures)))
    print("  claims: %d, of which relied upon by the Court: %d" % (n_all, n_relied))
    uncited = [c for c in pooled if not c["relied_upon"]]
    rate = lambda rows: (sum(1 for c in rows if c["supported"]) / len(rows)) if rows else None
    other = rate(uncited)
    print("  coverage, relied upon by the Court: %s over %d claims"
          % ("%.3f" % relied if relied is not None else "n/a", n_relied))
    print("  coverage, never cited:              %s over %d claims"
          % ("%.3f" % other if other is not None else "n/a", len(uncited)))
    if relied is not None and other is not None:
        print("  difference:                         %+.3f" % (relied - other))
    print("  coverage, pooled:                   %s over %d claims"
          % ("%.3f" % overall if overall is not None else "n/a", n_all))
    if meta.get("digest"):
        print("  summaries digest: %s" % meta["digest"])


if __name__ == "__main__":
    main()
