#!/usr/bin/env python3
"""Build the extractive control arm: summaries that are verbatim source paragraphs.

Writes the same file shape as scripts/build_summaries.py, so every runner consumes it
unchanged -- the arm is run by pointing --summaries at this file instead.

    python scripts/build_extractive.py \
      --cases data/processed/livehrb_1k.json \
      --summarizer x-ai/grok-4.6 --api-key-env OPENROUTER_API_KEY \
      --out data/processed/summaries_extractive.json

What it buys: the abstractive arm confounds omission with invention. Here the model
may only choose paragraphs, so any effect is omission alone, and the omission is the
exact list of paragraph numbers left out rather than something an entailment judge
estimated.
"""

import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))
from checkpoint import Checkpoint          # noqa: E402
from scoring import MAX_CASE_CHARS         # noqa: E402
from summaries import asserts_outcome      # noqa: E402
from extractive import (SELECT_TEMPLATE, assemble, is_verbatim,   # noqa: E402
                        omitted, parse_selection, split_paragraphs)

ATTEMPTS = 3


def extract(client, model, case, article, target_words, max_tokens):
    paras = split_paragraphs(case["text"])
    if len(paras) < 4:
        return None, "ERROR: fewer than four numbered paragraphs", None, {"prompt": 0, "completion": 0}
    numbered = "\n\n".join(f"[{n}] {body}" for n, body in paras)[:MAX_CASE_CHARS]
    prompt = SELECT_TEMPLATE.format(case_name=case["case_name"], numbered=numbered,
                                    article=article, target_words=target_words)
    valid = {n for n, _ in paras}
    usage = {"prompt": 0, "completion": 0}
    last = "ERROR: no attempt made"
    for attempt in range(ATTEMPTS):
        try:
            resp = client.chat.completions.create(
                model=model, messages=[{"role": "user", "content": prompt}],
                temperature=1.0, max_tokens=max_tokens)
            if resp.usage:
                usage["prompt"] += resp.usage.prompt_tokens or 0
                usage["completion"] += resp.usage.completion_tokens or 0
            chosen = parse_selection((resp.choices[0].message.content or ""), valid)
            if not chosen:
                last = "ERROR: no valid paragraph numbers in reply"
                continue
            text = assemble(paras, chosen)
            # The guarantee the whole arm rests on, checked rather than assumed.
            if not is_verbatim(text, case["text"]):
                last = "ERROR: assembled extract is not verbatim"
                continue
            if asserts_outcome(text, case["text"]):
                # Cannot happen for a true extract, since the source is the source.
                # Kept as a tripwire in case assembly ever stops being verbatim.
                last = "ERROR: extract asserts an outcome"
                continue
            return text, None, omitted(paras, chosen), usage
        except Exception as e:
            last = f"ERROR: {e}"
            time.sleep(2 * (attempt + 1))
    return None, last, None, usage


def main():
    p = argparse.ArgumentParser(description="Build the extractive control summaries")
    p.add_argument("--cases", required=True)
    p.add_argument("--summarizer", required=True)
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--out", required=True)
    p.add_argument("--versions", type=int, default=1,
                   help="extractive selection is far less variable than free writing, "
                        "so one version is the default")
    p.add_argument("--target-words", type=int, default=500)
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=4000)
    p.add_argument("--limit", type=int)
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    key = os.environ.get(args.api_key_env, "")
    if not key:
        sys.exit(f"ERROR: set {args.api_key_env}")

    instances = json.load(open(args.cases))
    judgments, article_of = {}, {}
    for c in instances:
        if c["item_id"] in judgments:
            continue
        text = c.get("full_case_text_no_verdict") or c.get("verdict_free_text") or ""
        judgments[c["item_id"]] = {"item_id": c["item_id"], "case_name": c["case_name"],
                                   "text": text[:MAX_CASE_CHARS]}
        article_of[c["item_id"]] = c["article"]
    cases = list(judgments.values())
    if args.limit:
        cases = cases[:args.limit]

    print(f"Selector: {args.summarizer}   judgments: {len(cases)}   "
          f"versions: {args.versions}   target: {args.target_words} words\n")

    ckpt = Checkpoint(args.out + ".jsonl", enabled=not args.no_resume)
    if ckpt.resumed:
        print(f"Resuming: {ckpt.resumed} already recorded\n")
    client = OpenAI(base_url=args.base_url, api_key=key)

    units = [(c, v) for c in cases for v in range(args.versions)
             if not ckpt.done(Checkpoint.key("extract", c["item_id"], "", v))]
    done = failed = 0
    tot = {"prompt": 0, "completion": 0}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(extract, client, args.summarizer, c,
                            article_of[c["item_id"]], args.target_words, args.max_tokens): (c, v)
                for c, v in units}
        for fut in as_completed(futs):
            case, v = futs[fut]
            text, err, drop, usage = fut.result()
            tot["prompt"] += usage["prompt"]; tot["completion"] += usage["completion"]
            done += 1
            if text:
                ckpt.record(Checkpoint.key("extract", case["item_id"], "", v),
                            {"item_id": case["item_id"], "case_name": case["case_name"],
                             "version": v, "summary": text, "omitted_paragraphs": drop})
            else:
                failed += 1
                print(f"\n  {case['case_name'][:40]} v{v}: {err[:90]}")
            print(f"\r  {done}/{len(units)} | failed {failed}", end="", flush=True)
    print()
    ckpt.close()

    by_case = {}
    for row in ckpt.rows():
        by_case.setdefault(row["item_id"], {})[row["version"]] = row
    summaries = {k: [v[i]["summary"] for i in range(args.versions) if i in v]
                 for k, v in by_case.items()}
    dropped = {k: v[0].get("omitted_paragraphs") for k, v in by_case.items() if 0 in v}
    complete = sum(1 for v in summaries.values() if len(v) == args.versions)

    json.dump({"summarizer": args.summarizer, "versions": args.versions,
               "mode": "extractive", "target_words": args.target_words,
               "n_judgments": len(cases), "n_complete": complete,
               "prompt_tokens": tot["prompt"], "completion_tokens": tot["completion"],
               "omitted_paragraphs": dropped, "summaries": summaries},
              open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"\nWrote {args.out}")
    print(f"  {complete}/{len(cases)} judgments complete")
    print(f"  tokens: {tot['prompt']:,} in, {tot['completion']:,} out")


if __name__ == "__main__":
    main()
