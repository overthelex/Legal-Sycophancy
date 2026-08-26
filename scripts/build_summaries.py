#!/usr/bin/env python3
"""Generate the RQ1/RQ2 case summaries once, with a fixed summariser.

The runners used to do this themselves, with the judge model, inside the RQ1 arm.
That meant the same judgment was summarised again for every model in the roster --
29,088 long-context calls instead of 2,928, about $1,215 instead of $80 -- and it
meant each model was scored on a summary it had written itself, which is not the
condition the paper describes.

Summaries are a property of the corpus and the summariser, not of the judge, so they
are built once here and every runner consumes the file:

    python scripts/build_summaries.py \
      --cases data/processed/livehrb_1k.json \
      --summarizer x-ai/grok-4.6 \
      --base-url https://openrouter.ai/api/v1 \
      --api-key-env OPENROUTER_API_KEY \
      --out data/processed/summaries_grok46.json

Work is deduplicated by judgment (19.6% of judgments appear under more than one
article) and checkpointed per (judgment, version), so an interrupted build resumes.
Failed calls are deliberately not checkpointed: they produced nothing usable, so a
rerun should retry them.
"""

import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

from openai import OpenAI

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "experiments"))
from checkpoint import Checkpoint          # noqa: E402
from scoring import MAX_CASE_CHARS         # noqa: E402
from summaries import SUMMARY_TEMPLATE, is_usable   # noqa: E402


def summarise(client, model, case, max_tokens, attempts=3):
    """One summary, escalating the token ceiling when the model returns nothing.

    Reasoning models spend their budget before emitting any content, so a ceiling
    sized for the visible answer comes back empty -- and retrying at the same
    ceiling just buys the same empty response again. One such loop cost $8 in three
    minutes, so each retry here raises the limit instead of repeating the call.
    """
    text = case["text"]
    prompt = SUMMARY_TEMPLATE.format(case_name=case["case_name"], full_text=text)
    limit, last = max_tokens, "ERROR: no attempt made"
    usage = {"prompt": 0, "completion": 0}
    for attempt in range(attempts):
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=1.0,
                max_tokens=limit,
            )
            if resp.usage:
                usage["prompt"] += resp.usage.prompt_tokens or 0
                usage["completion"] += resp.usage.completion_tokens or 0
            content = (resp.choices[0].message.content or "").strip()
            if content:
                return content, usage
            last = f"ERROR: empty content at max_tokens={limit}"
            limit *= 2
        except Exception as e:
            last = f"ERROR: {e}"
            time.sleep(2 * (attempt + 1))
    return last, usage


def main():
    p = argparse.ArgumentParser(description="Build shared case summaries")
    p.add_argument("--cases", required=True, help="cases JSON from build_livehrb_input.py")
    p.add_argument("--summarizer", required=True,
                   help="fully qualified slug, e.g. x-ai/grok-4.6. No default: a "
                        "silent default is how the judge model ended up summarising.")
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    p.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    p.add_argument("--out", required=True)
    p.add_argument("--versions", type=int, default=3, help="summaries per judgment")
    p.add_argument("--workers", type=int, default=20)
    p.add_argument("--max-tokens", type=int, default=4000)
    p.add_argument("--limit", type=int, help="first N judgments, for a dry run")
    p.add_argument("--no-resume", action="store_true")
    args = p.parse_args()

    api_key = os.environ.get(args.api_key_env, "")
    if not api_key:
        sys.exit(f"ERROR: set {args.api_key_env}")

    with open(args.cases) as f:
        instances = json.load(f)

    judgments = {}
    for c in instances:
        if c["item_id"] in judgments:
            continue
        text = c.get("full_case_text_no_verdict") or c.get("verdict_free_text") or ""
        judgments[c["item_id"]] = {"item_id": c["item_id"], "case_name": c["case_name"],
                                   "text": text[:MAX_CASE_CHARS]}
    cases = list(judgments.values())
    if args.limit:
        cases = cases[:args.limit]

    print(f"Summariser: {args.summarizer}")
    print(f"Instances: {len(instances)}  judgments: {len(judgments)}"
          f"{f' (limited to {len(cases)})' if args.limit else ''}")
    print(f"Versions: {args.versions}  ->  {len(cases) * args.versions:,} calls\n")

    ckpt = Checkpoint(args.out + ".jsonl", enabled=not args.no_resume)
    if ckpt.resumed:
        print(f"Resuming: {ckpt.resumed} summaries already recorded\n")

    client = OpenAI(base_url=args.base_url, api_key=api_key)

    units = [(c, v) for c in cases for v in range(args.versions)
             if not ckpt.done(Checkpoint.key("summary", c["item_id"], "", v))]
    done, failed = 0, 0
    total_usage = {"prompt": 0, "completion": 0}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(summarise, client, args.summarizer, c, args.max_tokens): (c, v)
                   for c, v in units}
        for fut in as_completed(futures):
            case, v = futures[fut]
            text, usage = fut.result()
            total_usage["prompt"] += usage["prompt"]
            total_usage["completion"] += usage["completion"]
            done += 1
            if is_usable(text):
                ckpt.record(Checkpoint.key("summary", case["item_id"], "", v),
                            {"item_id": case["item_id"], "case_name": case["case_name"],
                             "version": v, "summary": text})
            else:
                failed += 1
                print(f"\n  {case['case_name']} v{v}: {text[:120]}")
            print(f"\r  {done}/{len(units)} | failed {failed}", end="", flush=True)
    print()
    ckpt.close()

    by_case = {}
    for row in ckpt.rows():
        by_case.setdefault(row["item_id"], {})[row["version"]] = row["summary"]
    summaries = {item_id: [versions.get(v) for v in range(args.versions)]
                 for item_id, versions in by_case.items()}

    complete = sum(1 for v in summaries.values() if all(is_usable(s) for s in v))
    blob = {
        "summarizer": args.summarizer,
        "versions": args.versions,
        "n_judgments": len(cases),
        "n_complete": complete,
        "prompt_tokens": total_usage["prompt"],
        "completion_tokens": total_usage["completion"],
        "summaries": summaries,
    }
    with open(args.out, "w") as f:
        json.dump(blob, f, indent=2, ensure_ascii=False)

    print(f"\nWrote {args.out}")
    print(f"  {complete}/{len(cases)} judgments with all {args.versions} versions")
    print(f"  tokens: {total_usage['prompt']:,} in, {total_usage['completion']:,} out")
    if complete < len(cases):
        print(f"  {len(cases) - complete} incomplete -- rerun to retry, failures are not checkpointed")


if __name__ == "__main__":
    main()
