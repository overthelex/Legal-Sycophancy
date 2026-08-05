"""
20-case dry run to confirm Anthropic prompt caching actually saves money on the
state-swap before the full sweep (per Vladimir).

It runs one model (Opus by default) over N cases x 10 samples, cases on the
OUTSIDE and samples on the INSIDE, so a case's 10 calls land inside Anthropic's
~5-minute cache window. cache_control sits on the big case-text block only, not
the instruction. It reads the cache hit counts back from the usage field and
reports the hit rate plus a projected full-arm cost.

Run:
  python scripts/stateswap_cache_dryrun.py
  python scripts/stateswap_cache_dryrun.py --model deepseek/deepseek-v4-pro --cases 10
"""
import argparse, json, sys, urllib.request, urllib.error
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from lib.prompts import EVALUATION_SYSTEM_PROMPT, BASELINE_EVALUATION_TEMPLATE

INPUT = REPO / "data" / "processed" / "echr_stateswap.json"
ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
ARTICLE_TITLES = {
    "2": "Right to life", "3": "Prohibition of torture",
    "5": "Right to liberty and security", "6": "Right to a fair trial",
    "8": "Right to respect for private and family life",
    "10": "Freedom of expression", "14": "Prohibition of discrimination",
    "P1-1": "Protection of property",
}
# rough list input $/M for the projection (edit if your rates differ)
INPUT_PRICE = {"anthropic/claude-opus-4.8": 5.0, "anthropic/claude-sonnet-5": 2.0}


def read_key():
    for line in open(REPO / ".env"):
        if line.startswith("OPENROUTER_API_KEY="):
            return line.split("=", 1)[1].strip().strip('"').strip("'")
    raise SystemExit("no OPENROUTER_API_KEY in .env")


def split_prompt(case_text, article):
    """Cacheable block = intro + case text; uncached = the instruction."""
    pre, post = BASELINE_EVALUATION_TEMPLATE.split("{case_text}")
    cached = pre + case_text
    title = ARTICLE_TITLES.get(str(article), f"Article {article}")
    uncached = post.format(article=article, article_title=title)
    return cached, uncached


def call(key, model, cached, uncached, max_tokens):
    body = {
        "model": model,
        "messages": [
            {"role": "system", "content": EVALUATION_SYSTEM_PROMPT},
            {"role": "user", "content": [
                {"type": "text", "text": cached, "cache_control": {"type": "ephemeral"}},
                {"type": "text", "text": uncached},
            ]},
        ],
        "max_tokens": max_tokens, "temperature": 1.0,
        "usage": {"include": True},
    }
    req = urllib.request.Request(
        ENDPOINT, data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=180)).get("usage", {}) or {}


def cache_read(u):
    for path in [("cache_read_input_tokens",),
                 ("prompt_tokens_details", "cached_tokens"),
                 ("cached_tokens",)]:
        v = u
        ok = True
        for p in path:
            if isinstance(v, dict) and p in v:
                v = v[p]
            else:
                ok = False; break
        if ok and v:
            return int(v)
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="anthropic/claude-opus-4.8")
    ap.add_argument("--cases", type=int, default=20)
    ap.add_argument("--samples", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=16000)
    args = ap.parse_args()

    key = read_key()
    rows = json.load(open(INPUT))[:args.cases]
    print(f"dry run  model={args.model}  cases={len(rows)}  samples={args.samples}\n")

    tot_prompt = tot_read = fails = 0
    shown = False
    for i, c in enumerate(rows):
        cached, uncached = split_prompt(c["full_case_text"], c["article"])
        for s in range(args.samples):
            try:
                u = call(key, args.model, cached, uncached, args.max_tokens)
            except Exception as e:
                fails += 1; continue
            if not shown:
                print("first-call usage (so we can see the real field names):")
                print("  ", json.dumps(u), "\n"); shown = True
            tot_prompt += int(u.get("prompt_tokens", 0) or 0)
            tot_read += cache_read(u)
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{len(rows)} cases done")

    hit = tot_read / tot_prompt if tot_prompt else 0.0
    print(f"\ntotal prompt tokens : {tot_prompt:,}")
    print(f"cache-read tokens   : {tot_read:,}  ({hit:.0%} of input hit cache)")
    print(f"failed calls        : {fails}")

    price = INPUT_PRICE.get(args.model)
    if price:
        full_arm_tokens = (tot_prompt / max(1, len(rows))) * 3264   # scale to 3264 cases
        no_cache = full_arm_tokens * price / 1e6
        # cached input billed ~ (1-hit) full + hit at 10% read rate
        cached_cost = full_arm_tokens * ((1 - hit) + hit * 0.10) * price / 1e6
        print(f"\nprojected full arm ({args.model}), input only:")
        print(f"  without caching : ${no_cache:,.0f}")
        print(f"  with this hit rate: ${cached_cost:,.0f}")
    print("\nHigh hit rate (say 80%+) means caching works. Near 0 means the samples "
          "are not landing in the cache window and we need to tighten per-case locality.")


if __name__ == "__main__":
    main()
