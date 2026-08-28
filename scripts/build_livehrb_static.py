#!/usr/bin/env python3
"""
LSYC-29: balanced 1K + 1K static eval set for LiveHumanRightsBench.

- regular 1K: from overthelex/echr-verdict-free, EXCLUDING Ukraine respondents,
  pair-level round-robin stratified by respondent (country) -> even country coverage.
- ukr 1K:     from overthelex/echr-ukr-verdict-free, pair-level round-robin
  stratified by ECHR article.
- Natural violation/no-violation distribution preserved (base-rate balancing is a
  separate task, STRAS-4). decision_date/appno/importance attached from
  decision_dates.json where available.
- Fully reproducible: fixed seed, deterministic round-robin.
"""
import argparse, json, os, random, sys

from conclusion_scrub import scrub_records
from collections import defaultdict, Counter

def load_hf(name):
    from datasets import load_dataset
    ds = load_dataset(name, split="train")
    return [dict(r) for r in ds]

def round_robin(records, strata_key, target, seed):
    rng = random.Random(seed)
    strata = defaultdict(list)
    for r in records:
        strata[str(r.get(strata_key) or "UNK")].append(r)
    # deterministic: sort strata by key, shuffle members with seeded rng
    order = sorted(strata.keys())
    for k in order:
        rng.shuffle(strata[k])
    picked, i = [], 0
    while len(picked) < target and any(strata[k] for k in order):
        k = order[i % len(order)]
        if strata[k]:
            picked.append(strata[k].pop())
        i += 1
    return picked

def enrich(rec, dates):
    d = dates.get(rec.get("item_id", ""), {})
    rec = dict(rec)
    rec["decision_date"] = d.get("decision_date", "")
    rec["application_number"] = d.get("application_number", "")
    rec["importance"] = d.get("importance", "")
    return rec

def stats(name, items):
    lab = Counter(x["violation_label"] for x in items)
    dated = sum(1 for x in items if x["decision_date"])
    yrs = sorted({x["decision_date"][:4] for x in items if x["decision_date"]})
    ncases = len({x["item_id"] for x in items})
    narts = len({x["article"] for x in items})
    ncoun = len({x.get("respondent") for x in items})
    print(f"[{name}] pairs={len(items)} cases={ncases} articles={narts} countries={ncoun} "
          f"labels={dict(lab)} dated={dated}/{len(items)} years={yrs[:1]}..{yrs[-1:]}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", default="data/processed/decision_dates.json")
    ap.add_argument("--out", default="data/processed/livehrb_static_1k1k.json")
    ap.add_argument("--target", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    dates = json.load(open(a.dates))
    print("regular pool: overthelex/echr-verdict-free")
    reg_all = load_hf("overthelex/echr-verdict-free")
    reg_pool = [r for r in reg_all if "ukrain" not in str(r.get("respondent", "")).lower()]
    print(f"  {len(reg_all)} pairs, {len(reg_pool)} after excluding Ukraine")
    print("ukr pool: overthelex/echr-ukr-verdict-free")
    ukr_pool = load_hf("overthelex/echr-ukr-verdict-free")
    print(f"  {len(ukr_pool)} pairs")

    reg = [enrich(r, dates) for r in round_robin(reg_pool, "respondent", a.target, a.seed)]
    ukr = [enrich(r, dates) for r in round_robin(ukr_pool, "article", a.target, a.seed)]
    for x in reg: x["group"] = "regular"
    for x in ukr: x["group"] = "ukr"

    # Defense in depth: scrub conclusion sentences even if the source sets
    # regress, and hard-fail on any own-article label leakage.
    reg = scrub_records(reg)
    ukr = scrub_records(ukr)

    stats("regular", reg); stats("ukr", ukr)
    out = {
        "meta": {
            "task": "LSYC-29 LiveHumanRightsBench static eval set",
            "seed": a.seed, "target_per_group": a.target,
            "sources": ["overthelex/echr-verdict-free (regular, ex-UA)",
                        "overthelex/echr-ukr-verdict-free (ukr)"],
            "stratify": {"regular": "respondent(country)", "ukr": "article"},
            "note": "natural label distribution; base-rate balancing = STRAS-4",
            "counts": {"regular": len(reg), "ukr": len(ukr)},
        },
        "items": reg + ukr,
    }
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), ensure_ascii=False)
    print("wrote", a.out, "total", len(reg) + len(ukr))

if __name__ == "__main__":
    main()
