#!/usr/bin/env python3
"""
LSYC-30: temporally binned live split for LiveHumanRightsBench.

Two axes of temporal contamination control (ECtHR HUDOC, verdict-free, public
sources only -> redistributable). Reproducible: fixed seed, deterministic
round-robin.

- regular_temporal (1K): ex-Ukraine cases from overthelex/echr-verdict-free,
  binned by decision YEAR over a 10-year window (2017-2026), 100/bin,
  round-robin stratified by respondent within each year -> even country + time
  coverage. Enables per-model pre/post training-cutoff analysis and temporal
  drift plots (temporal-drift framework arXiv:2605.24452).
- ukr_temporal (1K): overthelex/echr-ukr-verdict-free split at Russia's
  full-scale invasion (2022-02-24): 500 pre / 500 post, round-robin by article
  within each stratum -> geopolitical-shift axis.

Adds two columns to the base echr schema: group (regular_temporal|ukr_temporal)
and bin (year string, or pre_2022|post_2022).
"""
import argparse

import json, os, random, datetime as dt
from conclusion_scrub import scrub_records
from collections import defaultdict, Counter

BOUNDARY = dt.date(2022, 2, 24)  # Russia full-scale invasion of Ukraine


def load_hf(name):
    from datasets import load_dataset
    return [dict(r) for r in load_dataset(name, split="train")]


def parse_date(s):
    if not s:
        return None
    s = s[:10]
    for f in ("%d/%m/%Y", "%Y-%m-%d", "%d-%m-%Y"):
        try:
            return dt.datetime.strptime(s, f).date()
        except ValueError:
            pass
    try:
        return dt.date(int(s[:4]), 1, 1)
    except ValueError:
        return None


def get_date(rec, dates):
    d = rec.get("decision_date") or dates.get(rec.get("item_id", ""), {}).get("decision_date", "")
    return parse_date(d)


def round_robin(records, strata_key, target, seed):
    rng = random.Random(seed)
    strata = defaultdict(list)
    for r in records:
        strata[str(r.get(strata_key) or "UNK")].append(r)
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


def enrich(rec, dates, group, bin_):
    d = dates.get(rec.get("item_id", ""), {})
    rec = dict(rec)
    rec["decision_date"] = rec.get("decision_date") or d.get("decision_date", "")
    rec["application_number"] = rec.get("application_number") or d.get("application_number", "")
    rec["importance"] = rec.get("importance") or d.get("importance", "")
    rec["ecli"] = rec.get("ecli") or d.get("ecli", "")
    rec["group"] = group
    rec["bin"] = bin_
    return rec


def stats(name, items):
    lab = Counter(x["violation_label"] for x in items)
    bins = Counter(x["bin"] for x in items)
    dated = sum(1 for x in items if x["decision_date"])
    ncoun = len({x.get("respondent") for x in items})
    print(f"[{name}] pairs={len(items)} labels={dict(lab)} countries={ncoun} "
          f"dated={dated}/{len(items)} bins={dict(sorted(bins.items()))}")


def build_card(repo, meta, reg_n, ukr_n):
    return """---
license: cc-by-4.0
task_categories:
- text-classification
language:
- en
tags:
- legal
- echr
- human-rights
- verdict-free
- temporal
size_categories:
- 1K<n<10K
---

# echr-livehrb-temporal-2k

Temporally binned evaluation split for **LiveHumanRightsBench** (ECtHR
human-rights judgment prediction). Two temporal contamination-control axes,
built from verdict-free (contamination-controlled) ECtHR text.

- **regular_temporal** (%d): ex-Ukraine cases from `overthelex/echr-verdict-free`,
  binned by decision year over %s-%s (100/bin), round-robin stratified by
  respondent country. Supports per-model pre/post training-cutoff analysis and
  temporal-drift plots (arXiv:2605.24452).
- **ukr_temporal** (%d): from `overthelex/echr-ukr-verdict-free`, split at
  Russia's full-scale invasion (2022-02-24): %d pre / %d post, round-robin by
  ECHR article -> geopolitical-shift axis.

Natural violation/no-violation base rate preserved (base-rate balancing is a
separate task). Public HUDOC sources only -> redistributable. Seed=%d.

## Columns

item_id, case_name, respondent, article, violation_label, verdict_free_text,
decision_date, application_number, importance, ecli, **group**
(regular_temporal | ukr_temporal), **bin** (year string | pre_2022 | post_2022),
plus length fields.

## Usage

    from datasets import load_dataset
    ds = load_dataset("%s", split="train")
    # per-year drift on regular cases
    y2024 = ds.filter(lambda r: r["group"] == "regular_temporal" and r["bin"] == "2024")
    # Ukraine pre/post full-scale invasion
    ukr_pre  = ds.filter(lambda r: r["bin"] == "pre_2022")
    ukr_post = ds.filter(lambda r: r["bin"] == "post_2022")
""" % (reg_n, meta["year_start"], meta["year_end"], ukr_n,
       meta["ukr_pre"], meta["ukr_post"], meta["seed"], repo)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", default="data/processed/decision_dates.json")
    ap.add_argument("--out", default="data/processed/livehrb_temporal_2k.json")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--year-start", type=int, default=2017)
    ap.add_argument("--year-end", type=int, default=2026)
    ap.add_argument("--per-bin", type=int, default=100)
    ap.add_argument("--ukr-per-side", type=int, default=500)
    ap.add_argument("--repo", default="overthelex/echr-livehrb-temporal-2k")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args()

    dates = json.load(open(a.dates))

    # --- regular_temporal: ex-UA, binned by year ---
    reg_all = load_hf("overthelex/echr-verdict-free")
    by_year = defaultdict(list)
    for r in reg_all:
        if "ukrain" in str(r.get("respondent", "")).lower():
            continue
        d = get_date(r, dates)
        if d and a.year_start <= d.year <= a.year_end:
            by_year[d.year].append(r)
    reg = []
    for y in range(a.year_start, a.year_end + 1):
        pool = by_year.get(y, [])
        picked = round_robin(pool, "respondent", a.per_bin, a.seed + y)
        reg += [enrich(r, dates, "regular_temporal", str(y)) for r in picked]
        print(f"  regular {y}: pool={len(pool)} picked={len(picked)}")

    # --- ukr_temporal: pre/post invasion ---
    ukr_all = load_hf("overthelex/echr-ukr-verdict-free")
    pre, post = [], []
    for r in ukr_all:
        d = get_date(r, dates)
        if not d:
            continue
        (pre if d < BOUNDARY else post).append(r)
    ukr = [enrich(r, dates, "ukr_temporal", "pre_2022")
           for r in round_robin(pre, "article", a.ukr_per_side, a.seed)]
    ukr += [enrich(r, dates, "ukr_temporal", "post_2022")
            for r in round_robin(post, "article", a.ukr_per_side, a.seed)]
    print(f"  ukr pool pre={len(pre)} post={len(post)}")

    # Defense in depth: scrub conclusion sentences even if the source sets
    # regress, and hard-fail on any own-article label leakage.
    reg = scrub_records(reg)
    ukr = scrub_records(ukr)

    stats("regular_temporal", reg)
    stats("ukr_temporal", ukr)
    items = reg + ukr

    ukr_pre = sum(1 for x in ukr if x["bin"] == "pre_2022")
    ukr_post = sum(1 for x in ukr if x["bin"] == "post_2022")
    meta = {
        "task": "LSYC-30 LiveHumanRightsBench temporal split",
        "seed": a.seed,
        "year_start": a.year_start, "year_end": a.year_end,
        "per_bin": a.per_bin, "ukr_per_side": a.ukr_per_side,
        "ukr_pre": ukr_pre, "ukr_post": ukr_post,
        "sources": ["overthelex/echr-verdict-free (regular, ex-UA)",
                    "overthelex/echr-ukr-verdict-free (ukr)"],
        "axes": {"regular_temporal": "year bins (per-model cutoff / drift)",
                 "ukr_temporal": "pre/post 2022-02-24 full-scale invasion"},
        "framework": "temporal-drift arXiv:2605.24452",
        "counts": {"regular_temporal": len(reg), "ukr_temporal": len(ukr)},
    }
    out = {"meta": meta, "items": items}
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(out, open(a.out, "w"), ensure_ascii=False)
    print("wrote", a.out, "total", len(items))

    if a.push:
        from datasets import Dataset
        from huggingface_hub import HfApi
        ds = Dataset.from_list(items)
        ds.push_to_hub(a.repo, commit_message=(
            "LiveHumanRightsBench temporal split: 1K regular year-binned + "
            "1K Ukrainian pre/post 2022-02-24 (verdict-free, dated)"))
        card = build_card(a.repo, meta, len(reg), len(ukr))
        HfApi().upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md",
                            repo_id=a.repo, repo_type="dataset",
                            commit_message="Add dataset card")
        print("PUSHED", a.repo)


if __name__ == "__main__":
    main()
