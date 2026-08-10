"""
Paper-ready analysis for the frontier perturbation runs (RQ1 summarization,
RQ2 framing, RQ3 reconsideration), matching the paper's stated methods:
  - change-rate (fraction of aggregated judgments that flip) with a 95%
    bootstrap CI (10,000 resamples; cluster-resampled by case for RQ3)
  - McNemar exact test on paired accuracy (baseline vs perturbed)
No API calls. Deterministic (fixed seed).

Run: python scripts/perturbation_significance.py
"""
import csv, os, statistics as st
import numpy as np
from scipy.stats import binomtest

RQ1 = "data/experiments/rq1_summarization"
RQ2 = "data/experiments/rq2_framing"
RQ3 = "data/experiments/rq3_confidence"
OUT = "data/experiments/perturbation_significance.csv"
N_BOOT = 10000
SEED = 12345
MODELS = ["gpt-5.6", "claude-opus-4.8", "gemini-3.5-flash", "deepseek-v4",
          "deepseek-v4-flash", "qwen3-8b", "qwen3-32b", "qwen3-235b"]


def verdict(avg):
    a = float(avg)
    return "viol" if a < 3 else ("noviol" if a > 3 else "abstain")


def correct(avg, label):
    a = float(avg)
    return (a < 3) if label == "violation" else (a > 3)


def load(path):
    if not os.path.exists(path):
        return {}
    return {(r["case_name"], r["article"]): r for r in csv.DictReader(open(path))}


def boot_ci(indicators, groups=None, seed=SEED):
    """95% CI for the mean of 0/1 indicators via bootstrap.
    If groups given (cluster ids), resample clusters (for correlated samples)."""
    rng = np.random.default_rng(seed)
    ind = np.asarray(indicators, float)
    if groups is None:
        n = len(ind)
        if n == 0:
            return (float("nan"), float("nan"))
        means = ind[rng.integers(0, n, size=(N_BOOT, n))].mean(axis=1)
    else:
        # cluster bootstrap: resample group ids, pool their rows
        from collections import defaultdict
        buckets = defaultdict(list)
        for v, g in zip(ind, groups):
            buckets[g].append(v)
        gids = list(buckets.keys())
        arrs = {g: np.asarray(buckets[g]) for g in gids}
        ng = len(gids)
        means = np.empty(N_BOOT)
        for b in range(N_BOOT):
            pick = rng.integers(0, ng, size=ng)
            pooled = np.concatenate([arrs[gids[i]] for i in pick])
            means[b] = pooled.mean()
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def mcnemar(paired):
    """paired: list of (baseline_correct, perturbed_correct) bools -> p-value."""
    b = sum(1 for x, y in paired if x and not y)
    c = sum(1 for x, y in paired if not x and y)
    if b + c == 0:
        return 1.0, b, c
    return binomtest(min(b, c), b + c, 0.5, alternative="two-sided").pvalue, b, c


rows_out = []

print("="*100)
print("RQ1 SUMMARIZATION  (full text -> fixed gpt5_2_v3 summary)")
print("="*100)
print(f"{'model':18} {'n':>4} {'change% [95% CI]':>22} {'acc_full':>9} {'acc_summ':>9} {'McNemar p (acc)':>16}")
for m in MODELS:
    full, summ = load(f"{RQ1}/{m}_full_text_samples10.csv"), load(f"{RQ1}/{m}_gpt5_2_v3_samples10.csv")
    keys = sorted(set(full) & set(summ))
    if not keys:
        print(f"{m:18} -- no matched cases --"); continue
    chg = [1 if verdict(full[k]["avg_rating"]) != verdict(summ[k]["avg_rating"]) else 0 for k in keys]
    lo, hi = boot_ci(chg)
    paired = [(correct(full[k]["avg_rating"], full[k]["violation_label"]),
               correct(summ[k]["avg_rating"], summ[k]["violation_label"])) for k in keys]
    p, b, c = mcnemar(paired)
    af = st.mean(1 if x else 0 for x, _ in paired); asu = st.mean(1 if y else 0 for _, y in paired)
    print(f"{m:18} {len(keys):>4} {st.mean(chg):>8.1%} [{lo:>4.1%},{hi:>5.1%}] {af:>8.1%} {asu:>8.1%} {p:>13.3f}{'*' if p<0.05 else ' '}")
    rows_out.append(dict(rq="RQ1", model=m, n=len(keys), change_rate=round(st.mean(chg),4),
                         ci_lo=round(lo,4), ci_hi=round(hi,4), acc_base=round(af,4),
                         acc_pert=round(asu,4), mcnemar_p=round(p,4)))

print("\n"+"="*100)
print("RQ2 FRAMING  (predictive baseline vs normative / factual)")
print("="*100)
print(f"{'model':18} {'n':>4} {'change% [95% CI]':>22} {'p_norm':>8} {'p_fact':>8}")
for m in MODELS:
    pred, norm, fact = (load(f"{RQ2}/{m}_predictive_samples10.csv"),
                        load(f"{RQ2}/{m}_normative_samples10.csv"),
                        load(f"{RQ2}/{m}_factual_samples10.csv"))
    keys = sorted(set(pred) & set(norm) & set(fact))
    if not keys:
        print(f"{m:18} -- no matched cases --"); continue
    chg = [1 if (verdict(norm[k]["avg_rating"]) != verdict(pred[k]["avg_rating"]) or
                 verdict(fact[k]["avg_rating"]) != verdict(pred[k]["avg_rating"])) else 0 for k in keys]
    lo, hi = boot_ci(chg)
    pn = mcnemar([(correct(pred[k]["avg_rating"], pred[k]["violation_label"]),
                   correct(norm[k]["avg_rating"], norm[k]["violation_label"])) for k in keys])[0]
    pf = mcnemar([(correct(pred[k]["avg_rating"], pred[k]["violation_label"]),
                   correct(fact[k]["avg_rating"], fact[k]["violation_label"])) for k in keys])[0]
    print(f"{m:18} {len(keys):>4} {st.mean(chg):>8.1%} [{lo:>4.1%},{hi:>5.1%}] {pn:>7.3f}{'*' if pn<0.05 else ' '} {pf:>7.3f}{'*' if pf<0.05 else ' '}")
    rows_out.append(dict(rq="RQ2", model=m, n=len(keys), change_rate=round(st.mean(chg),4),
                         ci_lo=round(lo,4), ci_hi=round(hi,4), mcnemar_p_norm=round(pn,4),
                         mcnemar_p_fact=round(pf,4)))

print("\n"+"="*100)
print("RQ3 RECONSIDERATION  ('Are you sure?', per generation; CI cluster-bootstrapped by case)")
print("="*100)
print(f"{'model':18} {'n_gen':>6} {'change% [95% CI]':>22} {'acc_init':>9} {'acc_chal':>9} {'McNemar p (acc)':>16}")
for m in MODELS:
    p = f"{RQ3}/{m}_confidence_challenge_samples10.csv"
    if not os.path.exists(p):
        print(f"{m:18} -- missing --"); continue
    rows = list(csv.DictReader(open(p)))
    chg = [1 if str(r["changed"]).lower() == "true" else 0 for r in rows]
    grp = [r.get("item_id") or r.get("case_name") for r in rows]
    lo, hi = boot_ci(chg, groups=grp)
    paired = [(str(r["is_accurate_initial"]).lower() == "true",
               str(r["is_accurate_challenged"]).lower() == "true") for r in rows]
    pv, b, c = mcnemar(paired)
    ai = st.mean(1 if x else 0 for x, _ in paired); ac = st.mean(1 if y else 0 for _, y in paired)
    print(f"{m:18} {len(rows):>6} {st.mean(chg):>8.1%} [{lo:>4.1%},{hi:>5.1%}] {ai:>8.1%} {ac:>8.1%} {pv:>13.3f}{'*' if pv<0.05 else ' '}")
    rows_out.append(dict(rq="RQ3", model=m, n=len(rows), change_rate=round(st.mean(chg),4),
                         ci_lo=round(lo,4), ci_hi=round(hi,4), acc_base=round(ai,4),
                         acc_pert=round(ac,4), mcnemar_p=round(pv,4)))

cols = ["rq","model","n","change_rate","ci_lo","ci_hi","acc_base","acc_pert",
        "mcnemar_p","mcnemar_p_norm","mcnemar_p_fact"]
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore"); w.writeheader()
    for r in rows_out: w.writerow(r)
print(f"\n* = accuracy change significant at p<0.05 (McNemar exact).  Wrote {OUT}")
