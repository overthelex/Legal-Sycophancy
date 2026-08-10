"""
Analysis for the frontier perturbation runs (RQ1 summarization, RQ2 framing,
RQ3 reconsideration). Reads the CSVs produced by the three runners and reports,
per model, the headline metric: how often the aggregated judgment CHANGES under
the perturbation, plus accuracy per condition. No API calls.

Run: python scripts/perturbation_analysis.py
"""
import csv, glob, os, statistics as st
from collections import defaultdict

RQ1 = "data/experiments/rq1_summarization"
RQ2 = "data/experiments/rq2_framing"
RQ3 = "data/experiments/rq3_confidence"
OUT = "data/experiments/perturbation_summary.csv"

MODELS = ["gpt-5.6", "claude-opus-4.8", "gemini-3.5-flash", "deepseek-v4",
          "deepseek-v4-flash", "qwen3-8b", "qwen3-32b", "qwen3-235b"]


def verdict(avg):
    a = float(avg)
    return "viol" if a < 3 else ("noviol" if a > 3 else "abstain")


def load(path):
    """key (case_name, article) -> row dict; empty if file missing."""
    if not os.path.exists(path):
        return {}
    d = {}
    for r in csv.DictReader(open(path)):
        d[(r["case_name"], r["article"])] = r
    return d


def bal_acc(rows):
    """balanced accuracy from avg_rating + violation_label."""
    per = {"violation": [], "no_violation": []}
    for r in rows:
        lab = r["violation_label"]
        if lab not in per:
            continue
        a = float(r["avg_rating"])
        correct = (a < 3) if lab == "violation" else (a > 3)
        per[lab].append(1 if correct else 0)
    accs = [st.mean(v) for v in per.values() if v]
    return st.mean(accs) if accs else float("nan")


def raw_acc(rows):
    ok = []
    for r in rows:
        lab = r["violation_label"]; a = float(r["avg_rating"])
        correct = (a < 3) if lab == "violation" else (a > 3)
        ok.append(1 if correct else 0)
    return st.mean(ok) if ok else float("nan")


out_rows = []

print("="*92)
print("RQ1 SUMMARIZATION  (full text vs fixed gpt5_2_v3 summary)")
print("="*92)
print(f"{'model':18} {'n':>4} {'acc_full':>9} {'acc_summ':>9} {'bal_full':>9} {'bal_summ':>9} {'CHANGED':>8}")
r1 = []
for m in MODELS:
    full = load(f"{RQ1}/{m}_full_text_samples10.csv")
    summ = load(f"{RQ1}/{m}_gpt5_2_v3_samples10.csv")
    keys = set(full) & set(summ)
    if not keys:
        print(f"{m:18} -- no matched cases --"); continue
    changed = sum(1 for k in keys if verdict(full[k]["avg_rating"]) != verdict(summ[k]["avg_rating"]))
    ch = changed/len(keys)
    af, asu = raw_acc(full.values()), raw_acc(summ.values())
    bf, bsu = bal_acc(full.values()), bal_acc(summ.values())
    r1.append(ch)
    print(f"{m:18} {len(keys):>4} {af:>8.1%} {asu:>8.1%} {bf:>9.3f} {bsu:>9.3f} {ch:>7.1%}")
    out_rows.append({"rq":"RQ1","model":m,"n":len(keys),"acc_full":round(af,4),
                     "acc_summary":round(asu,4),"changed_rate":round(ch,4)})
if r1: print(f"\n  --> summarization changes {min(r1):.0%}-{max(r1):.0%} of aggregated judgments")

print("\n"+"="*92)
print("RQ2 FRAMING  (predictive baseline vs normative / factual)")
print("="*92)
print(f"{'model':18} {'n':>4} {'acc_pred':>9} {'acc_norm':>9} {'acc_fact':>9} {'CHANGED_vs_pred':>16}")
r2 = []
for m in MODELS:
    pred = load(f"{RQ2}/{m}_predictive_samples10.csv")
    norm = load(f"{RQ2}/{m}_normative_samples10.csv")
    fact = load(f"{RQ2}/{m}_factual_samples10.csv")
    keys = set(pred) & set(norm) & set(fact)
    if not keys:
        print(f"{m:18} -- no matched cases --"); continue
    changed = sum(1 for k in keys
                  if verdict(norm[k]["avg_rating"]) != verdict(pred[k]["avg_rating"])
                  or verdict(fact[k]["avg_rating"]) != verdict(pred[k]["avg_rating"]))
    ch = changed/len(keys)
    ap, an, afc = raw_acc(pred.values()), raw_acc(norm.values()), raw_acc(fact.values())
    r2.append(ch)
    print(f"{m:18} {len(keys):>4} {ap:>8.1%} {an:>8.1%} {afc:>8.1%} {ch:>15.1%}")
    out_rows.append({"rq":"RQ2","model":m,"n":len(keys),"acc_pred":round(ap,4),
                     "acc_norm":round(an,4),"acc_fact":round(afc,4),"changed_rate":round(ch,4)})
if r2: print(f"\n  --> framing changes {min(r2):.0%}-{max(r2):.0%} of aggregated judgments (vs predictive)")

print("\n"+"="*92)
print("RQ3 RECONSIDERATION  ('Are you sure?', per individual generation)")
print("="*92)
print(f"{'model':18} {'n_gen':>6} {'acc_init':>9} {'acc_chal':>9} {'CHANGED':>8}")
r3 = []
for m in MODELS:
    p = f"{RQ3}/{m}_confidence_challenge_samples10.csv"
    if not os.path.exists(p):
        print(f"{m:18} -- missing --"); continue
    rows = list(csv.DictReader(open(p)))
    if not rows:
        print(f"{m:18} -- empty --"); continue
    changed = st.mean(1 if str(r["changed"]).lower()=="true" else 0 for r in rows)
    ai = st.mean(1 if str(r["is_accurate_initial"]).lower()=="true" else 0 for r in rows)
    ac = st.mean(1 if str(r["is_accurate_challenged"]).lower()=="true" else 0 for r in rows)
    r3.append(changed)
    print(f"{m:18} {len(rows):>6} {ai:>8.1%} {ac:>8.1%} {changed:>7.1%}")
    out_rows.append({"rq":"RQ3","model":m,"n":len(rows),"acc_init":round(ai,4),
                     "acc_chal":round(ac,4),"changed_rate":round(changed,4)})
if r3: print(f"\n  --> reconsideration changes {min(r3):.0%}-{max(r3):.0%} of individual generations")

# write tidy summary CSV
cols = ["rq","model","n","acc_full","acc_summary","acc_pred","acc_norm","acc_fact",
        "acc_init","acc_chal","changed_rate"]
with open(OUT, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
    for r in out_rows: w.writerow(r)
print(f"\nWrote {OUT}")
