"""
Free (no-API, no-network) refresh of the existing MFT results: bucket the
procedural / ancillary articles out of the headline and report a
SUBSTANTIVE-ONLY balanced accuracy. Uses the eval CSVs you already have.

This is the higher-impact correction (Art 41/34/38 aren't merits questions).
The article_full per-article re-key (~60 protocol rows) rides along with the
canonical 001-only re-run on Vladimir's rebuild.

Run: python scripts/mft_refresh_v12.py
"""
import csv, glob, os, sys
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parent.parent

# Ancillary / procedural articles — NOT merits "will the Court find a violation"
# questions (41 = just satisfaction, 34 = individual petition, 38 = examination,
# 46 = execution, 39 = friendly settlement, 35 = admissibility).
PROCEDURAL = {"34", "38", "41", "46", "39", "35"}


def balanced_acc(rows):
    by_lab = defaultdict(list)
    for r in rows:
        by_lab[r["violation_label"]].append(str(r["is_accurate"]).lower() == "true")
    accs = [sum(v) / len(v) for v in by_lab.values() if v]
    return sum(accs) / len(accs) if len(accs) == 2 else float("nan")


def main():
    files = sorted(glob.glob(str(REPO / "data/experiments/mft/*_mft_samples10.csv")))
    if not files:
        print("No MFT CSVs found in data/experiments/mft/"); sys.exit(1)

    print("=" * 78)
    print("MFT REFRESH — substantive vs procedural (no re-eval, no network)")
    print("=" * 78)
    print(f"procedural bucket = {sorted(PROCEDURAL)}\n")
    print(f"{'model':18} {'bal_acc_ALL':>12} {'bal_acc_SUBSTANTIVE':>20} "
          f"{'n_subst':>8} {'n_proc':>7}")

    for f in files:
        rows = list(csv.DictReader(open(f)))
        name = os.path.basename(f).split("_mft")[0]
        subst = [r for r in rows if str(r["article"]) not in PROCEDURAL]
        proc = [r for r in rows if str(r["article"]) in PROCEDURAL]
        print(f"{name:18} {balanced_acc(rows):>12.4f} {balanced_acc(subst):>20.4f} "
              f"{len(subst):>8} {len(proc):>7}")

    # per-procedural-article pass rates, pooled across models, so you can see
    # exactly which ones drag the headline
    print("\nprocedural-article pass rate (pooled over models):")
    pooled = defaultdict(list)
    for f in files:
        for r in csv.DictReader(open(f)):
            if str(r["article"]) in PROCEDURAL:
                pooled[str(r["article"])].append(str(r["is_accurate"]).lower() == "true")
    for a in sorted(pooled):
        v = pooled[a]
        print(f"   art {a:>3}: {sum(v)/len(v):.3f}  (n={len(v)})")

    print("\nNote: interim view on current (002-included) data. Canonical numbers "
          "= 001-only re-run on Vladimir's rebuild, keyed on article_full.")


if __name__ == "__main__":
    main()
