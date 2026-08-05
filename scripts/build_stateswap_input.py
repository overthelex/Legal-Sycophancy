"""
Build eval-ready input from the state-swap set (overthelex/echr-livehrb-stateswap).
Emits one case dict per arm; `group` is set to the arm so the shared
make_custom_id (item_id, article, group) stays unique across the 4 arms.

Score target = case_text_rendered ; gold = violation_label ; bucket = article_full ;
pair by swap_group_id.

Run: python scripts/build_stateswap_input.py
"""
import argparse, json, sys
from pathlib import Path
from collections import Counter

REPO_ROOT = Path(__file__).resolve().parent.parent
HF_DATASET = "overthelex/echr-livehrb-stateswap"
DEFAULT_OUTPUT = REPO_ROOT / "data" / "processed" / "echr_stateswap.json"
VALID_LABELS = {"violation", "no_violation"}
EXPECTED_ARMS = {"control_original", "control_neutral", "probe_ukraine", "probe_russia"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=HF_DATASET)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: pip install datasets"); sys.exit(1)

    ds = load_dataset(args.source)
    split = "train" if "train" in ds else list(ds.keys())[0]
    rows = [dict(r) for r in ds[split]]
    print(f"Loaded {len(rows)} source rows from {args.source} [{split}]")

    out, bad = [], 0
    for r in rows:
        text = r.get("case_text_rendered") or ""
        label = r.get("violation_label")
        if not text or label not in VALID_LABELS:
            bad += 1
            continue
        article = str(r.get("article_full") or r.get("article") or "").strip()
        out.append({
            "swap_group_id": r["swap_group_id"],
            "item_id": str(r["item_id"]),
            "arm": r["arm"],
            "article": article,          # article_full -> used in prompt + bucketing
            "group": r["arm"],           # make_custom_id keys on group => arm-unique
            "violation_label": label,
            "case_name": r.get("case_name", ""),
            "respondent": r.get("respondent", ""),
            "respondent_original": r.get("respondent_original", ""),
            "full_case_text": text,      # the runners read this
        })

    # integrity: every swap group should carry exactly the 4 arms
    by_grp = Counter(c["swap_group_id"] for c in out)
    arms = Counter(c["arm"] for c in out)
    bad_groups = {g: n for g, n in by_grp.items() if n != 4}

    print(f"\nConverted {len(out)} rows (dropped {bad}).")
    print(f"  arms: {dict(arms)}")
    print(f"  swap groups: {len(by_grp)}  (expect ~816)")
    print(f"  groups NOT having exactly 4 arms: {len(bad_groups)}"
          + (f"  e.g. {list(bad_groups.items())[:3]}" if bad_groups else ""))
    unexpected = set(arms) - EXPECTED_ARMS
    if unexpected:
        print(f"  WARNING: unexpected arm labels: {unexpected}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"\nSaved {len(out)} rows to {args.output}")
    print("Next: python experiments/stateswap_evaluation.py --all-evaluators")


if __name__ == "__main__":
    main()
