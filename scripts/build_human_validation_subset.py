"""
Build a human-validation subset for the state-swap perturbation. Samples N swap
groups and, for each, lays out the real-state arm next to a probe arm so a human
annotator can confirm the perturbation is fact-preserving (only the country and
its adjective changed, facts identical, reads naturally).

Output CSV has blank columns for the annotator to fill; agreement is then the
rate of "yes" across annotators.

Run: python scripts/build_human_validation_subset.py --n 40
"""
import argparse, csv, json, random
from pathlib import Path
from collections import defaultdict

REPO = Path(__file__).resolve().parent.parent
INPUT = REPO / "data" / "processed" / "echr_stateswap.json"
OUT = REPO / "data" / "experiments" / "stateswap" / "human_validation_subset.csv"
PROTOCOL = """HUMAN VALIDATION PROTOCOL (state-swap fact-preservation)
For each row, read text_real and text_probe side by side and answer:
 - facts_preserved      (yes/no): are all facts, dates, numbers, events identical?
 - only_country_changed (yes/no): does ONLY the country and its adjective differ?
 - reads_naturally      (yes/no): does the swapped version read like a normal judgment?
 - notes: anything odd (leaked original country, broken grammar, changed fact).
Agreement = share of rows where facts_preserved = only_country_changed = yes,
and inter-annotator agreement across 2+ annotators on those two columns."""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--probe", default="probe_ukraine",
                    choices=["probe_ukraine", "probe_russia"])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    d = json.load(open(INPUT))
    g = defaultdict(dict)
    for c in d:
        g[c["swap_group_id"]][c["arm"]] = c
    groups = [gid for gid, arms in g.items()
              if "control_original" in arms and args.probe in arms]
    random.seed(args.seed)
    sample = random.sample(groups, min(args.n, len(groups)))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["swap_group_id", "article", "real_country", "probe_country",
                    "text_real", "text_probe",
                    "facts_preserved", "only_country_changed", "reads_naturally",
                    "annotator", "notes"])
        for gid in sample:
            o = g[gid]["control_original"]; p = g[gid][args.probe]
            w.writerow([gid, o["article"], o["respondent"], p["respondent"],
                        o["full_case_text"], p["full_case_text"], "", "", "", "", ""])

    print(f"Wrote {len(sample)} rows to {OUT}")
    print(f"Pairing: control_original vs {args.probe}\n")
    print(PROTOCOL)
    with open(OUT.parent / "human_validation_protocol.txt", "w") as f:
        f.write(PROTOCOL + "\n")


if __name__ == "__main__":
    main()
