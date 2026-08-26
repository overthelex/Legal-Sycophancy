"""Re-cut the released datasets at the structural facts/law boundary.

    python scripts/leak_audit/rebuild_sets.py --out-dir build/

Only leaking rows are touched; every other row is written back byte-identical,
so a diff against the published version shows exactly the repair and nothing
else. Provenance fields are recomputed to follow the text -- leaving a stale
``retention_percentage`` behind is how the next audit ends up trusting the wrong
number.

Row membership is deliberately preserved rather than re-sampled. Re-running the
original sampler with ``seed=42`` risks changing which rows are in the set,
which would make every previously reported number incomparable for a second,
unrelated reason.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audit_leak import DATASETS, HUB  # noqa: E402
from leakdef import leaking, recut  # noqa: E402


def rebuild(name, column, out_dir):
    frame = pd.read_parquet(HUB.format(name=name))
    was_leaking = frame[column].map(leaking)
    frame[column] = [recut(text) if flag else text
                     for text, flag in zip(frame[column], was_leaking)]

    if {"original_length", "retention_percentage"} <= set(frame.columns):
        frame["retention_percentage"] = (
            100 * frame[column].str.len() / frame["original_length"]).round(1)
    if "verdict_free_length" in frame.columns:
        frame["verdict_free_length"] = frame[column].str.len()
    if "verdict_removal_method" in frame.columns:
        frame["verdict_removal_method"] = [
            f"{method}; recut: the_law_header" if flag else method
            for method, flag in zip(frame["verdict_removal_method"], was_leaking)]
    # the state-swap arms share one anonymised skeleton; it must stay shared
    if "case_text_templated" in frame.columns:
        frame["case_text_templated"] = [recut(text) if flag else text
                                        for text, flag in zip(frame["case_text_templated"],
                                                              was_leaking)]

    residual = frame[column].map(leaking).sum()
    print(f"{name}: re-cut {was_leaking.sum():,}/{len(frame):,} rows"
          f" ({100 * was_leaking.mean():.1f}%), leaking after: {residual}")
    if "swap_group_id" in frame.columns:
        shared = sum(len(set(g["case_text_templated"])) == 1
                     for _, g in frame.groupby("swap_group_id"))
        print(f"    templated skeleton identical across arms: "
              f"{shared}/{frame['swap_group_id'].nunique()} groups")

    out_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out_dir / f"{name}.parquet", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=list(DATASETS))
    parser.add_argument("--out-dir", type=Path, default=Path("build"))
    args = parser.parse_args()
    for name in args.datasets:
        rebuild(name, DATASETS[name], args.out_dir)


if __name__ == "__main__":
    main()
