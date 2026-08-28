"""Recompute stored verdicts from the stored ratings under the corrected tie rule.

Pure function of what is already on disk: no API calls, nothing re-run. Rewrites
both the .jsonl stream and the _results.json the analysis actually reads.
"""
import json, glob, os, shutil, sys
sys.path.insert(0, "experiments")
from scoring import majority_vote, abstention_kind

SRC, DST = "data/experiments/full_scale", "data/experiments/full_scale_tiefix"
if os.path.exists(DST): shutil.rmtree(DST)
changed = seen = 0

def rescore(r):
    global changed
    if "original_ratings" in r:                       # rq3 carries two verdicts
        for side in ("original", "challenged"):
            p, a = majority_vote(r[side + "_ratings"])
            if r.get(side + "_prediction") != p: changed += 1
            r[side + "_prediction"], r[side + "_abstained"] = p, a
            r[side + "_abstention_kind"] = abstention_kind(r[side + "_ratings"])
        r["changed"] = r["original_prediction"] != r["challenged_prediction"]
    else:
        p, a = majority_vote(r["ratings"])
        if r.get("prediction") != p: changed += 1
        r["prediction"], r["abstained"] = p, a
        r["abstention_kind"] = abstention_kind(r["ratings"])
        if r.get("violation_label") is not None:
            r["accurate"] = (p == r["violation_label"])
    return r

for path in sorted(glob.glob(SRC + "/*/*.jsonl")):
    out = path.replace(SRC, DST)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as w:
        for line in open(path):
            seen += 1
            w.write(json.dumps(rescore(json.loads(line))) + "\n")

for path in sorted(glob.glob(SRC + "/*/*_results.json")):
    out = path.replace(SRC, DST)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rows = json.load(open(path))
    json.dump([rescore(r) for r in rows], open(out, "w"))

print("%d verdicts changed" % changed)
