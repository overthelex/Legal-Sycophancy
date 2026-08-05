"""
State-swap Δ analysis. Pairs the 4 arms by swap_group_id and measures how the
country label moves each model's judgment, holding facts byte-fixed. All
contrasts are against control_neutral (Iceland):

  probe_ukraine   - control_neutral  -> Ukraine country prior (headline; time-free)
  probe_russia    - control_neutral  -> generic adversarial-state prior
  control_original- control_neutral  -> real-state prior

Positive mean_rating_delta / net_into_violation = the probe country pushes the
model TOWARD "violation" relative to Iceland, on identical facts.

Run: python scripts/stateswap_analysis.py
"""
import glob, os, sys
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = REPO_ROOT / "data" / "experiments" / "stateswap"
ANALYSIS_DIR = RESULTS_DIR / "analysis"
CONTROL = "control_neutral"
PROBES = ["probe_ukraine", "probe_russia", "control_original"]


def contrast(piv_r, piv_p, probe):
    if probe not in piv_r.columns or CONTROL not in piv_r.columns:
        return None
    r = piv_r.dropna(subset=[CONTROL, probe])
    p = piv_p.dropna(subset=[CONTROL, probe])
    n = len(r)
    if n == 0:
        return None
    delta = r[CONTROL] - r[probe]                      # >0 => probe leans more violation
    to_viol = int(((p[CONTROL] != "violation") & (p[probe] == "violation")).sum())
    from_viol = int(((p[CONTROL] == "violation") & (p[probe] != "violation")).sum())
    return {
        "probe": probe,
        "n": n,
        "flip_rate": round((p[CONTROL] != p[probe]).mean(), 4),
        "mean_rating_delta": round(float(delta.mean()), 4),   # + = toward violation
        "n_toward_violation": int((r[probe] < r[CONTROL]).sum()),
        "n_away": int((r[probe] > r[CONTROL]).sum()),
        "pred_into_violation": to_viol,
        "pred_out_of_violation": from_viol,
        "net_into_violation": to_viol - from_viol,
    }


def main():
    files = sorted(glob.glob(str(RESULTS_DIR / "*_stateswap_samples*.csv")))
    if not files:
        print(f"No state-swap CSVs in {RESULTS_DIR}. Run stateswap_evaluation.py first.")
        sys.exit(1)
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)

    all_rows = []
    for f in files:
        name = os.path.basename(f).split("_stateswap")[0]
        df = pd.read_csv(f)
        piv_r = df.pivot_table(index="swap_group_id", columns="arm",
                               values="avg_rating", aggfunc="first")
        piv_p = df.pivot_table(index="swap_group_id", columns="arm",
                               values="prediction", aggfunc="first")
        print("=" * 92)
        print(f"MODEL: {name}   (pairs with both arms present shown as n)")
        print("-" * 92)
        print(f"{'contrast':22} {'n':>4} {'flip':>6} {'Δrating→viol':>13} "
              f"{'→viol':>6} {'→noviol':>8} {'net_into_viol':>14}")
        for probe in PROBES:
            c = contrast(piv_r, piv_p, probe)
            if not c:
                continue
            c["model"] = name
            all_rows.append(c)
            print(f"{probe+' - '+CONTROL:22} {c['n']:>4} {c['flip_rate']:>6.3f} "
                  f"{c['mean_rating_delta']:>+13.3f} {c['pred_into_violation']:>6} "
                  f"{c['pred_out_of_violation']:>8} {c['net_into_violation']:>+14}")
        print()

    out = pd.DataFrame(all_rows)[
        ["model", "probe", "n", "flip_rate", "mean_rating_delta",
         "n_toward_violation", "n_away", "pred_into_violation",
         "pred_out_of_violation", "net_into_violation"]
    ]
    out.to_csv(ANALYSIS_DIR / "stateswap_contrasts.csv", index=False)
    print("=" * 92)
    print("HEADLINE = 'probe_ukraine - control_neutral'. Positive mean_rating_delta / "
          "net_into_violation\nmeans relabeling identical facts as Ukraine pushes the "
          "model toward 'violation' (a country prior,\nwith the time confound removed). "
          "Compare against probe_russia to see if it is Ukraine-specific.")
    print(f"\nSaved {ANALYSIS_DIR / 'stateswap_contrasts.csv'}")


if __name__ == "__main__":
    main()
