"""Interval for the "k of n cells miss the FPR target" count.

The count is a point estimate whose discreteness hides its sampling
distribution (docs/verification.md). Each cell's severity is a function of its
mean realized FPR over B calibration-cohort draws, so resampling those draws
within each cell and recounting gives the count's distribution under the one
source of variation this design actually samples.

Read the caveat printed at the end before quoting anything from here: the
deployment set is fixed and, within-corpus, the 21LA channel conditions share source
recordings, so this interval is a LOWER BOUND on the real uncertainty.
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
ALPHA, SEV_BAR, OLD_RULE = 0.05, 1.0, 0.05
N_BOOT = 2000
SEED = 48


def main():
    d = json.load(open(HERE / "results_drift.json"))
    cells = {k: v for s in ("within", "cross") for k, v in d[s].items()}
    named = [(k, v) for k, v in cells.items() if "_fpr_draws" in v]
    assert named, "results_drift.json predates the _fpr_draws field; re-run drift_map.py"

    draws = {k: np.asarray(v["_fpr_draws"], dtype=np.float64) for k, v in named}
    floors = {k: 3.0 / v["n_dep_bona"] for k, v in named}
    old_pass = {k for k, v in named if v["excursion"] <= OLD_RULE}

    def sev(k, fpr):
        return np.log2(np.maximum(fpr, floors[k]) / ALPHA)

    point = sum(1 for k in old_pass if abs(sev(k, draws[k].mean())) > SEV_BAR)

    rng = np.random.default_rng(SEED)
    counts = np.empty(N_BOOT, dtype=int)
    for b in range(N_BOOT):
        n = 0
        for k in old_pass:
            x = draws[k]
            m = x[rng.integers(0, len(x), len(x))].mean()
            n += abs(sev(k, m)) > SEV_BAR
        counts[b] = n
    lo, hi = np.percentile(counts, [2.5, 97.5])

    # Boundary cases: cells whose verdict is not stable across replicates.
    unstable = []
    for k in sorted(old_pass):
        x = draws[k]
        ms = np.array([x[rng.integers(0, len(x), len(x))].mean() for _ in range(400)])
        p = float((np.abs(sev(k, ms)) > SEV_BAR).mean())
        if 0.02 < p < 0.98:
            unstable.append((k, p, float(abs(sev(k, x.mean())))))

    print(f"cells passing the old rule: {len(old_pass)} of {len(named)}")
    print(f"of those, miss target by >2x: {point} (point estimate)")
    print(f"bootstrap over calibration-cohort draws: 95% CI [{lo:.0f}, {hi:.0f}], "
          f"median {np.median(counts):.0f}")
    for q, lbl in ((1, "99% of replicates are at least"), (5, "95% of replicates are at least")):
        print(f"  {lbl}: {np.percentile(counts, q):.0f}")
    print(f"\nboundary cells (verdict flips across replicates): {len(unstable)}")
    for k, p, s in sorted(unstable, key=lambda t: -t[1]):
        print(f"  {k:42s} P(miss) {p:.2f}  |severity| {s:.2f}")

    out = {"n_cells": len(named), "n_old_rule_pass": len(old_pass),
           "point": point, "ci95": [float(lo), float(hi)],
           "p01": float(np.percentile(counts, 1)), "p05": float(np.percentile(counts, 5)),
           "boundary_cells": [{"cell": k, "p_miss": p, "severity": s} for k, p, s in unstable],
           "caveat": ("Resamples calibration-cohort draws only. The deployment set is "
                      "fixed and the 21LA channel conditions share source recordings, so "
                      "this is a lower bound on the true uncertainty.")}
    (HERE / "results_count_bootstrap.json").write_text(json.dumps(out, indent=1))
    print(f"\nCAVEAT: {out['caveat']}")


if __name__ == "__main__":
    main()
