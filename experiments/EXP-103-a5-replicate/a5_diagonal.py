"""Same-codec control for the ASVspoof 5 replication.

For every condition c, calibrate on speaker half 0 of c and deploy on speaker half 1
of the same c (twin-free arm, N=500, B=1000), with the estimators of analyze.py.
A near-target result here attributes the off-diagonal failures to the codec change
rather than to the speaker split. Writes artifacts/a5_diagonal.json.
"""

import json
from pathlib import Path

import numpy as np

from analyze import ALPHA, B, N_CAL, OVERLAP_BAR, build, cells, load_scores, log2_ratio

HERE = Path(__file__).parent


def main():
    rows, twin_free = build()
    out = {}
    for model in ("ssl", "aasist"):
        cl = cells(rows, load_scores(model), twin_free, "twin_free")
        rng = np.random.default_rng(103)
        k = int(np.floor((N_CAL + 1) * ALPHA))
        res = {}
        for cond in sorted({c for c, _ in cl}):
            cal_b = cl.get((cond, 0), (np.array([]), np.array([])))[0]
            dep_b, dep_s = cl.get((cond, 1), (np.array([]), np.array([])))
            if len(cal_b) < N_CAL + 50 or len(dep_b) < 50 or len(dep_s) < 50:
                continue
            fprs, fnrs = [], []
            for _ in range(B):
                t = np.sort(cal_b[rng.choice(len(cal_b), N_CAL, replace=False)])[k - 1]
                fprs.append(float(np.mean(dep_b < t)))
                fnrs.append(float(np.mean(dep_s >= t)))
            fpr = float(np.mean(fprs))
            t_or = float(np.quantile(dep_b, ALPHA))
            fnr_or = float(np.mean(dep_s >= t_or))
            res[cond] = {"fpr": fpr, "log2_fpr_ratio": log2_ratio(fpr, len(dep_b)),
                         "fnr": float(np.mean(fnrs)), "fnr_oracle": fnr_or,
                         "fnr_price": float(np.mean(fnrs)) - fnr_or,
                         "oracle_overlap_dominated": bool(fnr_or > OVERLAP_BAR),
                         "n_dep_bona": len(dep_b), "n_dep_spoof": len(dep_s)}
        miss = sum(abs(v["log2_fpr_ratio"]) > 1 for v in res.values())
        out[model] = {"cells": res, "n": len(res), "miss_2x": miss,
                      "max_abs_log2": max(abs(v["log2_fpr_ratio"]) for v in res.values()),
                      "fpr_range": [min(v["fpr"] for v in res.values()), max(v["fpr"] for v in res.values())]}
        print(model, {k: v for k, v in out[model].items() if k != "cells"}, flush=True)
    (HERE / "artifacts" / "a5_diagonal.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
