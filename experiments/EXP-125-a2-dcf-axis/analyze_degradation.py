"""EXP-125 variants V2 and V3: Rubio et al.'s degradation estimand on the same grid.

V2 keeps A2's transported threshold and changes only the reference: cost on the deployment
condition against cost on the calibration condition, at the same threshold.
V3 additionally selects the threshold by minimizing cost on the calibration condition, as a
cost-optimal-threshold study does. See the PREREG amendment of 2026-09-10.
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
EXP = HERE.parent
sys.path.insert(0, str(EXP / "EXP-102-a2-campaign"))
from drift_map import ALPHA, B, N_CAL, SEED, la_channel_slices  # noqa: E402
from analyze import DETECTORS, PRIORS, dcf, optimal_dcf, rates  # noqa: E402


def main():
    out = {"alpha": ALPHA, "n_cal": N_CAL, "b_draws": B, "seed": SEED, "priors": list(PRIORS),
           "estimand": "degradation: (DCF on deployment - DCF on calibration) / DCF on calibration, one threshold",
           "detectors": {}}
    for det in DETECTORS:
        sl = la_channel_slices(det)
        conds = sorted(sl)
        srt = {c: (np.sort(sl[c][0]), np.sort(sl[c][1])) for c in conds}
        rng = np.random.default_rng(SEED)
        k = int(np.floor((N_CAL + 1) * ALPHA))
        thr = {}
        for c in conds:
            b = sl[c][0]
            thr[c] = np.array([np.sort(b[rng.choice(len(b), N_CAL, replace=False)])[k - 1] for _ in range(B)])
        # V3's threshold: the cost-optimal one on the calibration condition, per prior
        t_opt = {(c, pi): optimal_dcf(*srt[c], pi)[1] for c in conds for pi in PRIORS}
        cells = {}
        for cal in conds:
            for dep in conds:
                if cal == dep:
                    continue
                fpr_d, fnr_d = rates(thr[cal], *srt[dep])
                fpr_c, fnr_c = rates(thr[cal], *srt[cal])
                entry = {"log2_fpr_ratio": float(np.log2(max(fpr_d.mean(), 3.0 / len(srt[dep][0])) / ALPHA)), "by_prior": {}}
                for pi in PRIORS:
                    dep_cost = dcf(fpr_d, fnr_d, pi).mean()
                    cal_cost = dcf(fpr_c, fnr_c, pi).mean()
                    t3 = t_opt[(cal, pi)]
                    f3d, n3d = rates(np.array([t3]), *srt[dep])
                    f3c, n3c = rates(np.array([t3]), *srt[cal])
                    d3_dep, d3_cal = float(dcf(f3d, n3d, pi)[0]), float(dcf(f3c, n3c, pi)[0])
                    entry["by_prior"][str(pi)] = {
                        "v2_dcf_deployment": float(dep_cost), "v2_dcf_calibration": float(cal_cost),
                        "v2_delta": float((dep_cost - cal_cost) / cal_cost) if cal_cost > 0 else None,
                        "v3_threshold": t3, "v3_dcf_deployment": d3_dep, "v3_dcf_calibration": d3_cal,
                        "v3_delta": (d3_dep - d3_cal) / d3_cal if d3_cal > 0 else None,
                        "v3_fpr_deployment": float(f3d[0]), "v3_fnr_deployment": float(n3d[0])}
                cells[f"{cal}->{dep}"] = entry
        out["detectors"][det] = {"cells": cells, "summary_by_prior": {}}
        for pi in PRIORS:
            key = str(pi)
            cons = [v for v in cells.values() if v["log2_fpr_ratio"] < -1]
            lib = [v for v in cells.values() if v["log2_fpr_ratio"] > 1]
            s = {"n_conservative": len(cons), "n_liberal": len(lib)}
            for v in ("v2", "v3"):
                dd = np.array([c[v + "_delta"] for c in (x["by_prior"][key] for x in cells.values())])
                names = np.array(list(cells))
                worse = set(names[np.argsort(-dd)[: len(dd) // 2]])
                mc = [x["by_prior"][key][v + "_delta"] for x in cons]
                ml = [x["by_prior"][key][v + "_delta"] for x in lib]
                s[v] = {"median_conservative": float(np.median(mc)) if mc else None,
                        "median_liberal": float(np.median(ml)) if ml else None,
                        "median_all": float(np.median(dd)),
                        "n_conservative_negative": int(sum(1 for x in mc if x < 0)),
                        "n_liberal_negative": int(sum(1 for x in ml if x < 0)),
                        "frac_conservative_in_worse_half": sum(1 for k2, x in cells.items() if x in cons and k2 in worse) / len(cons) if cons else None,
                        "frac_liberal_in_worse_half": sum(1 for k2, x in cells.items() if x in lib and k2 in worse) / len(lib) if lib else None}
            out["detectors"][det]["summary_by_prior"][key] = s
            print(f"{det} pi={pi}: V2 cons median {s['v2']['median_conservative']:+.3f} (neg {s['v2']['n_conservative_negative']}/{len(cons)}) "
                  f"lib {s['v2']['median_liberal']:+.3f} | cons worse-half {s['v2']['frac_conservative_in_worse_half']:.2f} || "
                  f"V3 cons median {s['v3']['median_conservative']:+.3f} (neg {s['v3']['n_conservative_negative']}/{len(cons)}) "
                  f"lib {s['v3']['median_liberal']:+.3f} | cons worse-half {s['v3']['frac_conservative_in_worse_half']:.2f}", flush=True)
    (HERE / "results_degradation.json").write_text(json.dumps(out, indent=1))
    print("wrote results_degradation.json")


if __name__ == "__main__":
    main()
