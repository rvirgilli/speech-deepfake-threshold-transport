"""EXP-010: same-N bona-fide-cohort normalization vs conformal quantile (A2).

All policies spend the identical N labeled target bona-fide samples (paired
cohort draws). Decision rule: flag as spoof when score < threshold.
  - quantile: k-th smallest cohort score, k = floor((N+1)*alpha).
  - znorm:    t = mu_N + sd_N * t*, t* = alpha-quantile of dev bona scores
              standardized by dev-bona mean/std.
  - robust:   same with median and MAD*1.4826.
Realized FPR measured on bona fide held out from the cohort draw.
"""

import csv
import gzip
import json
from pathlib import Path

import numpy as np

SCORES = Path.home() / "projects/academic/icassp2027/experiments/EXP-001-scoring-campaign/scores"
import sys
sys.path.insert(0, str(SCORES.parent / "code"))
from protocol import drop_hidden  # noqa: E402

ALPHA = 0.05
NS = [100, 500]
B = 1000
SEED = 20260813
TARGETS = ["asv21la", "asv21df_full", "itw", "brspeech_test"]


def load(model, corpus):
    with gzip.open(SCORES / f"{model}_{corpus}.csv.gz", "rt") as f:
        rows = drop_hidden(list(csv.DictReader(f)))
    s = np.array([float(r["score"]) for r in rows])
    bona = np.array([r["label"] == "bonafide" for r in rows])
    return s[bona], s[~bona]


def rates(t, bona, spoof):
    return float(np.mean(bona < t)), float(np.mean(spoof >= t))


def main():
    results = {}
    for model in ["ssl", "aasist"]:
        dev_bona, dev_spoof = load(model, "asv19_dev")
        t_naive = float(np.quantile(dev_bona, ALPHA))
        mu_d, sd_d = float(dev_bona.mean()), float(dev_bona.std())
        med_d = float(np.median(dev_bona))
        mad_d = float(np.median(np.abs(dev_bona - med_d))) * 1.4826
        tstar_z = (t_naive - mu_d) / sd_d
        tstar_r = (t_naive - med_d) / mad_d
        results[model] = {"t_naive": t_naive, "tstar_znorm": tstar_z, "tstar_robust": tstar_r}
        rng = np.random.default_rng(SEED)
        for corpus in TARGETS:
            bona, spoof = load(model, corpus)
            t_oracle = float(np.quantile(bona, ALPHA))
            fpr_o, fnr_o = rates(t_oracle, bona, spoof)
            cell = {"n_bona": len(bona), "n_spoof": len(spoof),
                    "naive_transfer": dict(zip(("fpr", "fnr"), rates(t_naive, bona, spoof))),
                    "oracle": {"fpr": fpr_o, "fnr": fnr_o}}
            for N in NS:
                k = int(np.floor((N + 1) * ALPHA))
                acc = {p: {"fpr": [], "fnr": []} for p in ("quantile", "znorm", "robust")}
                for _ in range(B):
                    idx = rng.choice(len(bona), N, replace=False)
                    cohort = bona[idx]
                    mask = np.ones(len(bona), bool)
                    mask[idx] = False
                    held = bona[mask]
                    med_c = float(np.median(cohort))
                    mad_c = float(np.median(np.abs(cohort - med_c))) * 1.4826
                    ts = {"quantile": float(np.sort(cohort)[k - 1]),
                          "znorm": float(cohort.mean() + cohort.std() * tstar_z),
                          "robust": med_c + mad_c * tstar_r}
                    for p, t in ts.items():
                        fpr, fnr = rates(t, held, spoof)
                        acc[p]["fpr"].append(fpr)
                        acc[p]["fnr"].append(fnr)
                cell[N] = {}
                for p in acc:
                    fprs, fnrs = np.array(acc[p]["fpr"]), np.array(acc[p]["fnr"])
                    cell[N][p] = {
                        "fpr_mean": round(float(fprs.mean()), 4),
                        "fpr_ci": [round(float(np.percentile(fprs, 2.5)), 4),
                                   round(float(np.percentile(fprs, 97.5)), 4)],
                        "fnr_mean": round(float(fnrs.mean()), 4),
                        "fnr_minus_oracle_pts": round(100 * (float(fnrs.mean()) - fnr_o), 2),
                    }
            results[model][corpus] = cell
            n5 = cell[500]
            print(f"{model}/{corpus}: N=500 FPR quantile={n5['quantile']['fpr_mean']:.3f} "
                  f"znorm={n5['znorm']['fpr_mean']:.3f} robust={n5['robust']['fpr_mean']:.3f} | "
                  f"FNR-oracle(pts) q={n5['quantile']['fnr_minus_oracle_pts']} "
                  f"z={n5['znorm']['fnr_minus_oracle_pts']} r={n5['robust']['fnr_minus_oracle_pts']}",
                  flush=True)

    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
