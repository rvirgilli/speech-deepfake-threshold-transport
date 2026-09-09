"""EXP-102 cells 3-4: cost-of-labels N-sweep + calibration-set contamination.

Cell 3: quantile vs cohort z-norm, N in {30,100,300,1000,3000,10000}, B=1000;
realized FPR tightness (mean, CI, MAD from target) and FNR-oracle gap. The
Beta(k, N+1-k) mean and 95% band are an iid-continuous population reference,
not the exact law of the implemented finite-pool draw without replacement and
evaluation on its complement.
Cell 4: quantile with 1/2/5% spoof contamination in the calibration draw.
"""

import csv
import gzip
import json
from pathlib import Path

import numpy as np

SCORES = Path.home() / "projects/academic/icassp2027/experiments/EXP-001-scoring-campaign/scores"
ALPHA = 0.05
NS = [30, 100, 300, 1000, 3000, 10000]
CONTAM = [0.01, 0.02, 0.05]
CONTAM_NS = [100, 500]
B = 1000
SEED = 20260813
TARGETS = ["asv21la", "asv21df_100k", "itw", "brspeech_test"]


def load(model, corpus):
    with gzip.open(SCORES / f"{model}_{corpus}.csv.gz", "rt") as f:
        rows = list(csv.DictReader(f))
    s = np.array([float(r["score"]) for r in rows])
    bona = np.array([r["label"] == "bonafide" for r in rows])
    return s[bona], s[~bona]


def beta_band(N, alpha):
    """Theoretical iid-continuous population-FPR reference.

    The empirical experiment below samples without replacement from a fixed
    finite score pool and evaluates on the complementary pool. Its conditional
    rank distribution is finite-population, not exactly this Beta law.
    """
    from math import lgamma

    k = int(np.floor((N + 1) * alpha))
    # Population FPR ~ Beta(k, N + 1 - k) only under iid continuous sampling
    # from a fixed distribution.
    a, b = k, N + 1 - k
    mean = a / (a + b)
    # percentiles via numpy sampling (avoids scipy dependency)
    rng = np.random.default_rng(0)
    draws = rng.beta(a, b, 100_000)
    return {"kind": "iid_continuous_population_reference",
            "assumptions": "iid continuous sampling from a fixed score distribution",
            "not_exact_for": "finite-pool draws without replacement evaluated on the complement",
            "k": k, "mean": round(float(mean), 4),
            "band95": [round(float(np.percentile(draws, 2.5)), 4),
                       round(float(np.percentile(draws, 97.5)), 4)]}


def main():
    results = {"n_sweep": {}, "contamination": {}}
    for model in ["ssl", "aasist"]:
        dev_bona, _ = load(model, "asv19_dev")
        mu_d, sd_d = float(dev_bona.mean()), float(dev_bona.std())
        tstar = (float(np.quantile(dev_bona, ALPHA)) - mu_d) / sd_d
        rng = np.random.default_rng(SEED)
        for corpus in TARGETS:
            bona, spoof = load(model, corpus)
            fnr_o = float(np.mean(spoof >= np.quantile(bona, ALPHA)))
            key = f"{model}/{corpus}"
            results["n_sweep"][key] = {}
            for N in NS:
                if N >= len(bona) * 0.8:
                    results["n_sweep"][key][N] = "insufficient bona fide"
                    continue
                k = int(np.floor((N + 1) * ALPHA))
                if k < 1:
                    results["n_sweep"][key][N] = "N too small for alpha"
                    continue
                acc = {"quantile": ([], []), "znorm": ([], [])}
                for _ in range(B):
                    idx = rng.choice(len(bona), N, replace=False)
                    cohort = bona[idx]
                    mask = np.ones(len(bona), bool)
                    mask[idx] = False
                    held = bona[mask]
                    ts = {"quantile": float(np.sort(cohort)[k - 1]),
                          "znorm": float(cohort.mean() + cohort.std() * tstar)}
                    for p, t in ts.items():
                        acc[p][0].append(float(np.mean(held < t)))
                        acc[p][1].append(float(np.mean(spoof >= t)))
                cell = {
                    "iid_beta_reference": beta_band(N, ALPHA),
                    "empirical_design": "fixed pool; cohort sampled without replacement; evaluation on complement",
                }
                for p, (fprs, fnrs) in acc.items():
                    fprs, fnrs = np.array(fprs), np.array(fnrs)
                    cell[p] = {
                        "fpr_mean": round(float(fprs.mean()), 4),
                        "fpr_ci": [round(float(np.percentile(fprs, 2.5)), 4),
                                   round(float(np.percentile(fprs, 97.5)), 4)],
                        "fpr_mad_from_target": round(float(np.mean(np.abs(fprs - ALPHA))), 4),
                        "fnr_minus_oracle_pts": round(100 * (float(fnrs.mean()) - fnr_o), 2),
                    }
                results["n_sweep"][key][N] = cell
            # Cell 4: contamination (quantile policy only, per PREREG).
            results["contamination"][key] = {}
            for N in CONTAM_NS:
                k = int(np.floor((N + 1) * ALPHA))
                for rate in CONTAM:
                    n_sp = int(round(N * rate))
                    fprs, fnrs = [], []
                    for _ in range(B):
                        idx = rng.choice(len(bona), N - n_sp, replace=False)
                        sp_idx = rng.choice(len(spoof), n_sp, replace=False)
                        cohort = np.concatenate([bona[idx], spoof[sp_idx]])
                        t = float(np.sort(cohort)[k - 1])
                        mask = np.ones(len(bona), bool)
                        mask[idx] = False
                        fprs.append(float(np.mean(bona[mask] < t)))
                        fnrs.append(float(np.mean(spoof >= t)))
                    results["contamination"][key][f"N{N}_c{rate}"] = {
                        "fpr_mean": round(float(np.mean(fprs)), 4),
                        "fpr_ci": [round(float(np.percentile(fprs, 2.5)), 4),
                                   round(float(np.percentile(fprs, 97.5)), 4)],
                        "fnr_mean": round(float(np.mean(fnrs)), 4),
                    }
            print(f"{key}: N-sweep + contamination done", flush=True)

    out = Path(__file__).parent / "results_nsweep.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
