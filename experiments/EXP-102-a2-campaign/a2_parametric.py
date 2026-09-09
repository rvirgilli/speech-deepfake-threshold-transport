"""EXP-102 addendum cell D1: same-N parametric quantile baseline.

Gaussian fit to the N cohort bona-fide scores, t = mean - 1.645*sd (alpha=5%
lower tail). Same paired draws as the conformal quantile; isolates
distribution-freeness from the renormalization-vs-threshold confound.
"""

import csv
import gzip
import json
from pathlib import Path

import numpy as np

SCORES = Path.home() / "projects/academic/icassp2027/experiments/EXP-001-scoring-campaign/scores"
ALPHA = 0.05
Z_ALPHA = 1.6448536269514722
NS = [30, 100, 300, 500, 1000, 3000, 10000]
B = 1000
SEED = 20260813
TARGETS = ["asv21la", "asv21df_100k", "itw", "brspeech_test"]


def load(model, corpus):
    with gzip.open(SCORES / f"{model}_{corpus}.csv.gz", "rt") as f:
        rows = list(csv.DictReader(f))
    s = np.array([float(r["score"]) for r in rows])
    bona = np.array([r["label"] == "bonafide" for r in rows])
    return s[bona], s[~bona]


def main():
    results = {}
    for model in ["ssl", "aasist"]:
        rng = np.random.default_rng(SEED)
        results[model] = {}
        for corpus in TARGETS:
            bona, spoof = load(model, corpus)
            fnr_o = float(np.mean(spoof >= np.quantile(bona, ALPHA)))
            results[model][corpus] = {}
            for N in NS:
                if N >= len(bona) * 0.8:
                    continue
                k = int(np.floor((N + 1) * ALPHA))
                acc = {"quantile": [], "parametric": []}
                fnr = {"quantile": [], "parametric": []}
                for _ in range(B):
                    idx = rng.choice(len(bona), N, replace=False)
                    cohort = bona[idx]
                    mask = np.ones(len(bona), bool)
                    mask[idx] = False
                    held = bona[mask]
                    ts = {"quantile": float(np.sort(cohort)[k - 1]),
                          "parametric": float(cohort.mean() - Z_ALPHA * cohort.std())}
                    for p, t in ts.items():
                        acc[p].append(float(np.mean(held < t)))
                        fnr[p].append(float(np.mean(spoof >= t)))
                results[model][corpus][N] = {
                    p: {"fpr_mean": round(float(np.mean(acc[p])), 4),
                        "fpr_ci": [round(float(np.percentile(acc[p], 2.5)), 4),
                                   round(float(np.percentile(acc[p], 97.5)), 4)],
                        "fnr_minus_oracle_pts": round(100 * (float(np.mean(fnr[p])) - fnr_o), 2)}
                    for p in acc}
            n5 = results[model][corpus].get(500)
            if n5:
                print(f"{model}/{corpus} N=500: quantile fpr={n5['quantile']['fpr_mean']} "
                      f"parametric fpr={n5['parametric']['fpr_mean']}", flush=True)

    out = Path(__file__).parent / "results_parametric.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
