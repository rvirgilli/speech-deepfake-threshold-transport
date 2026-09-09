"""EXP-002: bona-fide-quantile threshold calibration vs naive transfer (A2 pilot).

Decision rule: flag as spoof when score < threshold (higher score = bona fide).
FPR = P(bona flagged), FNR = P(spoof passed). Conformal quantile threshold from
N target bona-fide samples: k-th smallest bona score, k = floor((N+1)*alpha).
"""

import csv
import os
import gzip
import json
from pathlib import Path

import numpy as np

SCORES = Path(os.environ.get("A2_SCORES", Path.home() / "projects/academic/icassp2027/experiments/EXP-001-scoring-campaign/scores"))
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "EXP-001-scoring-campaign/code"))
from protocol import drop_hidden  # noqa: E402

ALPHA = 0.05
NS = [50, 100, 500, 1000]
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
        results[model] = {"naive_threshold_from_19la_dev": t_naive,
                          "dev_check_fpr_fnr": rates(t_naive, dev_bona, dev_spoof)}
        rng = np.random.default_rng(SEED)
        for corpus in TARGETS:
            bona, spoof = load(model, corpus)
            t_oracle = float(np.quantile(bona, ALPHA))
            fpr_o, fnr_o = rates(t_oracle, bona, spoof)
            fpr_n, fnr_n = rates(t_naive, bona, spoof)
            cell = {
                "n_bona": len(bona), "n_spoof": len(spoof),
                "naive_transfer": {"fpr": float(fpr_n), "fnr": float(fnr_n)},
                "oracle": {"threshold": t_oracle, "fnr": float(fnr_o)},
                # The same bona-fide scores define and read back t_oracle. This
                # records the quantile/tie convention; it is not evidence.
                "same_sample_quantile_identity": float(fpr_o),
                "quantile": {},
            }
            for N in NS:
                if N >= len(bona):
                    cell["quantile"][N] = "insufficient bona fide"
                    continue
                k = int(np.floor((N + 1) * ALPHA))
                if k < 1:
                    cell["quantile"][N] = "N too small for alpha"
                    continue
                fprs, fnrs = [], []
                for _ in range(B):
                    idx = rng.choice(len(bona), N, replace=False)
                    t = np.sort(bona[idx])[k - 1]
                    mask = np.ones(len(bona), bool)
                    mask[idx] = False
                    fprs.append(np.mean(bona[mask] < t))
                    fnrs.append(np.mean(spoof >= t))
                cell["quantile"][N] = {
                    "fpr_mean": float(np.mean(fprs)),
                    "fpr_ci": [float(np.percentile(fprs, 2.5)),
                               float(np.percentile(fprs, 97.5))],
                    "fnr_mean": float(np.mean(fnrs)),
                    "fnr_ci": [float(np.percentile(fnrs, 2.5)),
                               float(np.percentile(fnrs, 97.5))],
                    "fnr_minus_oracle_pts": 100 * (float(np.mean(fnrs)) - fnr_o),
                }
            results[model][corpus] = cell
            q = cell["quantile"].get(500)
            print(f"{model}/{corpus}: naive FPR={fpr_n:.3f} FNR={fnr_n:.3f} | "
                  f"oracle FNR={fnr_o:.3f} | N=500 {q}", flush=True)

    out = Path(__file__).parent / "results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
