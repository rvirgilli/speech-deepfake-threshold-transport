"""EXP-102 cell 1 completion (CPU, after sls_score.py): SLS naive-transfer +
BRSpeech rows, plus cohort z-norm and parametric rows at N=500 for the full
policy table. Combines local sls_asv19_dev / sls_brspeech_test scores with the
official releases (21LA, 21DF full, ITW).
"""

import csv
import gzip
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
DATA = Path.home() / "data/corpora/anti-spoofing"
OFF = DATA / "official-scores/xlsr-sls"
ALPHA = 0.05
Z_ALPHA = 1.6448536269514722
N = 500
B = 1000
SEED = 20260813


def load_local(corpus):
    with gzip.open(HERE / f"scores/sls_{corpus}.csv.gz", "rt") as f:
        rows = list(csv.DictReader(f))
    s = np.array([float(r["score"]) for r in rows])
    bona = np.array([r["label"] == "bonafide" for r in rows])
    return s[bona], s[~bona]


def load_official(path, labels):
    d = {Path(k).stem: v for k, v in
         (l.split()[:2] for l in path.read_text().splitlines())}
    utts = [u for u in d if u in labels]
    s = np.array([float(d[u]) for u in utts])
    bona = np.array([labels[u] for u in utts])
    return s[bona], s[~bona]


def labels_21(key, phase_col=7):
    lab = {}
    for line in key.read_text().splitlines():
        p = line.split()
        if p[phase_col] != "hidden":  # eval + progress; see EXP-001/code/protocol.py
            lab[p[1]] = p[5] == "bonafide"
    return lab


def main():
    dev_bona, dev_spoof = load_local("asv19_dev")
    t_naive = float(np.quantile(dev_bona, ALPHA))
    mu_d, sd_d = float(dev_bona.mean()), float(dev_bona.std())
    tstar = (t_naive - mu_d) / sd_d
    print(f"dev: {len(dev_bona)} bona; naive t={t_naive:.3f} "
          f"(dev FPR {np.mean(dev_bona < t_naive):.4f}, FNR {np.mean(dev_spoof >= t_naive):.4f})",
          flush=True)

    with gzip.open(Path.home() / "projects/academic/icassp2027/experiments/"
                   "EXP-001-scoring-campaign/scores/ssl_itw.csv.gz", "rt") as f:
        itw_lab = {r["utt_id"]: r["label"] == "bonafide" for r in csv.DictReader(f)}
    corpora = {
        "asv21la": load_official(OFF / "scores_LA.txt", labels_21(DATA / "keys/LA/CM/trial_metadata.txt")),
        "asv21df_full": load_official(OFF / "scores_DF.txt", labels_21(DATA / "keys/DF/CM/trial_metadata.txt")),
        "itw": load_official(OFF / "scores_Wild.txt", itw_lab),
        "brspeech_test": load_local("brspeech_test"),
    }

    rng = np.random.default_rng(SEED)
    results = {}
    for corpus, (bona, spoof) in corpora.items():
        t_o = float(np.quantile(bona, ALPHA))
        cell = {
            "n_bona": len(bona), "n_spoof": len(spoof),
            "naive_transfer": {"fpr": round(float(np.mean(bona < t_naive)), 4),
                               "fnr": round(float(np.mean(spoof >= t_naive)), 4)},
            "oracle": {"fnr": round(float(np.mean(spoof >= t_o)), 4)},
            # Same-sample quantile/tie identity only; never validation evidence.
            "same_sample_quantile_identity": round(float(np.mean(bona < t_o)), 4),
        }
        k = int(np.floor((N + 1) * ALPHA))
        acc = {p: {"fpr": [], "fnr": []} for p in ("quantile", "znorm", "parametric")}
        for _ in range(B):
            idx = rng.choice(len(bona), N, replace=False)
            cohort = bona[idx]
            mask = np.ones(len(bona), bool)
            mask[idx] = False
            held = bona[mask]
            ts = {"quantile": float(np.sort(cohort)[k - 1]),
                  "znorm": float(cohort.mean() + cohort.std() * tstar),
                  "parametric": float(cohort.mean() - Z_ALPHA * cohort.std())}
            for p, t in ts.items():
                acc[p]["fpr"].append(float(np.mean(held < t)))
                acc[p]["fnr"].append(float(np.mean(spoof >= t)))
        for p, v in acc.items():
            cell[p] = {"fpr_mean": round(float(np.mean(v["fpr"])), 4),
                       "fnr_mean": round(float(np.mean(v["fnr"])), 4),
                       "fnr_minus_oracle_pts": round(
                           100 * (float(np.mean(v["fnr"])) - cell["oracle"]["fnr"]), 2)}
        results[corpus] = cell
        print(f"sls/{corpus}: naive FPR={cell['naive_transfer']['fpr']} "
              f"q={cell['quantile']['fpr_mean']} z={cell['znorm']['fpr_mean']} "
              f"par={cell['parametric']['fpr_mean']} oracleFNR={cell['oracle']['fnr']}",
              flush=True)

    out = HERE / "results_sls_complete.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
