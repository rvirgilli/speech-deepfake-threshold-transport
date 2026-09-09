"""Historical EXP-102 cell 1 precursor: XLS-R+SLS rows from official scores.

Official per-trial releases cover 21LA / 21DF (full 611k) / ITW — enough for the
quantile, cohort-norm and oracle policies (target-side only). The naive-transfer
row and BRSpeech-DF required 19LA-dev + BRSpeech scoring, so this script emitted
only the then-available quantile (N in {100,500}) and oracle rows. That work is
complete, not pending: `a2_sls_complete.py` is the current implementation and
`results_sls_complete.json` is the complete artifact used by the manuscript.
"""

import csv
import gzip
import json
from pathlib import Path

import numpy as np

DATA = Path.home() / "data/corpora/anti-spoofing"
OFF = DATA / "official-scores/xlsr-sls"
ALPHA = 0.05
NS = [100, 500]
B = 1000
SEED = 20260813


def eval_labels_21(key_path, phase_col=7):
    lab = {}
    for line in key_path.read_text().splitlines():
        p = line.split()
        if p[phase_col] == "eval":
            lab[p[1]] = p[5] == "bonafide"
    return lab


def itw_labels():
    with gzip.open(Path.home() / "projects/academic/icassp2027/experiments/"
                   "EXP-001-scoring-campaign/scores/ssl_itw.csv.gz", "rt") as f:
        return {r["utt_id"]: r["label"] == "bonafide" for r in csv.DictReader(f)}


def main():
    corpora = {
        "asv21la": (OFF / "scores_LA.txt", eval_labels_21(DATA / "keys/LA/CM/trial_metadata.txt")),
        "asv21df_full": (OFF / "scores_DF.txt", eval_labels_21(DATA / "keys/DF/CM/trial_metadata.txt")),
        "itw": (OFF / "scores_Wild.txt", itw_labels()),
    }
    rng = np.random.default_rng(SEED)
    results = {}
    for corpus, (path, lab) in corpora.items():
        d = {Path(k).stem: v for k, v in
             (l.split()[:2] for l in path.read_text().splitlines())}
        utts = [u for u in d if u in lab]
        s = np.array([float(d[u]) for u in utts])
        bona_mask = np.array([lab[u] for u in utts])
        bona, spoof = s[bona_mask], s[~bona_mask]
        t_o = float(np.quantile(bona, ALPHA))
        cell = {"n_bona": len(bona), "n_spoof": len(spoof),
                "pooled_eer_check": None,
                "oracle": {"fnr": round(float(np.mean(spoof >= t_o)), 4)},
                # Same-sample quantile/tie identity only; never validation evidence.
                "same_sample_quantile_identity": round(float(np.mean(bona < t_o)), 4)}
        for N in NS:
            k = int(np.floor((N + 1) * ALPHA))
            fprs, fnrs = [], []
            for _ in range(B):
                idx = rng.choice(len(bona), N, replace=False)
                t = float(np.sort(bona[idx])[k - 1])
                mask = np.ones(len(bona), bool)
                mask[idx] = False
                fprs.append(float(np.mean(bona[mask] < t)))
                fnrs.append(float(np.mean(spoof >= t)))
            cell[f"quantile_N{N}"] = {
                "fpr_mean": round(float(np.mean(fprs)), 4),
                "fpr_ci": [round(float(np.percentile(fprs, 2.5)), 4),
                           round(float(np.percentile(fprs, 97.5)), 4)],
                "fnr_mean": round(float(np.mean(fnrs)), 4),
                "fnr_minus_oracle_pts": round(100 * (float(np.mean(fnrs)) - cell["oracle"]["fnr"]), 2),
            }
        results[corpus] = cell
        print(f"sls/{corpus}: oracle fnr={cell['oracle']['fnr']} "
              f"N500 {cell['quantile_N500']}", flush=True)

    out = Path(__file__).parent / "results_sls_official.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
