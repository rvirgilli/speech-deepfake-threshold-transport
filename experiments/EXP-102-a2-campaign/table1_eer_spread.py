"""Discrimination reference and calibration dispersion for Table 1's twelve cells.

Same score sets as the Table 1 entries (EXP-002 for AASIST and SSL-AASIST;
a2_sls_complete.py for XLS-R+SLS). Per cell: EER, and the empirical 2.5/97.5
percentiles of realized FPR at the N=500 conformal quantile over B=1000
without-replacement cohort draws, evaluated on the held-out bona fide.
"""

import csv
import gzip
import json
from pathlib import Path

import numpy as np

from a2_sls_complete import labels_21, load_local, load_official, DATA, OFF

HERE = Path(__file__).parent
E001 = HERE.parent / "EXP-001-scoring-campaign/scores"
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "EXP-001-scoring-campaign/code"))
from protocol import drop_hidden  # noqa: E402

ALPHA, N, B, SEED = 0.05, 500, 1000, 20260813
OUT = HERE / "results_table1_eer_spread.json"


def load_e001(model, corpus):
    with gzip.open(E001 / f"{model}_{corpus}.csv.gz", "rt") as f:
        rows = drop_hidden(list(csv.DictReader(f)))
    s = np.array([float(r["score"]) for r in rows])
    bona = np.array([r["label"] == "bonafide" for r in rows])
    return s[bona], s[~bona]


def eer(bona, spoof):
    """Spoof flagged when score < t. EER at the crossing of FPR(t) and FNR(t)."""
    ts = np.unique(np.concatenate([bona, spoof]))
    fpr = np.searchsorted(np.sort(bona), ts, side="left") / len(bona)
    fnr = 1 - np.searchsorted(np.sort(spoof), ts, side="left") / len(spoof)
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)


def spread(bona, rng):
    k = int(np.floor((N + 1) * ALPHA))
    fprs = []
    for _ in range(B):
        idx = rng.choice(len(bona), N, replace=False)
        t = np.sort(bona[idx])[k - 1]
        held = np.delete(bona, idx)
        fprs.append(float(np.mean(held < t)))
    return [round(float(np.percentile(fprs, 2.5)), 4), round(float(np.percentile(fprs, 97.5)), 4)]


def main():
    with gzip.open(E001 / "ssl_itw.csv.gz", "rt") as f:
        itw_lab = {r["utt_id"]: r["label"] == "bonafide" for r in csv.DictReader(f)}
    sets = {}
    for model in ("aasist", "ssl"):
        for corpus in ("asv21la", "asv21df_full", "itw", "brspeech_test"):
            sets[f"{model}/{corpus}"] = load_e001(model, corpus)
    sets["sls/asv21la"] = load_official(OFF / "scores_LA.txt", labels_21(DATA / "keys/LA/CM/trial_metadata.txt"))
    sets["sls/asv21df_full"] = load_official(OFF / "scores_DF.txt", labels_21(DATA / "keys/DF/CM/trial_metadata.txt"))
    sets["sls/itw"] = load_official(OFF / "scores_Wild.txt", itw_lab)
    sets["sls/brspeech_test"] = load_local("brspeech_test")

    # Per-condition EER on 21LA: the pooled 21LA EER mixes seven bona-fide score
    # distributions, so it overstates within-condition discrimination.
    condition = {}
    for line in (DATA / "keys/LA/CM/trial_metadata.txt").read_text().splitlines():
        p = line.split()
        condition[p[1]] = p[2]
    by_condition = {}
    for model in ("aasist", "ssl"):
        with gzip.open(E001 / f"{model}_asv21la.csv.gz", "rt") as f:
            rows = drop_hidden(list(csv.DictReader(f)))
        by_condition[f"{model}/asv21la"] = {}
        for cond in sorted({condition[r["utt_id"]] for r in rows}):
            sub = [r for r in rows if condition[r["utt_id"]] == cond]
            b = np.array([float(r["score"]) for r in sub if r["label"] == "bonafide"])
            sp = np.array([float(r["score"]) for r in sub if r["label"] != "bonafide"])
            by_condition[f"{model}/asv21la"][cond] = round(eer(b, sp), 4)
    la_lab = labels_21(DATA / "keys/LA/CM/trial_metadata.txt")
    raw = {Path(k).stem: float(v) for k, v in (l.split()[:2] for l in (OFF / "scores_LA.txt").read_text().splitlines())}
    by_condition["sls/asv21la"] = {}
    for cond in sorted(set(condition[u] for u in la_lab)):
        b = np.array([raw[u] for u in la_lab if condition[u] == cond and la_lab[u] and u in raw])
        sp = np.array([raw[u] for u in la_lab if condition[u] == cond and not la_lab[u] and u in raw])
        by_condition["sls/asv21la"][cond] = round(eer(b, sp), 4)

    rng = np.random.default_rng(SEED)
    results = {}
    for cell, (bona, spoof) in sets.items():
        results[cell] = {"n_bona": int(len(bona)), "n_spoof": int(len(spoof)),
                         "eer": round(eer(bona, spoof), 4),
                         "quantile_N500_fpr_pct_2.5_97.5": spread(bona, rng)}
        print(cell, results[cell], flush=True)
    OUT.write_text(json.dumps({"alpha": ALPHA, "N": N, "B": B, "seed": SEED, "cells": results,
                               "eer_by_condition_21la": by_condition}, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
