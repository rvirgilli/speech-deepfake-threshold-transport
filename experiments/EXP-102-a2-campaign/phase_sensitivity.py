"""Sensitivity of the drift-map counts to the ASVspoof 2021 phase selection.

Recomputes the vanilla conformal-quantile cells of drift_map.py (same seed,
N_CAL and B; no weighting, monitors or ACI) under three trial selections and
reports the headline counts and the flagship cell for each. `eval+progress`
is the protocol of every reported result (EXP-001/code/protocol.py).
"""

import json
from pathlib import Path

import numpy as np

import drift_map as dm

HERE = Path(__file__).parent
KEYS = Path.home() / "data/corpora/anti-spoofing/keys"
PHASE = {}
for track in ("LA", "DF"):
    for line in (KEYS / f"{track}/CM/trial_metadata.txt").read_text().splitlines():
        p = line.split()
        PHASE[p[1]] = p[7]
SELECTIONS = {"all_phases": None, "eval+progress": {"eval", "progress"}, "eval_only": {"eval"}}
_raw_load = dm.load_scores.__wrapped__ if hasattr(dm.load_scores, "__wrapped__") else None


def raw_scores(model, corpus):
    import csv, gzip
    with gzip.open(dm.EXP001 / f"{model}_{corpus}.csv.gz", "rt") as f:
        return {r["utt_id"]: (float(r["score"]), r["label"] == "bonafide") for r in csv.DictReader(f)}


def cell(rng, cal, dep):
    cal_b, _ = cal
    dep_b, dep_s = dep
    k = int(np.floor((dm.N_CAL + 1) * dm.ALPHA))
    fpr, fnr = [], []
    for _ in range(dm.B):
        thr = np.sort(cal_b[rng.choice(len(cal_b), dm.N_CAL, replace=False)])[k - 1]
        fpr.append(float(np.mean(dep_b < thr)))
        fnr.append(float(np.mean(dep_s >= thr)))
    f = float(np.mean(fpr))
    t_o = float(np.quantile(dep_b, dm.ALPHA))
    return {"fpr": round(f, 4), "fnr": round(float(np.mean(fnr)), 4),
            "fnr_oracle": round(float(np.mean(dep_s >= t_o)), 4),
            "log2": round(dm.log2_fpr_ratio(f, len(dep_b)), 4)}


def run(selection):
    def load(model, corpus):
        d = raw_scores(model, corpus)
        if selection is not None and corpus in ("asv21la", "asv21df_full"):
            d = {u: v for u, v in d.items() if PHASE.get(u) in selection}
        return d
    dm.load_scores = load
    cells = {}
    for model in dm.MODELS:
        rng = np.random.default_rng(dm.SEED)
        la = dm.la_channel_slices(model)
        for a in sorted(la):
            for b in sorted(la):
                if a != b:
                    cells[f"{model}/{a}->{b}"] = cell(rng, la[a], la[b])
        cs = dm.cross_slices(model)
        for a in dm.CROSS:
            for b in dm.CROSS:
                if a != b:
                    cells[f"{model}/{a}->{b}"] = cell(rng, cs[a], cs[b])
    miss = [k for k, v in cells.items() if abs(v["log2"]) > dm.SEVERITY_BAR]
    band = [k for k, v in cells.items() if abs(v["fpr"] - dm.ALPHA) <= dm.ALPHA]
    within = [k for k in cells if "->" in k and not any(c in k for c in dm.CROSS)]
    return {"n_cells": len(cells), "miss_2x": len(miss),
            "within_miss_2x": sum(k in within for k in miss),
            "in_band": len(band), "hidden_in_band": sum(k in band for k in miss),
            "flagship_aasist_pstn_g722": cells["aasist/pstn->g722"]}


def main():
    out = {name: run(sel) for name, sel in SELECTIONS.items()}
    (HERE / "results_phase_sensitivity.json").write_text(json.dumps(out, indent=2) + "\n")
    for name, r in out.items():
        print(name, r, flush=True)


if __name__ == "__main__":
    main()
