"""EXP-110: per-checkpoint 21LA transport map and its primary statistic K.

K = the number of the 42 ordered condition pairs whose realized FPR misses the alpha
target by more than a factor of two. Cell statistics come from EXP-102's audited
`run_cell`; slices are built exactly as in EXP-109 cell A, over all 181,566 trials.
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "EXP-102-a2-campaign"))
from drift_map import ALPHA, SEED, run_cell  # noqa: E402

LA_KEY = Path.home() / "data/corpora/anti-spoofing/keys/LA/CM/trial_metadata.txt"
ARTIFACTS = Path("/home/rv/exp-artifacts/icassp2027/EXP-110")
# arm 1 scored its tied checkpoints into arm1/epoch_<e>; arm 2 (run_arm2.sh) into arm2/scores/epoch_<e>.
# Seed-factorial runs (run_seed.sh) follow arm 2's layout under <arm>_s<seed>/.
TIED = {"arm1": [20, 21, 27, 28, 29]}


def score_root(arm):
    return ARTIFACTS / "arm1" if arm == "arm1" else ARTIFACTS / arm / "scores"


def tied_epochs(arm):
    """Arm 1's tied epochs are fixed; every other run's come from its dev sweep with pick_tied's rule."""
    if arm in TIED:
        return TIED[arm]
    eer = json.load(open(ARTIFACTS / arm / "results_stage0.json"))["dev_eer_by_epoch"]
    ranked = sorted(eer.items(), key=lambda kv: (kv[1], int(kv[0])))[:5]
    return sorted(int(e) for e, _ in ranked)


def key_table():
    out = {}
    for line in LA_KEY.read_text().splitlines():
        p = line.split()
        if p[7] == "hidden":  # VAD-trimmed phase, excluded since EXP-102 AMENDMENT-3
            continue
        out[p[1]] = (p[2], p[5] == "bonafide")
    return out


def slices_for(arm, epoch, keys):
    d = score_root(arm) / f"epoch_{epoch}"
    chunks = sorted(d.glob("chunk_*.npz"))
    utts, scores = [], []
    for c in chunks:
        z = np.load(c, allow_pickle=True)
        utts.extend(z["utt_id"])
        scores.extend(z["score"])
    if len(utts) != 181566:  # every scored 21LA trial; the phase filter is applied by key lookup
        return None, len(utts)
    sl = {}
    for u, s in zip(utts, scores):
        meta = keys.get(str(u))
        if meta is None:
            continue
        cond, bona = meta
        sl.setdefault(cond, ([], []))[0 if bona else 1].append(float(s))
    return {c: (np.array(b), np.array(sp)) for c, (b, sp) in sl.items()}, len(utts)


def main():
    arm = sys.argv[1] if len(sys.argv) > 1 else "arm1"
    keys = key_table()
    out = {}
    for e in tied_epochs(arm):
        sl, n = slices_for(arm, e, keys)
        if sl is None:
            print(f"epoch_{e}: incomplete ({n}/181566) — skipped")
            continue
        rng = np.random.default_rng(SEED)
        cells = {}
        for cal in sorted(sl):
            for dep in sorted(sl):
                if cal == dep:
                    continue
                c = run_cell(rng, sl[cal], sl[dep])
                if c:
                    cells[f"{cal}->{dep}"] = c
        K = sum(1 for v in cells.values() if abs(v["log2_fpr_ratio"]) > 1.0)
        cons = sum(1 for v in cells.values() if v["log2_fpr_ratio"] < -1.0)
        out[f"epoch_{e}"] = {"n_cells": len(cells), "K": K, "conservative": cons,
                             "liberal": K - cons,
                             "worst_abs_log2": round(max(abs(v["log2_fpr_ratio"])
                                                         for v in cells.values()), 3),
                             "cells": cells}
        print(f"epoch_{e}: {len(cells)} cells, K={K} "
              f"(conservative {cons}, liberal {K - cons}), worst |log2|="
              f"{out[f'epoch_{e}']['worst_abs_log2']}")

    if out:
        Ks = [v["K"] for v in out.values()]
        print(f"\n{arm}: K median {int(np.median(Ks))}, range {min(Ks)}-{max(Ks)} "
              f"over {len(Ks)} tied checkpoints")
        Path(f"results_{arm}.json").write_text(json.dumps(
            {"alpha": ALPHA, "seed": SEED, "arm": arm, "tied_epochs": tied_epochs(arm), "epochs": out,
             "K_values": Ks, "K_median": float(np.median(Ks)),
             "K_range": [min(Ks), max(Ks)]}, indent=1))


if __name__ == "__main__":
    main()
