"""EXP-109 cell A: 21LA real-telephony transport map for the released-score detectors.

Six of the seven 21LA conditions are real transmission (Asterisk PBX VoIP; `pstn` a
real carrier); `none` is the untransmitted source. Each detector contributes 42
ordered (calibration, deployment) condition pairs.

Per-cell statistics come from EXP-102's audited `run_cell`; this module only supplies
slices built from released per-trial score files, which are a separate provenance
stratum from the EXP-001 reproductions (AMENDMENT-1).
"""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "EXP-102-a2-campaign"))

from drift_map import ALPHA, SEED, run_cell  # noqa: E402

LA_KEY = Path.home() / "data/corpora/anti-spoofing/keys/LA/CM/trial_metadata.txt"
SCORES = Path.home() / "data/corpora/anti-spoofing/official-scores"

# Variants frozen in AMENDMENT-1 before any cell was computed.
RELEASED = {
    "xlsr_sls": SCORES / "xlsr-sls/scores_LA.txt",
    "xlsr_mamba": SCORES / "xlsr-mamba/Bmamba5_LA_WCE_1e-06_ES144_NE12.txt",
    "xlsr_conformer": SCORES / "xlsr-conformer-rosello/Scores_Best_LA_Fixed_size_train.txt",
}
# Locally verified 21LA EERs (eval-phase subset) recorded in
# ~/data/corpora/anti-spoofing/official-scores/README.md on 2026-08-14.
PUBLISHED_EER = {"xlsr_sls": 2.868, "xlsr_mamba": 0.931, "xlsr_conformer": 1.378}
EER_TOLERANCE = 0.5


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def input_hashes():
    """Bind the result to the exact bytes it consumed (experiment-protocol.md)."""
    files = {"key": LA_KEY, "readme": SCORES / "README.md",
             "code": Path(__file__).resolve(),
             # drift_map.py supplies run_cell, i.e. the actual cell computation.
             "drift_map": REPO / "EXP-102-a2-campaign" / "drift_map.py",
             "prereg": Path(__file__).parent / "PREREG.md",
             "amendment": Path(__file__).parent / "AMENDMENT-1-detector-provenance.md"}
    files.update({f"scores_{k}": v for k, v in RELEASED.items()})
    out = {k: sha256(v) for k, v in files.items()}
    try:
        out["git_commit"] = subprocess.run(
            ["git", "-C", str(Path(__file__).parent), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        out["git_commit"] = "unavailable"
    return out


def key_table(eval_only, include_hidden=False):
    """utt_id -> (condition, is_bonafide).

    The sanity EER uses the eval-phase subset, matching the verified figures in the
    official-scores README. The transport cells use the eval and progress phases,
    matching EXP-102's 108-cell map after AMENDMENT-3 of that experiment: the
    hidden phase is VAD-trimmed audio and is excluded (see
    ../EXP-001-scoring-campaign/code/protocol.py), so the two provenance strata
    stay comparable. `include_hidden=True` reproduces the superseded all-phase map.
    """
    out = {}
    for line in LA_KEY.read_text().splitlines():
        p = line.split()
        if eval_only and p[7] != "eval":
            continue
        if not include_hidden and p[7] == "hidden":
            continue
        out[p[1]] = (p[2], p[5] == "bonafide")
    return out


def released_slices(path, keys):
    """Condition -> (bona scores, spoof scores) for one released score file."""
    slices, missing = {}, 0
    for line in path.read_text().splitlines():
        p = line.split()
        if len(p) < 2:
            continue
        meta = keys.get(p[0])
        if meta is None:
            missing += 1
            continue
        cond, bona = meta
        slices.setdefault(cond, ([], []))[0 if bona else 1].append(float(p[1]))
    return {c: (np.array(b), np.array(s)) for c, (b, s) in slices.items()}, missing


def eer(bona, spoof):
    scores = np.concatenate([bona, spoof])
    labels = np.concatenate([np.ones(len(bona)), np.zeros(len(spoof))])
    order = np.argsort(scores, kind="mergesort")
    labels = labels[order]
    # Higher score = bona fide, so sweeping the threshold upward rejects bona fide first.
    frr = np.cumsum(labels) / len(bona)
    far = 1.0 - np.cumsum(1 - labels) / len(spoof)
    i = np.nanargmin(np.abs(frr - far))
    return float((frr[i] + far[i]) / 2 * 100)


def main():
    keys_eval = key_table(eval_only=True)
    keys_all = key_table(eval_only=False)
    results, sanity = {}, {}
    for name, path in RELEASED.items():
        eval_slices, _ = released_slices(path, keys_eval)
        slices, missing = released_slices(path, keys_all)
        pooled_b = np.concatenate([b for b, _ in eval_slices.values()])
        pooled_s = np.concatenate([s for _, s in eval_slices.values()])
        observed = eer(pooled_b, pooled_s)
        delta = abs(observed - PUBLISHED_EER[name])
        sanity[name] = {
            "eer_observed": round(observed, 4),
            "eer_published": PUBLISHED_EER[name],
            "delta": round(delta, 4),
            "within_tolerance": bool(delta <= EER_TOLERANCE),
            "unmatched_ids": missing,
            "conditions": {c: [len(b), len(s)] for c, (b, s) in sorted(slices.items())},
        }
        if delta > EER_TOLERANCE:
            # PREREG kill criterion: drop before any transport cell is computed.
            print(f"DROP {name}: 21LA EER {observed:.3f} vs published "
                  f"{PUBLISHED_EER[name]} (delta {delta:.3f} > {EER_TOLERANCE})")
            continue
        rng = np.random.default_rng(SEED)
        cells = {}
        for cal in sorted(slices):
            for dep in sorted(slices):
                if cal == dep:
                    continue
                cell = run_cell(rng, slices[cal], slices[dep])
                if cell is not None:
                    cells[f"{cal}->{dep}"] = cell
        results[name] = cells
        print(f"{name}: EER {observed:.3f} (published {PUBLISHED_EER[name]}), "
              f"{len(cells)} cells")

    out = Path(__file__).parent / "results_cellA.json"
    out.write_text(json.dumps(
        {"alpha": ALPHA, "seed": SEED, "provenance": "released-scores",
         "phases": "eval+progress (hidden excluded; EXP-102 AMENDMENT-3)",
         "input_hashes": input_hashes(),
         "sanity": sanity, "cells": results}, indent=1))
    print("wrote", out)


def self_check():
    """A perfectly separated pair must give EER 0; identical distributions ~50."""
    assert eer(np.array([3.0, 4.0, 5.0]), np.array([0.0, 1.0, 2.0])) == 0.0
    same = np.linspace(0, 1, 400)
    assert 40.0 < eer(same, same.copy()) < 60.0
    assert len(key_table(eval_only=False, include_hidden=True)) == 181566
    assert len(key_table(eval_only=False)) == 164640
    keys = key_table(eval_only=True)
    assert len(keys) == 148176, len(keys)
    conds = {c for c, _ in keys.values()}
    assert conds == {"none", "alaw", "ulaw", "gsm", "g722", "opus", "pstn"}, conds
    print("self-check OK")


if __name__ == "__main__":
    if "--self-check" in sys.argv:
        self_check()
    else:
        main()
