"""Durable reproduction of the 2026-08-15 post-review speaker audit.

This is a post-hoc sensitivity analysis, not a pre-registered experiment.  It
uses only committed scores and ASVspoof 2021 LA metadata.  The sampling and
seeds reproduce POSTREVIEW-speaker-clustering.md:

* speaker-disjoint 90% realized-FPR widths over ten Monte Carlo seeds;
* a group-size-preserving permutation null (seed 23); and
* leave-one-speaker-out sensitivity for the two affected cells.

No model inference is performed and spoof scores are never used.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
SCORES = HERE.parent / "EXP-001-scoring-campaign" / "scores"
KEY = Path.home() / "data/corpora/anti-spoofing/keys/LA/CM/trial_metadata.txt"
OUT = HERE / "artifacts" / "speaker_clustering.json"

ALPHA = 0.05
N = 500
B = 400
PERMUTATIONS = 200
SEED_SWEEP = tuple(range(100, 110))
CONDITIONS = ("none", "pstn", "gsm")
DETECTORS = ("ssl", "aasist")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def metadata() -> dict[str, tuple[str, str, str]]:
    out = {}
    with KEY.open() as f:
        for line in f:
            row = line.split()
            out[row[1]] = (row[0], row[2], row[5])
    return out


def scores(detector: str) -> dict[str, float]:
    path = SCORES / f"{detector}_asv21la.csv.gz"
    with gzip.open(path, "rt") as f:
        return {row["utt_id"]: float(row["score"]) for row in csv.DictReader(f)}


def grouped_bona(
    score: dict[str, float], meta: dict[str, tuple[str, str, str]], condition: str
) -> dict[str, list[float]]:
    by = defaultdict(list)
    for utt, value in score.items():
        if utt not in meta:
            continue
        speaker, codec, label = meta[utt]
        if codec == condition and label == "bonafide":
            by[speaker].append(value)
    return dict(by)


def interval_width(
    by: dict[str, list[float]], speakers: list[str], rng: np.random.Generator
) -> tuple[float, list[int]]:
    """90% width of FPR over speaker-disjoint calibration/evaluation draws."""
    realized = []
    n_speakers = []
    for _ in range(B):
        order = list(rng.permutation(speakers))
        calibration = []
        used = 0
        for speaker in order:
            calibration.extend(by[speaker])
            used += 1
            if len(calibration) >= N:
                break
        evaluation = np.asarray(
            [value for speaker in order[used:] for value in by[speaker]], dtype=float
        )
        calibration = np.sort(np.asarray(calibration, dtype=float))
        k = int(np.floor((len(calibration) + 1) * ALPHA))
        if k < 1 or evaluation.size == 0:
            raise RuntimeError("invalid speaker-disjoint draw")
        realized.append(float(np.mean(evaluation < calibration[k - 1])))
        n_speakers.append(used)
    q05, q95 = np.percentile(realized, [5, 95])
    return float((q95 - q05) * 100), n_speakers


def permutation_test(
    by: dict[str, list[float]], speakers: list[str]
) -> dict[str, float | int]:
    sizes = [len(by[speaker]) for speaker in speakers]
    all_scores = np.asarray([value for speaker in speakers for value in by[speaker]])
    rng = np.random.default_rng(23)
    observed, n_speakers = interval_width(by, speakers, rng)
    null = []
    for _ in range(PERMUTATIONS):
        permuted = rng.permutation(all_scores)
        shuffled = {}
        start = 0
        for speaker, size in zip(speakers, sizes):
            shuffled[speaker] = list(permuted[start : start + size])
            start += size
        width, _ = interval_width(shuffled, speakers, rng)
        null.append(width)
    null_array = np.asarray(null)
    exceedances = int(np.sum(null_array >= observed))
    return {
        "observed_width_pp": observed,
        "null_median_pp": float(np.median(null_array)),
        "null_p95_pp": float(np.percentile(null_array, 95)),
        "null_exceedances": exceedances,
        "permutation_p_plus_one": (exceedances + 1) / (PERMUTATIONS + 1),
        "median_calibration_speakers": int(np.median(n_speakers)),
    }


def seed_sweep(
    by: dict[str, list[float]], speakers: list[str]
) -> dict[str, float | list[float]]:
    values = [
        interval_width(by, speakers, np.random.default_rng(seed))[0]
        for seed in SEED_SWEEP
    ]
    return {
        "widths_pp": values,
        "mean_width_pp": float(np.mean(values)),
        "sd_width_pp": float(np.std(values)),
    }


def leave_one_out(
    by: dict[str, list[float]], speakers: list[str]
) -> dict[str, object]:
    full = np.mean(
        [interval_width(by, speakers, np.random.default_rng(200 + i))[0] for i in range(4)]
    )
    drops = []
    for dropped in speakers:
        reduced = {speaker: values for speaker, values in by.items() if speaker != dropped}
        reduced_speakers = [speaker for speaker in speakers if speaker != dropped]
        width = np.mean(
            [
                interval_width(reduced, reduced_speakers, np.random.default_rng(300 + i))[0]
                for i in range(2)
            ]
        )
        drops.append({"speaker": dropped, "width_pp": float(width)})
    drops.sort(key=lambda row: row["width_pp"])
    return {
        "full_pool_width_pp": float(full),
        "minimum": drops[0],
        "median_width_pp": float(np.median([row["width_pp"] for row in drops])),
        "maximum": drops[-1],
        "all_drops": drops,
    }


def main() -> None:
    meta = metadata()
    result = {
        "status": "post_hoc_sensitivity_not_preregistered",
        "estimand": "90% realized-FPR interval width under speaker-disjoint same-condition calibration",
        "alpha": ALPHA,
        "nominal_N_utterances": N,
        "draws_per_width": B,
        "permutations": PERMUTATIONS,
        "seed_sweep": list(SEED_SWEEP),
        "inputs": {
            "metadata": {
                "path": "~/data/corpora/anti-spoofing/keys/LA/CM/trial_metadata.txt",
                "sha256": sha256(KEY),
            },
            **{
                detector: {
                    "path": (
                        "experiments/EXP-001-scoring-campaign/scores/"
                        f"{detector}_asv21la.csv.gz"
                    ),
                    "sha256": sha256(SCORES / f"{detector}_asv21la.csv.gz"),
                }
                for detector in DETECTORS
            },
        },
        "cells": {},
    }
    for detector in DETECTORS:
        score = scores(detector)
        for condition in CONDITIONS:
            by = grouped_bona(score, meta, condition)
            speakers = sorted(by)
            cell = f"{detector}/{condition}"
            print(f"{cell}: {sum(map(len, by.values()))} utterances, {len(speakers)} speakers")
            result["cells"][cell] = {
                "n_utterances": sum(map(len, by.values())),
                "n_speakers": len(speakers),
                "seed_sweep": seed_sweep(by, speakers),
                "permutation": permutation_test(by, speakers),
            }
            if detector == "aasist" and condition in ("pstn", "gsm"):
                result["cells"][cell]["leave_one_speaker_out"] = leave_one_out(by, speakers)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
