"""Descriptive reconstruction of the held-out 21LA -> ASVspoof 5 cost map.

The former report treated sqrt(held-out R^2) as an iid Pearson correlation at
n=12 and attached a Fisher interval/p-value.  That inference is invalid for a
fixed line fitted on one crossed directed grid and evaluated on another.  This
script intentionally computes point and delete-one-condition sensitivities
only.  It makes no inferential claim.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


HERE = Path(__file__).resolve().parent
DRIFT_PATH = HERE.parent / "EXP-102-a2-campaign" / "results_drift.json"
A5_PATH = HERE / "artifacts" / "results_a5.json"
OUT = HERE / "artifacts" / "cost_map_sensitivity.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def line_fit(rows: list[tuple[float, float]]) -> tuple[float, float]:
    x = np.asarray([row[0] for row in rows], dtype=float)
    y = np.asarray([row[1] for row in rows], dtype=float)
    design = np.column_stack((x, np.ones_like(x)))
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    return float(slope), float(intercept)


def heldout_r2(rows: list[tuple[float, float]], slope: float, intercept: float) -> float:
    x = np.asarray([row[0] for row in rows], dtype=float)
    y = np.asarray([row[1] for row in rows], dtype=float)
    prediction = slope * x + intercept
    return float(1 - np.sum((y - prediction) ** 2) / np.sum((y - np.mean(y)) ** 2))


def average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2 + 1
        start = end
    return ranks


def spearman(rows: list[tuple[float, float]]) -> float:
    x = average_ranks(np.asarray([row[0] for row in rows], dtype=float))
    y = average_ranks(np.asarray([row[1] for row in rows], dtype=float))
    return float(np.corrcoef(x, y)[0, 1])


def pair_conditions(key: str) -> tuple[str, str]:
    pair = key.split("/", 1)[-1]
    return tuple(pair.split("->", 1))


def main() -> None:
    drift = json.loads(DRIFT_PATH.read_text())
    a5 = json.loads(A5_PATH.read_text())

    train = []
    train_nonlimited = []
    train_by_key = {}
    for key, value in drift["within"].items():
        if not key.startswith("ssl/"):
            continue
        row = (float(value["log2_fpr_ratio"]), float(value["fnr_price"]))
        train.append(row)
        train_by_key[key] = row
        if not value["resolution_limited"]:
            train_nonlimited.append(row)

    test = []
    test_by_key = {}
    for key, value in a5["ssl/twin_free"].items():
        if value["oracle_overlap_dominated"]:
            continue
        row = (float(value["log2_fpr_ratio"]), float(value["fnr_price"]))
        test.append(row)
        test_by_key[key] = row

    slope, intercept = line_fit(train)
    slope_nl, intercept_nl = line_fit(train_nonlimited)

    a5_conditions = sorted({condition for key in test_by_key for condition in pair_conditions(key)})
    delete_a5 = {}
    for condition in a5_conditions:
        retained = [
            row for key, row in test_by_key.items() if condition not in pair_conditions(key)
        ]
        delete_a5[condition] = heldout_r2(retained, slope, intercept)

    la_conditions = sorted({condition for key in train_by_key for condition in pair_conditions(key)})
    delete_train_nonlimited = {}
    for condition in la_conditions:
        retained = [
            row
            for key, row in train_by_key.items()
            if condition not in pair_conditions(key)
            and not drift["within"][key]["resolution_limited"]
        ]
        loo_slope, loo_intercept = line_fit(retained)
        delete_train_nonlimited[condition] = heldout_r2(test, loo_slope, loo_intercept)

    result = {
        "status": "descriptive_sensitivity_no_interval_or_p_value",
        "inputs": {
            "drift": {
                "path": "experiments/EXP-102-a2-campaign/results_drift.json",
                "sha256": sha256(DRIFT_PATH),
            },
            "a5": {
                "path": "experiments/EXP-103-a5-replicate/artifacts/results_a5.json",
                "sha256": sha256(A5_PATH),
            },
        },
        "primary": {
            "training_cells_21la": len(train),
            "heldout_viable_cells_a5": len(test),
            "slope": slope,
            "intercept": intercept,
            "heldout_r2": heldout_r2(test, slope, intercept),
            "heldout_spearman": spearman(test),
        },
        "exclude_resolution_limited_training": {
            "training_cells_21la": len(train_nonlimited),
            "slope": slope_nl,
            "intercept": intercept_nl,
            "heldout_r2": heldout_r2(test, slope_nl, intercept_nl),
        },
        "delete_one_a5_condition_fixed_primary_line": {
            "values": delete_a5,
            "range": [min(delete_a5.values()), max(delete_a5.values())],
        },
        "delete_one_21la_condition_refit_nonlimited": {
            "values": delete_train_nonlimited,
            "range": [min(delete_train_nonlimited.values()), max(delete_train_nonlimited.values())],
        },
        "inference": None,
        "reason": "directed cells reuse crossed conditions; Fisher-transforming sqrt(held-out R^2) is invalid",
    }
    OUT.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result["primary"], indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
