"""Interval for the price-vs-W1 null, over the unit that was actually sampled.

The claim under test: the spoof-side price of a miscalibrated threshold is
uncorrelated with the bona-fide-side distance (oracle W1) that a deployment can
observe. If true, no refinement of a bona-fide-side audit recovers the cost.

A null with no interval cannot carry a paper, and the cell count is not n. Each
cell is a (from-condition -> to-condition) pair, so cells sharing a condition
share their calibration or deployment slice: the design is dyadic over
conditions, 7 channel/transmission conditions within-corpus and 4 corpora
cross-corpus, and
the two margins are indexed by the SAME units. That is the self-paired case in
which this project's pigeonhole variance correction was found to be wrong, so
nothing clever is attempted here. Three readings are reported:

  cell    - iid resample of cells. WRONG here, reported only to show the width
            the dependence costs.
  cluster - resample the condition set with replacement, keep the cells whose
            both endpoints are drawn. Honest about the dyadic structure; its
            weakness is that with 7 conditions the resampled sets are coarse.
  jack    - delete-one-condition jackknife (drop every cell touching it), the
            analogue of the two-way jackknife this project's audit certified.

Detectors are not resampled: n=2. Each is reported separately and pooled.
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
N_BOOT = 4000
SEED = 51


def spearman(x, y):
    if len(x) < 4:
        return np.nan
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def load():
    d = json.load(open(HERE / "results_drift.json"))
    rows = []
    for section in ("within", "cross"):
        for k, v in d[section].items():
            det, pair = k.split("/")
            src, dst = pair.split("->")
            rows.append({"det": det, "sec": section, "src": src, "dst": dst,
                         "w1": v["w1_bona_oracle"], "price": v["fnr_price"],
                         "sev": abs(v["log2_fpr_ratio"])})
    return rows


def ci(vals, lo=2.5, hi=97.5):
    vals = np.asarray([v for v in vals if np.isfinite(v)])
    return float(np.percentile(vals, lo)), float(np.percentile(vals, hi)), len(vals)


def analyse(rows, label, out):
    x = np.array([r["w1"] for r in rows])
    y = np.array([r["price"] for r in rows])
    point = spearman(x, y)
    conds = {s: sorted({r[k] for r in rows if r["sec"] == s for k in ("src", "dst")})
             for s in {r["sec"] for r in rows}}
    rng = np.random.default_rng(SEED)

    # (1) iid over cells -- known to be wrong here, kept as the contrast
    b_cell = [spearman(x[i], y[i]) for i in
              (rng.integers(0, len(rows), len(rows)) for _ in range(N_BOOT))]

    # (2) cluster over conditions, per stratum, both margins from one draw
    b_clu = []
    for _ in range(N_BOOT):
        keep = []
        for sec, cs in conds.items():
            drawn = list(rng.choice(cs, len(cs), replace=True))
            mult = {c: drawn.count(c) for c in set(drawn)}
            for r in rows:
                if r["sec"] != sec:
                    continue
                m = mult.get(r["src"], 0) * mult.get(r["dst"], 0)
                keep.extend([r] * m)
        if len(keep) >= 4:
            b_clu.append(spearman(np.array([r["w1"] for r in keep]),
                                  np.array([r["price"] for r in keep])))

    # (3) delete-one-condition jackknife
    jack = []
    for sec, cs in conds.items():
        for c in cs:
            sub = [r for r in rows if not (r["sec"] == sec and c in (r["src"], r["dst"]))]
            v = spearman(np.array([r["w1"] for r in sub]),
                         np.array([r["price"] for r in sub]))
            if np.isfinite(v):
                jack.append(v)
    jack = np.array(jack)
    n_j = len(jack)
    jack_se = float(np.sqrt((n_j - 1) / n_j * np.sum((jack - jack.mean()) ** 2))) if n_j > 1 else np.nan

    c_cell, c_clu = ci(b_cell), ci(b_clu)
    print(f"\n=== {label}  (cells={len(rows)}, conditions={ {s: len(c) for s, c in conds.items()} }) ===")
    print(f"  Spearman(price, W1) point           {point:+.3f}")
    print(f"  iid-over-cells CI95   [{c_cell[0]:+.3f}, {c_cell[1]:+.3f}]   <- WRONG unit, shown for contrast")
    print(f"  condition-cluster CI95 [{c_clu[0]:+.3f}, {c_clu[1]:+.3f}]   ({c_clu[2]} usable replicates)")
    print(f"  delete-one-condition jackknife SE {jack_se:.3f} -> "
          f"[{point - 1.96 * jack_se:+.3f}, {point + 1.96 * jack_se:+.3f}]  "
          f"(range over leave-one-out: [{jack.min():+.3f}, {jack.max():+.3f}])")
    out[label] = {"n_cells": len(rows), "point": point,
                  "ci_iid_cells": [c_cell[0], c_cell[1]],
                  "ci_condition_cluster": [c_clu[0], c_clu[1]],
                  "jackknife_se": jack_se,
                  "ci_jackknife": [point - 1.96 * jack_se, point + 1.96 * jack_se],
                  "jack_min": float(jack.min()), "jack_max": float(jack.max())}


def main():
    rows = load()
    out = {}
    for det in ("ssl", "aasist"):
        analyse([r for r in rows if r["det"] == det], f"{det} (all cells)", out)
    analyse(rows, "pooled (both detectors)", out)
    for det in ("ssl", "aasist"):
        analyse([r for r in rows if r["det"] == det and r["sec"] == "within"],
                f"{det} within-corpus only", out)

    # Reference: does the same design detect a correlation that IS there?
    print("\n=== positive control: severity vs W1, same units ===")
    for det in ("ssl", "aasist"):
        sub = [r for r in rows if r["det"] == det]
        print(f"  {det}: Spearman(|log2 FPR/alpha|, W1) = "
              f"{spearman(np.array([r['w1'] for r in sub]), np.array([r['sev'] for r in sub])):+.3f}")

    (HERE / "results_price_w1.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {HERE/'results_price_w1.json'}")


if __name__ == "__main__":
    main()
