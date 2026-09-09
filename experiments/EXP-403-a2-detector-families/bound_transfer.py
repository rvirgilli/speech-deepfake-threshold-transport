"""Does a bona-fide-only bound on spoof-side cost transfer across detectors?

A2's stated limitation is that the observable ranks cells by cost but cannot
give a rate. If a conservative bound transferred, that limitation would become
an instrument and the paper would have a method. This tests it.

At n=2 detectors the answer is "no, and we cannot tell why" -- see the table in
paper/A2/STRONG-ACCEPT.md. The interesting version needs >=4 families so the
test can be leave-one-out, which is what EXP-403 exists to enable. This script
is written to take however many detectors are present, so it answers the n=2
question now and the n=4 question when the scores land.
"""

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

DRIFT = (Path(__file__).parent.parent / "EXP-102-a2-campaign/results_drift.json")
COVERAGE_BAR = 0.95   # a bound that covers less than this is not a bound


def cells_for(cells, det):
    ks = [k for k in cells if k.startswith(det + "/") and not cells[k]["resolution_limited"]]
    return (np.array([cells[k]["log2_fpr_ratio"] for k in ks]),
            np.array([cells[k]["fnr_price"] for k in ks]))


def envelope(x_tr, y_tr, x_te):
    """Monotone conservative bound: at observable x, the worst price seen among
    training cells at least as extreme. Price falls as the observable rises, so
    the envelope is a reverse cumulative max over x-sorted training cells."""
    o = np.argsort(x_tr)
    xs, ys = x_tr[o], y_tr[o]
    env = np.maximum.accumulate(ys[::-1])[::-1]
    return np.interp(x_te, xs, env)


def main():
    cells = {k: v for sec in ("within", "cross")
             for k, v in json.load(open(DRIFT))[sec].items()}
    dets = sorted({k.split("/")[0] for k in cells})
    print(f"detectors present: {dets}\n")

    data = {d: cells_for(cells, d) for d in dets}
    for d, (x, y) in data.items():
        print(f"  {d:8s} n={len(x):3d} cells  price [{y.min():+.3f}, {y.max():+.3f}]")

    print("\npairwise transfer (fit on A, test on B):")
    rows = {}
    for a, b in combinations(dets, 2):
        for tr, te in ((a, b), (b, a)):
            xt, yt = data[tr]
            xe, ye = data[te]
            cov = float((ye <= envelope(xt, yt, xe) + 1e-9).mean())
            rows[f"{tr}->{te}"] = cov
            print(f"  {tr:8s} -> {te:8s}  covers {cov*100:5.1f}%  "
                  f"{'OK' if cov >= COVERAGE_BAR else 'NOT A BOUND'}")

    if len(dets) >= 4:
        print("\nleave-one-out (fit on all others, test on held-out) -- the test that")
        print("distinguishes an idiosyncratic pair difference from a per-detector gain:")
        for held in dets:
            others = [d for d in dets if d != held]
            xt = np.concatenate([data[d][0] for d in others])
            yt = np.concatenate([data[d][1] for d in others])
            xe, ye = data[held]
            cov = float((ye <= envelope(xt, yt, xe) + 1e-9).mean())
            rows[f"LOO:{held}"] = cov
            print(f"  hold out {held:8s}  covers {cov*100:5.1f}%  "
                  f"{'BOUND TRANSFERS' if cov >= COVERAGE_BAR else 'no transfer'}")
    else:
        print(f"\nonly {len(dets)} detectors: leave-one-out needs >=4, so this cannot")
        print("distinguish an idiosyncratic pair difference from a per-detector scale")
        print("factor. That is EXP-403's whole purpose -- see paper/A2/STRONG-ACCEPT.md.")

    out = Path(__file__).parent / "artifacts"
    out.mkdir(exist_ok=True)
    (out / "bound_transfer.json").write_text(json.dumps(
        {"detectors": dets, "coverage": rows, "coverage_bar": COVERAGE_BAR}, indent=1))
    print(f"\nwrote {out/'bound_transfer.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
