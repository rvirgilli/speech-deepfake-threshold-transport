"""Does calibration fragility dissociate from discrimination robustness?

main.tex asserts "bona-fide score stability, not the policy, is what fails".
That sentence is load-bearing or deletable. It is testable: practitioners
monitor discrimination (EER/AUC, threshold-free). If a shift can leave
discrimination intact while destroying the transported operating point, the
failure is invisible to what people actually watch.

Per cell (cal -> dep) we compare:
  severity   |log2(FPR_realized / alpha)|  -- the operating point, threshold-bound
  dAUC       AUC(dep) - AUC(cal)           -- discrimination, threshold-free
Both are computed from the same score arrays that produced the drift map.

Guard against the folklore reading: the claim of interest is a dissociation
between two robustness properties WITHIN a detector, not a ranking of the two
detectors. Everything is reported per detector.

Guard required by docs/verification.md: a correlation must be checked for being
structurally forced, and a null for being structurally impossible; both need a
positive control. Included below.
"""

import json
from pathlib import Path

import numpy as np

from drift_map import ALPHA, MODELS, N_CAL, SEED, cross_slices, la_channel_slices

HERE = Path(__file__).parent
SEV_BAR = 1.0
DAUC_STABLE = 0.02   # |dAUC| below this = discrimination essentially unchanged
B = 400


def auc(bona, spoof):
    """P(bona scores above spoof); rank-based, threshold-free."""
    x = np.concatenate([bona, spoof])
    r = np.argsort(np.argsort(x)).astype(float) + 1
    return float((r[:len(bona)].sum() - len(bona) * (len(bona) + 1) / 2)
                 / (len(bona) * len(spoof)))


def spearman(x, y):
    if len(x) < 4:
        return np.nan
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    drift = json.load(open(HERE / "results_drift.json"))
    rng = np.random.default_rng(SEED)
    out = {}

    for model in MODELS:
        slices = dict(la_channel_slices(model))
        cs = cross_slices(model)
        for k, v in cs.items():
            slices[k] = v

        # Discrimination is a property of a condition, not of a pair.
        cond_auc, cond_med = {}, {}
        for c, (b, s) in slices.items():
            if len(b) < 50 or len(s) < 50:
                continue
            cond_auc[c] = auc(b, s)
            cond_med[c] = float(np.median(b))

        rows = []
        for sec in ("within", "cross"):
            for key, cell in drift[sec].items():
                det, pair = key.split("/")
                if det != model:
                    continue
                src, dst = pair.split("->")
                if src not in cond_auc or dst not in cond_auc:
                    continue
                rows.append({
                    "cell": key, "sec": sec, "src": src, "dst": dst,
                    "sev": abs(cell["log2_fpr_ratio"]),
                    "dauc": cond_auc[dst] - cond_auc[src],
                    "auc_cal": cond_auc[src], "auc_dep": cond_auc[dst],
                    # score-location shift, the quantity the sentence blames
                    "dmed": cond_med[dst] - cond_med[src],
                })

        sev = np.array([r["sev"] for r in rows])
        dauc = np.array([r["dauc"] for r in rows])
        dmed = np.array([r["dmed"] for r in rows])
        stable = np.abs(dauc) < DAUC_STABLE
        blown = sev > SEV_BAR

        print(f"\n=== {model}: {len(rows)} cells ===")
        print(f"  AUC per condition: " +
              ", ".join(f"{c}={a:.3f}" for c, a in sorted(cond_auc.items())))
        print(f"  discrimination stable (|dAUC|<{DAUC_STABLE}): {stable.sum()}/{len(rows)} cells")
        print(f"  of those, operating point blown (|log2|>1): "
              f"**{(stable & blown).sum()}/{stable.sum()}**")
        print(f"  of the discrimination-unstable cells, blown: "
              f"{(~stable & blown).sum()}/{(~stable).sum()}")
        print(f"  Spearman(|dAUC|, severity)  {spearman(np.abs(dauc), sev):+.3f}   "
              f"<- if ~0, the two robustness properties are unrelated")
        print(f"  POSITIVE CONTROL Spearman(|dmedian shift|, severity) "
              f"{spearman(np.abs(dmed), sev):+.3f}   <- score LOCATION should drive it")

        # Interval on the headline count, delete-one-condition jackknife.
        conds = sorted({r["src"] for r in rows} | {r["dst"] for r in rows})
        jack = []
        for c in conds:
            sub = [r for r in rows if c not in (r["src"], r["dst"])]
            if not sub:
                continue
            st = np.array([abs(r["dauc"]) < DAUC_STABLE for r in sub])
            bl = np.array([r["sev"] > SEV_BAR for r in sub])
            if st.sum():
                jack.append((st & bl).sum() / st.sum())
        jack = np.array(jack)
        se = float(np.sqrt((len(jack) - 1) / len(jack) * np.sum((jack - jack.mean()) ** 2)))
        frac = float((stable & blown).sum() / max(stable.sum(), 1))
        print(f"  fraction blown among stable = {frac:.3f}, "
              f"delete-one-condition jackknife SE {se:.3f} -> "
              f"[{max(0, frac - 1.96 * se):.3f}, {min(1, frac + 1.96 * se):.3f}] "
              f"(unit: condition, n={len(conds)})")

        worst = sorted([r for r in rows if abs(r["dauc"]) < DAUC_STABLE],
                       key=lambda r: -r["sev"])[:5]
        print("  worst dissociating cells (discrimination intact, operating point gone):")
        for r in worst:
            print(f"    {r['cell']:42s} dAUC {r['dauc']:+.4f} "
                  f"(AUC {r['auc_cal']:.3f}->{r['auc_dep']:.3f})  |log2| {r['sev']:.2f}")

        out[model] = {
            "n_cells": len(rows), "cond_auc": cond_auc,
            "n_stable": int(stable.sum()), "n_stable_and_blown": int((stable & blown).sum()),
            "frac_blown_among_stable": frac, "jackknife_se": se,
            "spearman_absdauc_sev": spearman(np.abs(dauc), sev),
            "spearman_absdmed_sev": spearman(np.abs(dmed), sev),
            "worst": [{k: r[k] for k in ("cell", "dauc", "auc_cal", "auc_dep", "sev")}
                      for r in worst],
        }

    (HERE / "results_dissociation.json").write_text(json.dumps(out, indent=1))
    print(f"\nwrote {HERE/'results_dissociation.json'}")


if __name__ == "__main__":
    main()
