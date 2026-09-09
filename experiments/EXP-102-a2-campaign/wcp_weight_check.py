"""Does the score-weighting heuristic run through weight degeneracy?

Leong (arXiv 2606.11949) reports that weighted CP fails on decoder classifiers
because logistic density-ratio estimation in 3584-4096 dimensions separates the
two samples perfectly and clips every importance weight to zero; projecting to
<=32 dimensions restores coverage. If our failure has the same signature it is
an implementation artifact with a known fix, not a result about audio.

Our estimator is structurally different -- a 1-D histogram ratio on the detector
score, floored at 0.1 -- but "structurally different" is an argument, not a
measurement. This recomputes the weights actually used by drift_map.run_cell and
reports, per cell: the fraction pinned at each clip boundary, the weight range,
and the effective sample size n_eff = (sum w)^2 / sum w^2 as a fraction of N.

Degeneracy signature (any of): mass at the lower clip, or n_eff/N below ~0.1.
"""

import numpy as np

from drift_map import (ALPHA, CROSS, MODELS, N_CAL, SEED, cross_slices,
                       density_ratio_weights, la_channel_slices)

DEGENERATE_NEFF = 0.10
B = 50


def probe(rng, cal, dep):
    cal_bona, cal_spoof = cal
    dep_bona, dep_spoof = dep
    if len(cal_bona) < N_CAL + 100:
        return None
    cal_mix = np.concatenate([cal_bona, cal_spoof])
    dep_mix = np.concatenate([dep_bona, dep_spoof])
    lo, hi, neff, wmin, wmax = [], [], [], [], []
    for _ in range(B):
        cohort = cal_bona[rng.choice(len(cal_bona), N_CAL, replace=False)]
        w = density_ratio_weights(cohort, cal_mix, dep_mix)
        lo.append(float(np.mean(w <= 0.1 + 1e-12)))
        hi.append(float(np.mean(w >= 10.0 - 1e-12)))
        neff.append(n_eff_frac(w))
        wmin.append(float(w.min()))
        wmax.append(float(w.max()))
    return {"frac_at_lower_clip": float(np.mean(lo)),
            "frac_at_upper_clip": float(np.mean(hi)),
            "n_eff_frac": float(np.mean(neff)),
            "w_min": float(np.min(wmin)), "w_max": float(np.max(wmax))}


def n_eff_frac(w):
    """Kish effective sample size as a fraction of N. Scale-invariant, so it
    measures how unevenly the weight is spread, not how large the weights are."""
    return float(w.sum() ** 2 / np.sum(w ** 2)) / len(w)


def self_check():
    """The probe must call each failure mode when it is there and not otherwise.

    The two modes are distinct and n_eff sees only the second. A weighted
    quantile is invariant to the scale of w, so weights all pinned at one clip
    boundary are informationless but leave n_eff at N: Leong's clip-to-zero
    regime, floored, is not a collapse but a silent reversion to the unweighted quantile.
    """
    rng = np.random.default_rng(0)
    n = 4000
    # Mode 1, inert: calibration and deployment mixtures share no support, so
    # the histogram ratio is 0 everywhere the cohort lives and every weight
    # sits on the floor. n_eff is blind to this by construction.
    sep = probe(rng, (rng.normal(0, 1, n), rng.normal(0, 1, n)),
                (rng.normal(60, 1, n), rng.normal(60, 1, n)))
    assert sep["frac_at_lower_clip"] > 0.9, sep
    assert sep["n_eff_frac"] > 0.9, sep
    # Mode 2, concentrated: a few points carry the whole weight. Tested on the
    # weight vector directly, because the [0.1, 10] clip bounds the ratio at
    # 100:1 and no score distribution can drive n_eff much below 0.07 through
    # the histogram -- which is itself why this mode is not reachable here.
    assert n_eff_frac(np.r_[np.full(499, 0.1), 10.0]) < DEGENERATE_NEFF
    assert n_eff_frac(np.full(500, 0.1)) > 0.99
    # Benign: identical distributions, every weight near 1.
    same = probe(rng, (rng.normal(0, 1, n), rng.normal(2, 1, n)),
                 (rng.normal(0, 1, n), rng.normal(2, 1, n)))
    assert same["frac_at_lower_clip"] < 0.02, same
    assert same["n_eff_frac"] > 0.8, same
    print("self-check ok: probe separates inert weights, concentrated weights "
          "and a benign ratio")


def main():
    self_check()
    worst = None
    n_cells = 0
    for model in MODELS:
        rng = np.random.default_rng(SEED)
        la = la_channel_slices(model)
        cs = cross_slices(model)
        pairs = ([(f"{a}->{b}", la[a], la[b]) for a in sorted(la) for b in sorted(la) if a != b]
                 + [(f"{a}->{b}", cs[a], cs[b]) for a in CROSS for b in CROSS if a != b])
        for name, cal, dep in pairs:
            r = probe(rng, cal, dep)
            if r is None:
                continue
            n_cells += 1
            if worst is None or r["n_eff_frac"] < worst[1]["n_eff_frac"]:
                worst = (f"{model}/{name}", r)
            if r["frac_at_lower_clip"] > 0.01 or r["n_eff_frac"] < DEGENERATE_NEFF:
                print(f"  FLAG {model}/{name}: {r}")

    print(f"\n{n_cells} cells probed at alpha={ALPHA}, N={N_CAL}")
    k, r = worst
    print(f"lowest effective sample size: {k}  n_eff/N = {r['n_eff_frac']:.3f}, "
          f"weights in [{r['w_min']:.3f}, {r['w_max']:.3f}], "
          f"{r['frac_at_lower_clip']*100:.2f}% at the 0.1 floor")


def explains_failure():
    """Does low effective sample size explain the score-weighting miss?

    Hennhoefer & Preisach (arXiv 2603.23205) give the mechanism: importance
    weights localise, n_eff drops, p-values go conservative. If our heuristic's
    negative runs through that, n_eff should predict the direction and size of
    the miss. If it does not, the negative is about exchangeability, not the
    estimator, and the paper may say so.
    """
    import json
    from scipy.stats import spearmanr

    HERE = __file__.rsplit("/", 1)[0]
    d = json.load(open(f"{HERE}/results_drift.json"))
    cells = {k: v for sec in ("within", "cross") for k, v in d[sec].items()}

    rows = []
    for model in MODELS:
        rng = np.random.default_rng(SEED)
        la = la_channel_slices(model)
        cs = cross_slices(model)
        pairs = ([(f"{a}->{b}", la[a], la[b]) for a in sorted(la) for b in sorted(la) if a != b]
                 + [(f"{a}->{b}", cs[a], cs[b]) for a in CROSS for b in CROSS if a != b])
        for name, cal, dep in pairs:
            key = f"{model}/{name}"
            if key not in cells or "weighted_fpr_mean" not in cells[key]:
                continue
            r = probe(rng, cal, dep)
            if r is None:
                continue
            rows.append((r["n_eff_frac"], r["frac_at_lower_clip"],
                         cells[key]["weighted_fpr_mean"], key))

    neff, floor, wfpr, keys = map(np.array, zip(*rows))
    wfpr = wfpr.astype(float)
    signed = wfpr - ALPHA
    print(f"\n{len(rows)} cells with a score-weighting reading")
    print(f"  weighted FPR range {wfpr.min():.4f}--{wfpr.max():.4f}; "
          f"{int((wfpr < ALPHA).sum())} of {len(wfpr)} conservative")
    for label, x in (("n_eff/N", neff), ("frac at floor", floor)):
        for tgt_name, tgt in (("signed miss", signed), ("|miss|", np.abs(signed))):
            rho, p = spearmanr(x, tgt)
            print(f"  {label:14s} vs {tgt_name:12s} rho={rho:+.3f}  p={p:.3g}")
    i = int(np.argmin(neff))
    print(f"  lowest n_eff cell {keys[i]}: n_eff/N={neff[i]:.3f}, "
          f"weighted FPR {wfpr[i]:.4f}")


if __name__ == "__main__":
    main()
    explains_failure()
