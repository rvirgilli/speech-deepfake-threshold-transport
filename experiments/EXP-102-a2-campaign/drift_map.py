"""EXP-102 cells 5-9: drift map, score weighting, W1 curve, unlabeled monitors.

Drift cell = (system, calibration slice, deployment slice). Within-corpus axis:
21LA channel conditions (six real transmissions plus the untransmitted source;
7x6 ordered pairs). Cross-corpus axis: {21LA-none,
21DF-100k, ITW, BRSpeech} ordered pairs. Per cell:
  - vanilla conformal quantile (N=500 bona from calibration slice, B draws):
    realized FPR on deployment bona, its excursion from the 5% target, the
    realized FNR on deployment spoof at that same threshold, and the two-sided
    log2 ratio to target. FPR alone cannot separate "on target" from "threshold
    collapsed": |FPR - 0.05| is capped at 0.05 on the conservative side, so a
    cell at FPR~0 (which buys that FPR by ceasing to flag spoof at all) scores
    better than one 2x too liberal. Severity is therefore log2(FPR/alpha) and
    the operational price is fnr_price;
  - score-weighting heuristic: calibration weights = a clipped 15-bin density
    ratio of deployment-mixture to calibration-mixture detector scores. This is
    not Tibshirani et al.'s covariate-shift conformal construction and carries
    no test-point weight;
  - outcome-adaptive conformal ceiling with oracle bona-fide labels. It is not
    label-free because online FPR feedback requires those labels;
  - oracle W1 between calibration/deployment bona score distributions (cell 8);
  - unlabeled monitors (cell 9): (a) W1 between deployment and calibration
    MIXTURE score distributions; (b) mean binary entropy of sigmoid of scores
    standardized by calibration-mixture stats. Prevalence confound probed by
    rescaling deployment spoof share x0.5 / x1.5.
"""

import csv
import gzip
import json
from pathlib import Path

import numpy as np

EXP001 = Path.home() / "projects/academic/icassp2027/experiments/EXP-001-scoring-campaign/scores"
import sys
sys.path.insert(0, str(EXP001.parent / "code"))
from protocol import drop_hidden  # noqa: E402

LA_KEY = Path.home() / "data/corpora/anti-spoofing/keys/LA/CM/trial_metadata.txt"
ALPHA = 0.05
N_CAL = 500
B = 1000
B_WEIGHTED = 200
ACI_GAMMA = 0.005
SEVERITY_BAR = 1.0  # |log2(FPR/alpha)|: off target by more than a factor of two
ACI_REPS = 5
SEED = 20260813
MODELS = ["ssl", "aasist"]
CROSS = ["asv21la_nocodec", "asv21df_full", "itw", "brspeech_test"]


def load_scores(model, corpus):
    with gzip.open(EXP001 / f"{model}_{corpus}.csv.gz", "rt") as f:
        rows = drop_hidden(list(csv.DictReader(f)))
    return {r["utt_id"]: (float(r["score"]), r["label"] == "bonafide") for r in rows}


def la_channel_slices(model):
    """21LA scores split by released channel/transmission condition."""
    channel = {}
    for line in LA_KEY.read_text().splitlines():
        p = line.split()
        channel[p[1]] = p[2]
    data = load_scores(model, "asv21la")
    slices = {}
    for utt, (s, bona) in data.items():
        c = channel.get(utt)
        if c is None:
            continue
        slices.setdefault(c, ([], []))[0 if bona else 1].append(s)
    return {c: (np.array(b), np.array(sp)) for c, (b, sp) in slices.items()}


def cross_slices(model):
    la = la_channel_slices(model)
    out = {"asv21la_nocodec": la["none"]}
    for corpus in CROSS[1:]:
        data = load_scores(model, corpus)
        b = np.array([s for s, bona in data.values() if bona])
        sp = np.array([s for s, bona in data.values() if not bona])
        out[corpus] = (b, sp)
    return out


def w1(a, b, grid=2000):
    q = np.linspace(0, 1, grid)
    return float(np.mean(np.abs(np.quantile(a, q) - np.quantile(b, q))))


def density_ratio_weights(cal_scores, cal_mix, dep_mix, bins=15):
    """Clipped mixture-score histogram ratio q_dep/p_cal at cohort scores.

    This one-dimensional heuristic is not the covariate-shift conformal
    construction of Tibshirani et al.: the mixtures contain both classes and
    no test-point weight enters the weighted empirical quantile.
    """
    lo = min(cal_mix.min(), dep_mix.min())
    hi = max(cal_mix.max(), dep_mix.max())
    edges = np.linspace(lo, hi, bins + 1)
    p, _ = np.histogram(cal_mix, edges, density=True)
    q, _ = np.histogram(dep_mix, edges, density=True)
    ratio = np.clip(q / np.maximum(p, 1e-8), 0.1, 10.0)
    idx = np.clip(np.searchsorted(edges, cal_scores) - 1, 0, bins - 1)
    return ratio[idx]


def weighted_quantile(scores, weights, alpha):
    o = np.argsort(scores)
    cw = np.cumsum(weights[o])
    i = int(np.searchsorted(cw / cw[-1], alpha))
    return float(scores[o][min(i, len(scores) - 1)])


def aci_fpr(rng, cohort, dep_bona, dep_spoof):
    """ACI with oracle label feedback on bona items; returns realized FPR/FNR."""
    stream = np.concatenate([np.stack([dep_bona, np.ones(len(dep_bona))], 1),
                             np.stack([dep_spoof, np.zeros(len(dep_spoof))], 1)])
    fprs, fnrs = [], []
    srt = np.sort(cohort)
    for _ in range(ACI_REPS):
        rng.shuffle(stream)
        a = ALPHA
        fb = fs = nb = ns = 0
        for s, is_bona in stream:
            k = int(np.clip(np.floor((len(srt) + 1) * a), 1, len(srt)))
            t = srt[k - 1]
            flagged = s < t
            if is_bona:
                nb += 1
                fb += flagged
                a = float(np.clip(a + ACI_GAMMA * (ALPHA - flagged), 0.001, 0.5))
            else:
                ns += 1
                fs += not flagged
        fprs.append(fb / nb)
        fnrs.append(fs / ns)
    return float(np.mean(fprs)), float(np.mean(fnrs))


def log2_fpr_ratio(fpr, n_dep_bona):
    """Scale-free two-sided miss of the FPR target, in doublings.

    |fpr - ALPHA| cannot exceed ALPHA on the conservative side (fpr >= 0), so a
    detector whose threshold has collapsed to fpr ~ 0 scores as "on target"
    while a 2x-too-liberal cell scores worse; the released pass rule
    |fpr - ALPHA| <= ALPHA is in fact vacuous on that side. The log ratio is
    symmetric under k-fold error in either direction: 0 on target, -1 at half
    the target rate, +1 at twice.

    A zero observed rate is censored, not extrapolated: the value is floored at
    3/n, the one-sided 95% upper bound on a rate that produced no events in n
    trials (rule of three), so the reported severity is the largest miss the
    data can actually support rather than an artifact of how the mean was
    rounded. Cells whose fpr is below 5/n are additionally flagged
    `resolution_limited` -- their severity rests on a handful of utterances and
    must not be printed as a point value.
    """
    return float(np.log2(max(fpr, 3.0 / n_dep_bona) / ALPHA))


def run_cell(rng, cal, dep):
    cal_bona, cal_spoof = cal
    dep_bona, dep_spoof = dep
    if len(cal_bona) < N_CAL + 100:
        return None
    k = int(np.floor((N_CAL + 1) * ALPHA))
    cal_mix = np.concatenate([cal_bona, cal_spoof])
    dep_mix = np.concatenate([dep_bona, dep_spoof])

    v_fpr, v_fnr, c_fnr, w_fpr = [], [], [], []
    for i in range(B):
        cohort = cal_bona[rng.choice(len(cal_bona), N_CAL, replace=False)]
        thr = np.sort(cohort)[k - 1]
        v_fpr.append(float(np.mean(dep_bona < thr)))
        # Same threshold, spoof side: a cell can hold FPR near 0 only by moving
        # the threshold so low that spoof stops being flagged. Without this the
        # conservative failure mode is unobservable (see docstring).
        v_fnr.append(float(np.mean(dep_spoof >= thr)))
        c_fnr.append(float(np.mean(cal_spoof >= thr)))
        if i < B_WEIGHTED:
            wts = density_ratio_weights(cohort, cal_mix, dep_mix)
            w_fpr.append(float(np.mean(dep_bona < weighted_quantile(cohort, wts, ALPHA))))
    cohort = cal_bona[rng.choice(len(cal_bona), N_CAL, replace=False)]
    aci_f, aci_n = aci_fpr(rng, cohort, dep_bona, dep_spoof)

    # Oracle reference: the threshold that realizes alpha ON THE DEPLOYMENT bona
    # fide. The spoof-side cost attributable to miscalibration is the extra FNR
    # the conformal threshold pays over this one. Differencing against the
    # calibration condition's FNR instead would conflate that cost with the
    # intrinsic difficulty gap between calibration and deployment attacks.
    t_oracle = float(np.quantile(dep_bona, ALPHA))
    fnr_oracle = float(np.mean(dep_spoof >= t_oracle))
    price = float(np.mean(v_fnr)) - fnr_oracle

    # Cell 9 monitors (label-free) + prevalence confound.
    mon = {"w1_mixture": w1(dep_mix, cal_mix)}
    mu, sd = cal_mix.mean(), cal_mix.std()
    p = 1.0 / (1.0 + np.exp(-(dep_mix - mu) / sd))
    p = np.clip(p, 1e-6, 1 - 1e-6)
    mon["entropy"] = float(np.mean(-p * np.log(p) - (1 - p) * np.log(1 - p)))
    for f, tag in ((0.5, "half"), (1.5, "x15")):
        n_sp = int(min(len(dep_spoof) * f, len(dep_spoof) * 1.0)) if f <= 1 else len(dep_spoof)
        sp = dep_spoof[rng.choice(len(dep_spoof), int(len(dep_spoof) * min(f, 1.0)), replace=False)]
        if f > 1:  # oversample by concatenating a resample
            extra = dep_spoof[rng.choice(len(dep_spoof), int(len(dep_spoof) * (f - 1)), replace=True)]
            sp = np.concatenate([dep_spoof, extra])
        mix = np.concatenate([dep_bona, sp])
        mon[f"w1_mixture_prev_{tag}"] = w1(mix, cal_mix)

    return {
        "n_cal_bona": len(cal_bona), "n_dep_bona": len(dep_bona), "n_dep_spoof": len(dep_spoof),
        "vanilla_fpr_mean": round(float(np.mean(v_fpr)), 4),
        "vanilla_fpr_ci": [round(float(np.percentile(v_fpr, 2.5)), 4),
                           round(float(np.percentile(v_fpr, 97.5)), 4)],
        "excursion": round(abs(float(np.mean(v_fpr)) - ALPHA), 4),
        "vanilla_fpr_mean_full": float(np.mean(v_fpr)),  # unrounded: severity is
        # not reproducible from the 4-dp value when FPR is O(1e-4)
        "vanilla_fnr_mean": round(float(np.mean(v_fnr)), 4),
        "cal_fnr_at_threshold": round(float(np.mean(c_fnr)), 4),
        # Diagnostic only. NOT the cost of miscalibration: it mixes in the
        # difficulty gap between calibration-condition and deployment attacks.
        "fnr_shift_diagnostic": round(float(np.mean(v_fnr)) - float(np.mean(c_fnr)), 4),
        "fnr_oracle": round(fnr_oracle, 4),
        # Same scores define and read back this quantile. This records only the
        # quantile/tie convention; it is an identity, not validation evidence.
        "same_sample_quantile_identity": round(float(np.mean(dep_bona < t_oracle)), 4),
        "fnr_price": round(price, 4),
        "log2_fpr_ratio": round(log2_fpr_ratio(float(np.mean(v_fpr)), len(dep_bona)), 4),
        "resolution_limited": bool(float(np.mean(v_fpr)) < 5.0 / len(dep_bona)),
        # Kept so the "k of n cells miss target" count can be bootstrapped: a
        # count over a threshold is a point estimate whose sampling distribution
        # the discreteness hides (docs/verification.md).
        "_fpr_draws": [round(x, 6) for x in v_fpr],
        "weighted_fpr_mean": round(float(np.mean(w_fpr)), 4),
        "aci_oracle_fpr": round(aci_f, 4), "aci_oracle_fnr": round(aci_n, 4),
        "w1_bona_oracle": round(w1(cal_bona, dep_bona), 4),
        "monitors": {m: round(v, 4) for m, v in mon.items()},
    }


def main():
    results = {"within": {}, "cross": {}}
    for model in MODELS:
        rng = np.random.default_rng(SEED)
        la = la_channel_slices(model)
        print(f"{model}: 21LA channel slices: " +
              ", ".join(f"{c}:{len(b)}b/{len(s)}s" for c, (b, s) in sorted(la.items())), flush=True)
        for c_cal in sorted(la):
            for c_dep in sorted(la):
                if c_cal == c_dep:
                    continue
                cell = run_cell(rng, la[c_cal], la[c_dep])
                if cell:
                    results["within"][f"{model}/{c_cal}->{c_dep}"] = cell
        print(f"{model}: within-corpus done", flush=True)
        cs = cross_slices(model)
        for a in CROSS:
            for b in CROSS:
                if a == b:
                    continue
                cell = run_cell(rng, cs[a], cs[b])
                if cell:
                    results["cross"][f"{model}/{a}->{b}"] = cell
        print(f"{model}: cross-corpus done", flush=True)

    # Cell 8: response-vs-W1 fit with cross-system transfer validation.
    # PREREG names the response "realized-FPR excursion". That quantity is the
    # saturating one, so both readings are reported: the pre-registered one and
    # the corrected one. The region-2 bar (held-out R^2 >= 0.5) is missed by
    # both, so the decision is unchanged -- but the substitution is a deviation
    # and is printed as such rather than made silently.
    def cells_of(model, key):
        return [(v["w1_bona_oracle"], key(v))
                for k, v in list(results["within"].items()) + list(results["cross"].items())
                if k.startswith(model + "/")]

    def transfer_fit(key):
        out = {}
        for fit_m, test_m in [("ssl", "aasist"), ("aasist", "ssl")]:
            fx, fy = map(np.array, zip(*cells_of(fit_m, key)))
            tx, ty = map(np.array, zip(*cells_of(test_m, key)))
            coef = np.polyfit(fx, fy, 1)
            pred = np.polyval(coef, tx)
            ss_res = float(np.sum((ty - pred) ** 2))
            ss_tot = float(np.sum((ty - ty.mean()) ** 2))
            out[f"fit_{fit_m}_test_{test_m}"] = {
                "slope": round(float(coef[0]), 4), "intercept": round(float(coef[1]), 4),
                "r2_transfer": round(1 - ss_res / ss_tot, 3) if ss_tot > 0 else None,
            }
        return out

    results["severity_vs_w1_transfer"] = transfer_fit(lambda v: abs(v["log2_fpr_ratio"]))
    results["excursion_vs_w1_transfer_PREREGISTERED"] = transfer_fit(lambda v: v["excursion"])

    # Cell 9 evaluation: monitor ROC vs off-target cells + Spearman, per model.
    # The old bar (|FPR-alpha| > 2pp) inherited the saturation: a collapsed cell
    # scored as "nothing to detect", so conservative collapses counted as monitor
    # true negatives. Bar is now "off target by more than a factor of two".
    def spearman(x, y):
        rx = np.argsort(np.argsort(x)).astype(float)
        ry = np.argsort(np.argsort(y)).astype(float)
        return float(np.corrcoef(rx, ry)[0, 1])

    results["monitor_eval"] = {}
    results["monitor_eval_PREREGISTERED"] = {}
    for model in MODELS:
        allc = [v for k, v in list(results["within"].items()) + list(results["cross"].items())
                if k.startswith(model + "/")]
        # (target key, flag rule) -- pre-registered reading kept alongside the
        # corrected one; the flag-definition change is post-hoc (PREREG cell 9
        # fixes "excursion > 2pp") and must be visible, not substituted.
        readings = [
            ("monitor_eval", np.array([abs(c["log2_fpr_ratio"]) for c in allc]), SEVERITY_BAR),
            ("monitor_eval_PREREGISTERED", np.array([c["excursion"] for c in allc]), 0.02),
        ]
        for dest, exc, bar in readings:
            flag = exc > bar
            for mname in ("w1_mixture", "entropy"):
                m = np.array([c["monitors"][mname] for c in allc])
                best = None
                for t in np.unique(m):
                    tpr = float(np.mean(m[flag] >= t)) if flag.any() else 0.0
                    fpr = float(np.mean(m[~flag] >= t)) if (~flag).any() else 0.0
                    if tpr >= 0.8 and fpr <= 0.2:
                        best = {"threshold": round(float(t), 4), "tpr": round(tpr, 3),
                                "fpr": round(fpr, 3)}
                        break
                results[dest][f"{model}/{mname}"] = {
                    "n_cells": len(allc), "n_flagged": int(flag.sum()),
                    "spearman_vs_target": round(spearman(m, exc), 3),
                    "achieves_tpr80_fpr20": best,
                }

    out = Path(__file__).parent / "results_drift.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


def self_check():
    """A collapsed cell must not read as on-target. Fails if saturation returns."""
    # Pin the scale itself: base, target, symmetry, and which n the floor uses.
    # Without these the cell-level assertions below pass under log10, under a
    # wrong target, or when the floor is fed the calibration n.
    big = 10 ** 9  # large enough that the floor cannot bind
    assert abs(log2_fpr_ratio(2 * ALPHA, big) - 1.0) < 1e-12
    assert abs(log2_fpr_ratio(ALPHA / 2, big) + 1.0) < 1e-12
    assert abs(log2_fpr_ratio(ALPHA, big)) < 1e-12
    assert abs(log2_fpr_ratio(0.0, 600) - np.log2((3.0 / 600) / ALPHA)) < 1e-12
    assert log2_fpr_ratio(0.0, 600) != log2_fpr_ratio(0.0, 6000), "floor must use deployment n"

    rng = np.random.default_rng(0)
    cal_bona, cal_spoof = rng.normal(0, 1, 5000), rng.normal(-4, 1, 5000)

    # Healthy: deployment matches calibration -> FPR ~ alpha, FNR ~ calibration.
    ok = run_cell(rng, (cal_bona, cal_spoof), (rng.normal(0, 1, 5000), rng.normal(-4, 1, 5000)))
    assert abs(ok["vanilla_fpr_mean"] - ALPHA) < 0.02, ok["vanilla_fpr_mean"]
    assert abs(ok["fnr_price"]) < 0.05, ok["fnr_price"]
    assert abs(ok["same_sample_quantile_identity"] - ALPHA) < 0.01, ok["same_sample_quantile_identity"]
    assert not ok["resolution_limited"]

    # Collapsed: deployment bona shifted far up, spoof with it. The threshold is
    # calibrated on the old scale, so nothing is flagged: FPR -> 0 AND FNR -> 1.
    bad = run_cell(rng, (cal_bona, cal_spoof), (rng.normal(6, 1, 5000), rng.normal(3, 1, 5000)))
    assert bad["vanilla_fpr_mean"] < 0.001, bad["vanilla_fpr_mean"]
    assert bad["vanilla_fnr_mean"] > 0.99, bad["vanilla_fnr_mean"]
    # The old scalar calls this graceful; the corrected ones must not.
    assert bad["excursion"] <= ALPHA, "premise of the bug: |FPR-alpha| saturates at alpha"
    assert bad["log2_fpr_ratio"] < -SEVERITY_BAR, bad["log2_fpr_ratio"]
    assert bad["fnr_price"] > 0.5, bad["fnr_price"]

    # The correctness audit's counterexample: when the calibration condition's
    # own attacks are near-undetectable, the calibration-referenced difference
    # goes quiet while the miscalibration is in fact maximal. fnr_shift is a
    # diagnostic; only the oracle-referenced price may carry a claim.
    hard_cal = rng.normal(-0.2, 1, 5000)
    sneaky = run_cell(rng, (cal_bona, hard_cal), (rng.normal(6, 1, 5000), rng.normal(3, 1, 5000)))
    assert sneaky["fnr_shift_diagnostic"] < 0.10, sneaky["fnr_shift_diagnostic"]
    assert sneaky["fnr_price"] > 0.5, sneaky["fnr_price"]

    print("drift_map self-check passes: conservative collapse is visible "
          f"(excursion {bad['excursion']:.3f} 'graceful' vs log2 ratio "
          f"{bad['log2_fpr_ratio']:.2f}, price {bad['fnr_price']:+.3f}); and the "
          f"calibration-referenced diagnostic misses a maximal miscalibration "
          f"({sneaky['fnr_shift_diagnostic']:+.3f}) that the oracle-referenced "
          f"price catches ({sneaky['fnr_price']:+.3f})")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--self-check":
        self_check()
    else:
        main()
