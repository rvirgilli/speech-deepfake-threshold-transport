# EXP-102 amendment 2 — estimator identity and finite-pool reference

Date: **2026-08-28**. This amendment corrects estimator labels in the original
preregistration before publication; it does not change code, outcomes, or the
registered decision branches.

## Score-weighting estimator

`PREREG.md` cell 7 called the registered score-weighting baseline “weighted
conformal (Tibshirani 2019).” That attribution was wrong. The estimator that ran
is the one implemented in `drift_map.py::density_ratio_weights` and
`weighted_quantile`:

1. pool calibration bona-fide and spoof scores into a calibration mixture;
2. pool deployment bona-fide and spoof scores into a deployment mixture;
3. estimate a one-dimensional deployment/calibration mixture-score density
   ratio with 15 equal-width histogram bins;
4. floor the histogram denominator at `1e-8`, clip the ratio to `[0.1, 10]`,
   and evaluate it at the 500 bona-fide cohort scores;
5. take the weighted empirical 5% quantile of that cohort.

This is a **clipped 15-bin mixture-score importance-weighting heuristic**. It is
not the covariate-shift conformal construction of Tibshirani et al.: it does not
estimate a target/source bona-fide covariate ratio and it carries no test-point
weight. All 33/108, 26/108, 70/108, 54/108, localisation, and effective-sample-
size results are scoped to this heuristic.

The outcome-adaptive row uses the true deployment bona-fide labels to update its
target online. It is an **oracle-label adaptive ceiling**, not a label-free
Gibbs–Candès repair. It remains useful only as a record of what outcome feedback
buys.

## Conditional-FPR reference

The `Beta(k,N+1-k)` law is exact for the population FPR of an order-statistic
threshold under iid continuous sampling from a fixed score distribution. The
N-sweep implemented here samples without replacement from a fixed finite score
pool and evaluates on the complement. Its empirical spread is therefore a
finite-population statistic; the serialized Beta band is only the theoretical
iid population reference. `n_sweep.py` and regenerated `results_nsweep.json`
now encode this distinction explicitly.

