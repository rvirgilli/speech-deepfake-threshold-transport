# EXP-102 addendum — score-weighting diagnostics and estimator-scope correction

> **Terminology correction (2026-08-28):** this document originally called our
> clipped 15-bin mixture-score heuristic “weighted CP” and attributed it to
> Tibshirani. That attribution is superseded by
> `AMENDMENT-2-estimator-scope.md`. Literature descriptions below remain
> descriptions of cited work; every statement about our estimator means only
> the mixture-score heuristic actually implemented.

Date: 2026-08-15. Trigger: the Interspeech coordinator flagged Leong, *Online
Shift Detection and Conformal Adaptation for Deployed Safety Classifiers*
(arXiv 2606.11949v4), which reports that weighted conformal prediction fails on
decoder classifiers because logistic density-ratio estimation in 3584–4096
dimensions separates the samples perfectly and clips every importance weight to
zero; projecting to ≤32 dimensions restores coverage (+33 pp). The question put
to us: does our score-weighting negative run through the same degeneracy, in which
case it is a known implementation artifact with a known fix rather than a
result about audio?

Answer: **no, and the diagnostic strengthens the negative.** But running it
surfaced a second, larger problem in the paper, recorded below as finding 2.

Probe: `wcp_weight_check.py` (self-checking; recomputes the weights
`drift_map.run_cell` actually uses, 50 draws × 108 cells, N=500, α=5%).

## Finding 1 — not Leong's signature, but weight localisation is real

Our estimator is a **1-D histogram density ratio on the detector score**,
clipped to [0.1, 10] (`drift_map.density_ratio_weights`). Leong's mechanism is
unreachable here for two independent reasons, both now measured rather than
argued:

- **Dimension.** We are at 1; Leong's fix is "project to ≤32". The bona-fide
  score supports of calibration and deployment overlap heavily in every cell,
  so the perfect separability that drives their collapse cannot arise.
- **Floor.** No weight is zero. Across all 108 cells the realized range is
  [0.100, 10.000], and the floor means that *total* degeneracy would not
  collapse the method — its weighted quantile is invariant to the scale of `w`,
  so all-weights-equal reverts silently to the unweighted quantile. This is a distinct
  failure mode from Leong's and `n_eff` is blind to it by construction; the
  probe therefore tests both signatures separately.

What *is* present is partial localisation, on the cells one would expect:

| | value |
|---|---|
| cohort mass pinned at the 0.1 floor | 0–68% by cell (40 of 108 cells above 1%; 42 above 0) |
| lowest effective sample size | `ssl/asv21la_nocodec->itw`, n_eff/N = **0.084** (500 → ~42) |
| n_eff/N vs \|FPR miss\| | Spearman **−0.65**, p = 2.7e−14 |
| n_eff/N vs *signed* miss | Spearman **−0.07**, p = 0.47 |
| frac-at-floor vs \|FPR miss\| | Spearman **+0.41**, p = 9.0e−06 |

So localisation sets **how large** the heuristic's miss is but not **which
way** it goes. That is the honest decomposition: the estimator's effective
sample size explains the magnitude, and the direction is left to the
exchangeability break. The worst-localised cell is not the worst-failing one
(n_eff/N = 0.084 lands at FPR 5.78%, near target), which is what rules out a
pure estimator artifact.

This is the mechanism Hennhöfer & Preisach (arXiv 2603.23205) give
theoretically — weights localise, effective sample size drops, p-values go
conservative — instantiated on a measured grid. Their direction prediction
(conservative) matches ours; we can cite them as theory for the measurement.

Head-to-head at ±2 pp of the 5% target, all 108 cells:

| policy | within 2 pp | median \|miss\| | worst |
|---|---|---|---|
| unweighted quantile | 26/108 | 4.02 pp | 37.9 pp |
| **15-bin mixture-score weighting heuristic** | **33/108** | **3.28 pp** | **23.5 pp** |
| ACI with oracle label feedback | 99/108 | 0.32 pp | 14.0 pp |

The heuristic beats unweighted on 70 of 108 cells and still controls nothing:
a marginal improvement, no operating point.

## Finding 2 — the paper overclaims, and reports neither number

**The then-current abstract said "label-free conformal repairs do not restore control" and
the Conclusion said every label-free correction fails. §4 reported no
score-weighting or adaptive result at all** — that superseded claim had no measurement behind it in
the text, which is exactly the omission `check_numbers.py`'s presence half
exists to catch and does not currently cover.

Worse, the ACI row above is **not a label-free repair**. `drift_map.aci_fpr`
updates α from the true label of every deployment item
(`a += γ·(α − flagged)` inside `if is_bona`); its docstring says so. It is the
*labeled* reference, and it is the best-performing policy in the grid. We have
**not** tested a label-free Gibbs–Candès variant. Zhou & Wang's §3.5 names
weighted and adaptive conformal as constructive directions; our implementation
is neither the Tibshirani construction nor a label-free adaptive method.

Disposition, pre-committed here before any rewrite:

1. §4 must report the score-weighting measurement it already has, with the
   non-degeneracy statement, or drop the claim.
2. The blanket "every label-free correction fails" must be scoped to what was
   measured: the mixture-score weighting heuristic, and the three unlabeled corrections of `eerhides26`
   re-measured in `results_cmethods.json`. ACI is reported as the labeled
   upper bound and labeled as such, never as a failed label-free repair.
3. Label-free ACI is listed as untested, with the reason — and the reason is
   not cost. ACI's update needs feedback on whether each prediction was
   correct, which at our operating point means knowing whether a flagged item
   was bona fide. A deployment holding no target labels cannot supply that. The
   only label-free substitute is to update from the *mixture* flag rate and
   divide out an assumed attack prevalence π, which makes the repair a function
   of the one quantity our pre-registered monitoring negative shows is not
   observable: none of our monitors survives a prevalence shift. So "label-free
   ACI" is not a variant we declined to run for budget reasons; it is
   ill-posed in this resource class unless π is supplied, and if π is supplied
   the method is no longer label-free in the sense the claim uses.

   That is a paper sentence, not a deferral — but it is also a design decision
   made after seeing results, so if we ever do run the π-parameterised version
   it needs its own PREREG with the sensitivity band on π fixed in advance.

Neither finding is self-certified: item 2 is a correctness defect I found in
my own paper and it goes to an independent audit before any claim is written.

## Cost

Probe + correlations: ~6 min CPU, no GPU. No new anchors for
`docs/calibration.md`.
