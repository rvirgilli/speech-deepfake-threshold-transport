# EXP-103 — Report: the drift map replicates on recording-disjoint data; exploratory cost-map sensitivity

## Correction — 2026-08-17 estimator audit

Reading 2a survives: threshold-transport failure persists on recording- and
speaker-disjoint data. Its 71% versus 50% cross-grid difference is descriptive,
not an effect of disjointness: corpus, codecs and attacks also change.

Reading 2b survives only as a point sensitivity. The held-out $R^2=0.592$
reproduces, but the interval and p-value below are **void**: the original report
Fisher-transformed `sqrt(held-out R²)` as though it were an iid Pearson
correlation at n=12. Held-out regression $R^2$ is not that statistic, directed
cells reuse crossed conditions, and the committed `analyze.py` did not implement
the regression. `cost_map_sensitivity.py` now reproduces the point and
delete-one-condition sensitivities in
`artifacts/cost_map_sensitivity.json`, explicitly with `inference: null`.

The invalid inferential and causal language has been corrected in place below;
the prior text remains available in version history rather than beside the
reviewable artifact where it could be mistaken for a current claim.

- **Current status:** 2a replicates; 2b clears its descriptive point bar but has
  no valid inferential interval or p-value.
- **Budget:** PREREG ≈2–3 GPU-h for two detectors → **actual ~4.7 h** for both
  detectors over 680,774 utterances each, inside a 5.7 h chain that also carried
  EXP-401's reading. No download, no transcode, no DGX.
- **Gate:** VIABLE. SSL-AASIST oracle FNR at 5% FPR = **7.9%**; AASIST = **63.9%**.

## The gate, and the constraint it imposes

AASIST is **overlap-dominated on A5** by A2's own definition (oracle FNR > 50%),
so it is excluded from viable claims exactly as BRSpeech and ITW cells already
are. This is not a workaround — it is the paper's existing machinery applied
consistently.

**The consequence is real and must be stated: the price axis rests on one
detector.** A2's flagship cell is an *AASIST* cell (PSTN→G.722), and the
replicate cannot reproduce it on that detector because AASIST cannot discriminate
A5's attacks at all. The FPR-severity axis needs no spoof labels and is reported
for both.

## Reading 2a — the mechanism replicates; the rate difference is descriptive

Like-for-like against 21LA's within-corpus channel/transmission transport, SSL-AASIST, on the
**twin-free** subset (sources appearing under exactly one condition,
speaker-disjoint 50/50 by seeded hash on top):

| | 21LA within-corpus | **A5 twin-free** |
|---|---|---|
| ordered condition pairs | 42 | **132** |
| miss target by >2× | 17 (**40%**) | **99 (75%)** |
| conservative / liberal | 21 / 21 | 65 / 67 |
| median signed log₂(FPR/α) | −0.021 | +0.114 |
| range | — | [−8.32, +3.76] |

**The transported threshold misses target by more than 2× on three-quarters of
deployment pairs**, against 40% on 21LA — on data where calibration and
deployment share no recording and no speaker. This establishes persistence,
not a disjointness effect: corpus, codecs, attacks and source composition also
change, so the two rates cannot be causally compared.

The direction split is symmetric on both corpora (65/67 here, 21/21 there),
which is consistent with A2's corrected abstract: the conservative collapse is
forced *within* the ±α band, not across the whole grid.

## Reading 2b — exploratory cross-corpus cost-map sensitivity

The severity→price map fitted on **21LA only**, applied **entirely held out** to
A5: different corpus, speakers, codecs, and unseen attacks.

- fitted on 21LA: `price = −0.0767 · log₂(FPR/α) − 0.0167`
- **held-out R² on A5 = +0.592**, against the pre-registered bar of **0.5**
- within-A5 Spearman(log₂ ratio, price) = **−0.901**, n=66 viable cells

This is a descriptive cross-corpus reconstruction, not an inferential transfer
law. Directed cells reuse 12 crossed conditions, so the 66 cells are not iid
units and no valid interval or p-value is available. Deleting one A5 evaluation
condition at a time gives held-out R² **0.451--0.640**; deleting one
non-resolution-limited 21LA training condition and refitting gives
**0.646--0.680**. Both ranges and the primary point are recorded in
`artifacts/cost_map_sensitivity.json`.

**Scope, stated before it can be overclaimed.** This is transfer across
*corpora* for a *fixed detector*. Transfer across *detectors* was tested
separately and fails — 37.7%/76.0% envelope coverage against a ~95% requirement,
because the shape transfers and the gain does not
(`../EXP-403-a2-detector-families/bound_transfer.py`). The claim is
corpus-transfer, and the detector-transfer negative belongs beside it.

## The twin arm — what the twinning is worth

The 19.5% of sources appearing under 11 conditions form a paired grid with
exactly 21LA's twin structure, on the same corpus and detector:

| SSL-AASIST | pairs | >2× miss | median log₂ ratio |
|---|---|---|---|
| twin-free | 132 | 75% | +0.114 |
| **crossed (twinned)** | 110 | 69% | −0.020 |

**The twinning does not inflate severity here — if anything it slightly
depresses it** (69% vs 75%). That is directly relevant to A2's dependence
disclosure: the concern that 21LA's shared-recording structure inflates the
headline is not supported on the one corpus where both structures can be
measured side by side.

**Descriptive only, as pre-registered.** n=1 corpus, one detector, and the two
arms differ in which sources they contain as well as in structure. No general
correction factor is claimed and none should be quoted.

## Deviations from PREREG

- **Two detectors, not four.** The M1 line declined to have its frozen A5 arm
  run on this account; scope narrowed accordingly and cost fell.
- **The twin-free restriction was added blind**, after discovering that the
  original "recording-disjoint" premise read the wrong protocol column. It made
  the experiment better specified rather than weaker.
- **AASIST's price axis is void**, per the gate. Anticipated by the frozen
  branch, which is why the branch existed.

## What would make this worthless

- **Quoting R² = 0.592 without its descriptive delete-one-condition range, or
  manufacturing an interval from the 12 crossed conditions.**
- **Reading 2b as a detector-transferable bound.** It is not; that was tested
  and failed.
- **Pooling the twin-free and crossed arms**, or quoting the twin comparison as
  a correction factor.
- **Claiming the flagship cell replicates.** It cannot — it is an AASIST cell and
  AASIST is overlap-dominated on this corpus.


## Addendum, 2026-09-09: trial-phase correction supersedes Reading 2b

The 21LA training cells of the cost map included the ASVspoof 2021 hidden
(VAD-trimmed) phase (`../EXP-102-a2-campaign/AMENDMENT-3-trial-phase-protocol.md`).
With that phase excluded and the map refitted (`cost_map_sensitivity.py`,
`artifacts/cost_map_sensitivity.json`), the held-out $R^2$ on the 66 viable A5
SSL-AASIST cells is **$-0.81$** (delete-one-condition range $-1.31$ to $-0.67$;
Spearman $-0.90$). Reading 2b is therefore withdrawn from the manuscript, which
no longer prints a cost map. Readings 1 and 2a (the ASVspoof 5 replication) do
not depend on 21LA and are unchanged. The values above this addendum are
superseded and kept for the record.

## Addendum, 2026-09-09: usable-pair cost summary and same-condition control

Two descriptive additions requested after the external review, both computed
from the twin-free arm with the estimators of `analyze.py`:

- `a5_usable_cost.py` → `artifacts/a5_usable_cost.json`. On the 66 SSL-AASIST
  pairs whose destination has oracle FNR ≤ 50% for every source, the spoof-side
  cost of the transported threshold (mean FNR minus deployment-oracle FNR) has
  median +41.5 pp, IQR 4.6–70.2 pp; 47 pairs exceed +10 pp, 51 are positive,
  49 miss the FPR target conservatively by more than 2×. AASIST has no usable
  destination, so no cost is summarised for it.
- `a5_diagonal.py` → `artifacts/a5_diagonal.json`. Same-condition control:
  calibrate on speaker half 0 and deploy on speaker half 1 of the same
  condition (N = 500, B = 1000). Over the 12 conditions, no cell misses target
  by more than 2×; realized FPR spans 3.2–6.0% (SSL-AASIST, max |log2| 0.67)
  and 3.8–7.3% (AASIST, max |log2| 0.54). The off-diagonal failures of Reading
  2a therefore follow the codec change, not the speaker split.

Both are descriptive; the dependence unit is the ordered condition pair, and
pairs sharing a destination are not independent.
