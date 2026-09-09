# Post-review analysis: does speaker clustering inflate the calibration interval?

**NOT PRE-REGISTERED.** Everything in this file — the speaker-disjoint
comparison, the permutation test, the seed sweep, and the leave-one-speaker-out
— was run on 2026-08-15 in response to external review MAJOR 2, after the
results it examines already existed. It is kept visibly separate from
the pre-registered set because MINOR 2 of the same review says our
preregistration claims cannot be credited without a record — folding a post-hoc
analysis in beside them would compound exactly that objection.

## The question

The reviewer asks whether the 1,000 calibration resamples are utterance-level or
speaker-clustered, and warns: *"If several utterances per speaker are treated as
500 independent calibration points, the stated Beta law and the implied sample
efficiency do not follow for new-speaker deployment. The empirical spread over
random utterance splits would also be optimistic."*

**The premise is correct.** Verified in the code: `drift_map.py:161` and
`analyze.py:101` both draw `rng.choice(len(cal_bona), N_CAL)` — utterance-level,
no clustering anywhere. ASVspoof 2021 LA has **67 bona-fide speakers with 8–102
utterances each**, so a 500-utterance cohort contains ~65 of the 67.

## First attempt, and why it was not yet a claim

Comparing 90% interval widths under utterance-level versus speaker-*disjoint*
calibration gave six bare point estimates: four flat, two roughly doubled
(`aasist/pstn` 3.51→6.98, `aasist/gsm` 3.32→6.12).

That is not evidence. Two of six, no uncertainty on any of the twelve widths,
and **the two that moved are the cells with the fewest effective calibration
speakers under channel transmission — which is what plain variance-inflation-at-
small-n predicts, with no detector-conditional content at all.** An interval
width computed from median-13-speaker draws is itself a noisy statistic.

## The test that settles it

Permutation under the null of **no clustering effect**, which controls the
alternative explanation directly: reassign scores to speakers at random while
**preserving the group sizes exactly**. The null therefore has the same 67
groups, the same 8–102 utterance counts, and the same marginal score
distribution — only the speaker↔score association is destroyed. Any
small-effective-n inflation is present in the null too and cancels.

Observed = 90% interval width of realized FPR under speaker-disjoint calibration
(median 13 speakers calibrate, the remaining 54 evaluate), B=400 draws,
PERM=200, N=500, α=5%, same-condition.

| cell | observed width | null median | null 95th | p |
|---|---|---|---|---|
| ssl/none | 3.48 | 3.44 | 4.04 | 0.448 |
| ssl/pstn | 3.85 | 3.43 | 4.00 | 0.104 |
| ssl/gsm | 3.40 | 3.42 | 4.07 | 0.517 |
| aasist/none | 3.24 | 3.49 | 4.12 | 0.706 |
| **aasist/pstn** | **6.98** | 3.45 | 3.94 | **1/201** |
| **aasist/gsm** | **5.84** | 3.42 | 3.98 | **1/201** |

**Outcome (a).** Both transmitted-condition AASIST cells exceeded **all 200** null draws,
by margins of 77% and 47% over the null's 95th percentile. Bonferroni over six
tests puts the threshold at 0.0083; both clear it.

With zero exceedances in 200 sampled nulls, the usual corrected Monte Carlo
randomization-test p-value is $(0+1)/(200+1)=1/201$. This is not an upper
confidence bound on the exhaustive permutation-tail probability.

## Two readouts disagreed, and the diagnosis is in the estimator's noise

The six widths were reported twice an hour apart and **five of six moved**
(0.05–0.34) while `aasist/pstn` printed 6.98 both times. That pattern needed
explaining before any of it became a claim.

**Cause: seed and draw count. Nothing else.** Run 1 used seed 11, B=800, and
drew the utterance-level and speaker-disjoint samples interleaved from one
generator; run 2 used seed 23, B=400, disjoint only. Run 2 also carried an
`if len(ev) < 200: continue` guard that run 1 lacked — **measured, that guard is
inert**: with and without it the widths are identical to two decimals on all six
cells, because the smallest calibration draw still leaves well over 200
evaluation utterances.

**All six differences are Monte Carlo noise.** Across 10 seeds at B=400 the
seed-to-seed SD of the width is **0.12–0.23 points**, so every observed movement
sits inside one SD. `aasist/pstn` matching to two decimals twice is coincidence
at that resolution, not stability — with SD 0.17 the chance of two draws sharing
a 0.01-wide bin is a few percent per cell, and there were six cells.

**Canonical values are the multi-seed means. The single-seed numbers above are
superseded, and the reason matters more than the fact: this statistic has a
seed-to-seed SD of ~0.2 points.** Anyone re-running it will get different
widths from the ones printed here, and should not read that as disagreement —
report a multi-seed mean with its SD, or two of your own runs will look like a
contradiction the way mine did.

| cell | width, mean ± SD over 10 seeds |
|---|---|
| ssl/none | 3.45 ± 0.14 |
| ssl/pstn | 3.75 ± 0.13 |
| ssl/gsm | 3.54 ± 0.12 |
| aasist/none | 3.53 ± 0.15 |
| **aasist/pstn** | **7.03 ± 0.17** |
| **aasist/gsm** | **5.72 ± 0.23** |

The separation is ~3.3 points against SDs under 0.25, so the permutation verdict
is unchanged — but the statistic now carries the uncertainty it should have had
when it was first reported.

## Leave-one-speaker-out: not two strange speakers

The permutation excludes small-n inflation but not a few atypical speakers
carrying the effect. Dropping each of the 67 speakers from the calibration pool
in turn, recomputing the width:

| | full pool | min over drops | median | collapses to SSL level (~3.6)? |
|---|---|---|---|---|
| aasist/pstn | 7.06 | **4.89** (drop `LA_0032`) | 6.99 | no |
| aasist/gsm | 5.54 | 5.18 (drop `LA_0021`) | 5.53 | no |

**The effect survives on both cells**, so it is not an artifact of one or two
speakers. But it is not uniform either: on `pstn`, removing `LA_0032` alone cuts
the excess over the SSL level by about 60% (7.06→4.89 against ~3.6). That single
speaker is disproportionately influential and is named here rather than averaged
away. On `gsm` the effect is flat across drops.

## What this licenses, and what it does not

**Licensed:** speaker clustering of bona-fide scores inflates the calibration
interval for AASIST under channel transmission and not measurably for SSL-AASIST or
for either detector on clean audio. The dependence unit is the speaker (67), and
the honest n for the disjoint experiment is 67 minus the calibration draw.

**It lands on `aasist/pstn`, which is the calibration condition of the paper's
flagship cell** — so the detector the paper already calls more fragile on both
axes is also the one whose calibration is speaker-sensitive.

**Not licensed:** any claim that this generalises beyond these two detectors and
one corpus. The effect size should be quoted as the multi-seed widths above with
their SDs, not as "roughly doubles". And on `aasist/pstn` one speaker carries
about 60% of the excess, so the per-cell magnitude is not a stable property of
the detector-condition pair even though its existence survives every single-
speaker drop.

## Reproduce

Run `speaker_clustering_audit.py`. It writes
`artifacts/speaker_clustering.json`, including input hashes, the ten-seed width
sweep, all 200 permutation outcomes summarized by cell, and every
leave-one-speaker-out result. Scores come from
`EXP-001-scoring-campaign/scores/{ssl,aasist}_asv21la.csv.gz`; speaker and channel-
condition labels come from `keys/LA/CM/trial_metadata.txt` (the condition column's
historical field name is `codec`). The canonical multi-seed
means reproduce exactly. Null percentiles can move by about 0.1 point across
NumPy RNG versions; both affected cells still exceed all 200 null draws.


## Addendum, 2026-09-09: trial-phase correction

The tables above were computed on all three ASVspoof 2021 phases. After the
hidden (VAD-trimmed) phase was excluded (`AMENDMENT-3-trial-phase-protocol.md`),
`speaker_clustering_audit.py` was rerun and `artifacts/speaker_clustering.json`
regenerated. Under the corrected protocol every one of the six cells exceeds its
size-preserving permutation null (200 permutations; null medians 3.4--3.5 pp,
95th percentiles 4.0--4.1 pp; zero exceedances), with a median of 15
calibration speakers:

| cell | 90% width (pp), mean over 10 seeds | SD over seeds |
|---|---|---|
| ssl/none | 9.30 | 0.41 |
| ssl/pstn | 6.68 | 0.31 |
| ssl/gsm | 7.08 | 0.23 |
| aasist/none | 6.00 | 0.25 |
| aasist/pstn | 10.08 | 0.30 |
| aasist/gsm | 6.75 | 0.47 |

The earlier reading "two of six cells move" does not survive the correction;
the speaker-clustering inflation is present on every tested cell. The analysis
remains post-hoc and is reported as such.
