# EXP-110 — Report: channel-matched training materially reduces, and does not remove, transport failure

Arm 2 trained 2026-09-09 (30 epochs, GPU v2 run `4f1b36b6…`, resumed once from the
epoch-8 checkpoint after the queue migration); dev sweep, tied-checkpoint pick and 21LA
scoring 2026-09-09 19:35 → 2026-09-10 00:28 (run `5dd46900…`, `run_arm2_score.sh`).
Analysis `analyze_arm.py arm2` under the trial-phase protocol (AMENDMENT of 2026-09-09).
Registered readings applied as frozen in PREREG.md.

## Kill criteria

- Codec augmentation import-verified and 5/5 codecs alter the waveform (`codec_gate.json`,
  2026-08-21).
- Arm 2 did not diverge; best dev EER 0.040% (epoch 28) against arm 1's 0.047% (epoch 20),
  within 1 pp. The five dev-tied checkpoints are 28, 27, 29, 15, 10 (dev EER 0.04–0.16%;
  the selection bar of `score_epochs_arm.py` reads "tied", as it did for arm 1).

## Primary statistic, K = ordered 21LA pairs (of 42) with |log2(FPR/α)| > 1

Aggregation order is executed, not described: `verify_manuscript_values.py` recomputes
every value the manuscript prints and exits non-zero if one stops reproducing. K is
counted per checkpoint over the 42 ordered pairs; a run's median is taken over its five
selected checkpoints; the maximum excess FNR is taken over every cell of every selected
checkpoint in an arm, and over all seeds for SSL-AASIST.

The two printed ranges are different objects. SSL-AASIST's spans the three run medians,
one per seed; AASIST's spans the five checkpoint values of its single run. The registered
material-reduction rule is evaluated on AASIST's selected-checkpoint ranges, never on a
range of run medians.

| Arm | tied checkpoints | K per checkpoint | median | range | worst |log2| |
|---|---|---|---:|---|---|
| 1, EXP-401 recipe (19LA train) | 20, 21, 27, 28, 29 | 32, 25, 32, 30, 27 | 30 | 25–32 | 5.30 (all five) |
| 2, channel-matched (same recipe, train set passed through the five 21LA codec families) | 10, 15, 27, 28, 29 | 9, 10, 16, 18, 17 | 16 | 9–18 | 1.57–2.34 |

**Registered reading: arm 2's K range lies entirely below arm 1's, and the medians differ
(14) by more than the wider within-arm range (9). Channel-matched training materially
reduces transport failure.** It does not remove it: 9–18 of the 42 pairs still miss the
5% target by more than 2× at the transported threshold, split evenly between conservative
(5–8) and liberal (4–10) misses.

## Secondary, descriptive (from `results_arm2.json`, not pre-registered)

- Severity collapses: the worst pair drops from log2 = −5.30 (arm 1, PSTN→G.722 among
  others) to −1.6 to −2.3. The flagship-type cell PSTN→G.722 realizes 1.4–2.9% FPR on arm 2
  against 0.59–0.67% on arm 1.
- The spoof-side cost of transport disappears on arm 2: maximum cost over the 42 pairs is
  0.1–0.2 pp per checkpoint (arm 1: 1.2–8.6 pp; the AASIST flagship in the paper: 69.4 pp).
  This is the EXP-123 mechanism seen from the other side: a detector with no weak
  calibration channel has nothing to pay when the threshold moves.
- Direction structure persists: the conservative misses of arm 2 remain the GSM/PSTN rows
  (the channels the augmentation reproduces least faithfully), the liberal ones the
  GSM/PSTN columns.

## What this changes for the next version

- The ICASSP recommendation "recalibrate after a channel change" gets its complement:
  training on the deployment channels roughly halves the pair count that needs it and
  removes the spoof-side cost, on this one recipe and seed, while a quarter to two fifths
  of the pairs still need target-channel recalibration for the 5% target to hold.
- Reporting rule kept from the PREREG: one seed per arm, one architecture, one training
  corpus; the within-arm range is checkpoint variance, not seed variance. No claim
  generalises to other architectures or unseen channels.

## Budget

PREREG derived 20.4 GPU-h. Actual: arm 1 scoring 2026-08-21 (see EXP-401); arm 2 training
≈ 11.5 GPU-h across two attempts (killed at 08:55 by the queue migration, resumed from
epoch 8); dev sweep + 21LA scoring 4.9 h wall-clock under host-memory pressure (the
dataloaders ran at roughly a third of the calibration.md scoring rate). One scoring attempt
failed before starting (relative command path under the v2 queue) and one on an import
path; both fixed in `run_arm2_score.sh` / `score_epochs_arm.py`.

## What would make this worthless

- Reading the K reduction as an EER effect (dev EERs are equal to within 0.01 pp).
- Claiming the residual 9–18 misses are noise: they exceed the 2× bar on every tied
  checkpoint and sit on the same channel pairs.
- Quoting the arm-2 spoof-side cost against the paper's AASIST flagship as if they were
  the same detector; arm 1 (same SSL-AASIST recipe) already pays at most 8.6 pp.

## Operational note for the next factorial (2026-09-11)

The recipe's training loader runs eight workers and is not persistent, so every epoch
re-forks a parent that already holds the 317M-parameter XLS-R model. Each worker then
carries about 1.1 GB of copy-on-write pages it never touches again; the kernel swaps them
out, resident memory per worker stays near 0.4 GB, but the host's swap fills. Setting
`persistent_workers=True` would stop the per-epoch re-fork entirely.

It was not changed during this campaign, and neither was `num_workers`. RawBoost runs
inside `__getitem__` with `--algo` defaulting to 5, and the loader is built without
`worker_init_fn` and without a generator, so which worker draws which augmentation depends
on the worker count: changing either mid-factorial would fold the loader configuration into
the arm and seed contrasts. Both are decisions for the start of a campaign, not its middle.

## Seed factorial: result, 2026-09-13

All six runs completed, 30 of 30 epochs each, five development-tied checkpoints scored on
21LA per run. `analyze_arm.py <run>` produces every number below; nothing here is copied
from a log.

| run | K median | K range | max excess FNR over the destination oracle | selected epochs |
|---|---:|---|---:|---|
| arm1 | 30 | 25--32 | +8.57 pp | 20/21/27/28/29 |
| arm1_s1235 | 33 | 32--34 | +17.50 pp | 8/11/18/22/28 |
| arm1_s1236 | 32 | 32--34 | +8.84 pp | 12/15/17/18/29 |
| arm2 | 16 | 9--18 | +0.18 pp | 10/15/27/28/29 |
| arm2_s1235 | 20 | 19--20 | +0.55 pp | 11/20/24/25/29 |
| arm2_s1236 | 23 | 21--30 | +0.50 pp | 14/19/20/21/29 |

Every arm-2 median (16, 20, 23) lies below every arm-1 median (30, 32, 33); the seedwise
reductions are 14, 13 and 9 of 42 ordered pairs. Arm 2's largest excess FNR is 0.55 pp
against 17.50 pp in arm 1, so the registered 1-pp ceiling holds on all three seeds. Both
registered numerical criteria are therefore met. No kill criterion fired: each run's best
development EER stays within 1 pp of its arm-mate's.

**The registered significance reading was withdrawn.** See the post-result addendum in
PREREG.md. The registered exact Mann-Whitney assumes exchangeability across all six runs,
which the common-seed design does not establish; the paired alternative is a post-result
sensitivity calculation, not the registered test. The manuscript reports the reduction
descriptively and carries one clause in its conclusion.

**Caveat that travels with the statistic.** The development signal that elects each run's
five checkpoints is saturated: development EER spans 0 to 0.16% across the six runs, a
handful of absolute errors, and arm1_s1236 has two epochs at exactly 0.0 while arm1_s1235
has two bit-identical at 0.00449%. The five checkpoints are separated by one or two
development errors. This supports the median-over-five design; it does not make the median
more precise than the signal behind it. K itself is not saturated, spanning 9 to 34.

**Estimated versus actual.** The pre-registration derived 20.4 GPU-h for the original two
arms. The factorial multiplied that by three seeds, and the measured checkpoint span across
the six training runs is about 156 h of wall clock on a shared GPU, against roughly 61 h of
derived GPU time for six runs. The gap is contention, not a bad estimate: the runs spent
long stretches paused or queued behind other work, and two of them alternated on the device
for a full day at roughly half throughput each. The per-epoch cost itself matched: about 20
minutes per training epoch and 28 minutes per scored checkpoint.

**Provenance limit.** `results_stage0.json` does not record its own seed; run identity rests
on the directory name.

**Not done.** The two AASIST arms, which would test whether the reduction holds on a second
architecture, were still training when this was written.

## AASIST arms: partial replication, 2026-09-14

Both AASIST arms completed 100 of 100 epochs with five development-tied checkpoints scored
on 21LA. One seed per arm, 1234.

| run | K median | K range | excess FNR median | max | cells above 10 pp | selected epochs |
|---|---:|---|---:|---:|---:|---|
| aasist_arm1_s1234 | 30 | 28--33 | 0.15 pp | 56.1 pp | 30% | 69/74/81/87/98 |
| aasist_arm2_s1234 | 22 | 18--27 | 0.16 pp | 69.4 pp | 14% | 40/50/53/56/66 |

**The K reduction is directional, and the registered criterion is NOT met.** The
channel-matched arm's median K is 22 against 30 for its control, and the two ranges do not
overlap (18--27 against 28--33). The registration asks for more than separation on this arm:
"AASIST arm 2's K range entirely below arm 1's, medians differing by more than the wider
within-arm range". The wider within-arm range is 9 (18--27); the medians differ by 8. So the
registered "materially reduces on AASIST as well" reading does not apply, and what is
reportable is a descriptive directional decrease.

An earlier draft of this section claimed the criterion was met, using "a strictly lower
median K". That weaker wording came from the decision rule fixed for this session's own
read-out, not from PREREG.md, and conflating the two understated what the registration asks.
Raised as finding B1 of the audit of 2026-09-14 and corrected the same day.

**The spoof-side benefit does not.** On SSL-AASIST the matched arm drove the largest excess
FNR over the destination oracle from 17.5 points down to 0.5. No matched-arm cell exceeds 10
points on any of the three seeds; the control arm's seed 1235 has 7 of its 210 pair-checkpoint
cells above 10, which is where its 17.5-point maximum sits. On AASIST the largest excess FNR rises, 56.1 points in the control against
69.4 in the matched arm, and 30 of 210 matched-arm pair-checkpoint cells still exceed 10 points, 14%, against 63 of
210 in its control, 30%. So matched training reduces how often the threshold misses its target on this
architecture without removing the expensive misses.

The two architectures do not start from the same place: AASIST's control already puts 30%
of cells above 10 points where every SSL-AASIST arm puts none. The comparison that is
registered is within-architecture, and that is how it is read here.

One seed per arm. This is a replication of direction, not of magnitude, and it establishes
nothing about seed variance on AASIST.
