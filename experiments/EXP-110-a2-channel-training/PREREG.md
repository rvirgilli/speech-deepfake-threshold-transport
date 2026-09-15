# EXP-110 — does channel-matched training buy operating-point stability?

Pre-registered **2026-08-21**, before any training or scoring. Protocol:
[../../docs/experiment-protocol.md](../../docs/experiment-protocol.md). Follows
[EXP-109](../EXP-109-a2-cors-transport/PREREG.md), which established that A2's transport
failure is detector-general across 126 cells.

**Not yet authorised to run.** Derived cost is 20.4 GPU-h, above the 12-GPU-h standing
authorisation of 2026-08-17, so this needs an explicit decision.

## The claim this may and may not make

Phase 0 of the campaign established that *"robust training does not remove the need for
recalibration"* is **folklore in speaker verification** — trained calibration backends have
separated robustness from calibration for two decades (Ferrer et al., `2002.03802`,
`2102.01760`; the 2004 channel-robust line). Under the practitioner test it fails.

The admissible object is therefore a **measurement**: *how much* of A2's transport failure
does channel-matched training remove, in the bona-fide-only regime where an SV-style
calibration backend is unavailable because no spoof labels exist. Magnitude, not existence.

## The design problem, and the fix

A single trained checkpoint per arm would be indefensible here, and specifically indefensible
against **our own M1**: [EXP-401](../EXP-401-areaII-stage0/REPORT.md) measured that the 19LA
dev criterion cannot resolve its own top candidates — the top-5 checkpoints span 0.0353 pp of
dev EER against a one-utterance resolution floor of 0.0392 pp, so which checkpoint is "best"
falls to the training script's tie-break. A 1-vs-1 comparison would attribute to augmentation
whatever the tie-break happened to pick. Submitting that alongside M1 would be incoherent.

**Fix, using EXP-401's finding constructively:** score the **top-5 dev-tied checkpoints of
each arm** and treat their spread as the within-arm realisation variance. The between-arm
difference is only claimed if it exceeds that spread. This costs five scoring passes per arm
rather than five training runs, so realisation variance is bought at 4 GPU-h instead of 62.

## Arms — one variable

| Arm | Training | Status |
|---|---|---|
| 1 — baseline | SSL-AASIST on 19LA, authors' LA recipe, RawBoost `--algo 5` (series 1+2: convolutive plus impulsive noise) | **already trained**; all 30 epoch checkpoints exist from EXP-401 job 127 |
| 2 — channel-matched | identical recipe and seed, plus codec round-trip augmentation matched to the five 21LA VoIP codecs (a-law, µ-law, GSM-FR, G.722, Opus) | to train |

Only the augmentation differs. Same data, same recipe, same seed, same schedule, same
selection rule.

## Primary statistic

For each scored checkpoint, the 42-cell 21LA transport map from
[EXP-109 cell A](../EXP-109-a2-cors-transport/transport_released.py), and from it:

**K = the number of the 42 ordered condition pairs with |log2(FPR/α)| > 1.**

Reported per arm as the median K over its five tied checkpoints, with the min–max range.

## Registered readings, symmetric before outcomes

| Outcome | Reading |
|---|---|
| Arm-2 K range lies entirely below arm-1's, and the medians differ by more than the wider within-arm range | Channel-matched training **materially reduces** transport failure. A2's "recalibrate after a channel change" prescription weakens and the paper must say so and quantify it |
| The arms' K ranges overlap | The training effect is **not resolved** at this realisation variance. Reported as such, with the ranges printed — consistent with M1 |
| Arm-2 K is higher | Channel-matched training **worsens** operating-point transport while plausibly improving EER. The sharpest outcome, reported as the headline of this experiment |
| Arm 2 fails to reach a dev EER within EXP-401's resolution floor of arm 1 | The augmentation broke training rather than tested it. Infrastructure failure, no scientific reading |

EER is recorded for both arms but is **not** the primary statistic; a training arm that
improves EER while leaving K unchanged is the whole point of the measurement.

## Derived cost

From [calibration.md](../../docs/calibration.md) anchors: SSL-AASIST fine-tune 20.3 min/epoch
× 30 = 10.2 h; all-epoch dev selection 2.2 h; 21LA scoring 181,566 ÷ 64 utts/s = 0.79 h.

| Item | GPU-h |
|---|---:|
| Arm 1: score 5 tied checkpoints on 21LA | 4.0 |
| Arm 2: train | 10.2 |
| Arm 2: all-epoch dev selection | 2.2 |
| Arm 2: score 5 tied checkpoints on 21LA | 4.0 |
| **Total** | **20.4** |

Roughly one day of 3090 wall-clock, and it must finish by **Sep 1** to preserve the 16-day
verification reserve.

## Kill criteria

- The codec augmentation must be **import-verified on CPU** on a persistent venv before any
  GPU submission — 20 audio files through all five codec round-trips, hashes recorded. The
  EXP-201 anchor prices an unverified unfamiliar stack at ~35 min of lost queue time.
- Every augmentation codec must be checked to alter the waveform: a round-trip that returns
  the input unchanged is a silent no-op and would make arm 2 a duplicate of arm 1. Pass rate
  is predicted at 5/5 and checked against that prediction before training.
- If arm 2's training diverges or its dev EER exceeds arm 1's by more than 1 pp, stop and
  report infrastructure failure rather than a scientific result.

## What this cannot support

One seed per arm, one architecture, one training corpus. The realisation variance measured
here is **checkpoint** variance within a single training run, not **seed** variance across
runs — the seed factorial remains a next-paper design. No claim generalises to other
architectures or to unseen channels.

## Amendment (2026-09-09): trial-phase protocol and arm parameterization

`analyze_arm.py` now excludes the ASVspoof 2021 hidden (VAD-trimmed) phase, as every A2
analysis does since EXP-102 AMENDMENT-3, and takes the arm as an argument (`arm1` with
its fixed tied epochs 20/21/27/28/29; `arm2` with tied epochs read from its dev sweep by
the same top-5 rule). Under that protocol arm 1's five tied checkpoints give
K = 32/25/32/30/27 (median 30, range 25–32); the all-phase values 19–22 are superseded.
Arm 2 will be read against this range.

## Amendment (2026-09-10 01:05, before any run): seed factorial

Arm 1 and arm 2 each exist for one seed (the recipe default, 1234). Two more seeds per
arm, 1235 and 1236, with everything else identical: same recipe, batch, learning rate,
30 epochs, all-epoch dev selection, five dev-tied checkpoints scored on 21LA, analysis under
the trial-phase protocol. Runs are named `<arm>_s<seed>`; the existing runs are
`arm1` and `arm2` (seed 1234).

**Statistic:** K per run = median over its five tied checkpoints (as registered above).
Three values per arm.

**Registered readings (frozen):**
- **Seed-robust reduction** iff every arm-2 K median lies below every arm-1 K median
  (exact one-sided Mann–Whitney p = 1/20 with 3 vs 3) and the arm-2 spoof-side cost
  maximum stays below 1 pp on every seed.
- **Not resolved** iff the two sets of three medians overlap: the one-seed reading of
  2026-09-10 is then reported as one seed, not as an effect.
- **Reversed** iff every arm-2 median lies above every arm-1 median.

**Kill:** a run that diverges, or whose best dev EER exceeds the arm's seed-1234 value by
more than 1 pp, is reported as an infrastructure failure and excluded; the reading then
uses the seeds that completed and says so.

**Derived cost:** per run 10.2 GPU-h training + 2.2 h dev sweep + 0.8 h scoring
(calibration.md anchors; the 2026-09-09 arm-2 scoring ran three times slower under host
memory pressure, so the wall-clock may reach ~20 h per run). Four runs ≈ 53–80 h of
3090 wall-clock. Submitted at normal priority behind whatever the other lines hold.

## Amendment (2026-09-10 01:40, before any run): AASIST arms

EXP-123 predicts a larger transport cost, and therefore a larger training effect, where
the detector is weak on the calibration channel; AASIST is that detector in the paper.
Two runs with the official AASIST recipe (clovaai/aasist `AASIST.conf`: 100 epochs, batch
24, Adam with cosine schedule, weighted CCE, seed 1234), differing only in the training
set: `aasist_arm1_s1234` on the original 19LA train, `aasist_arm2_s1234` on the
channel-matched train set of arm 2 (`arm2_db`). Selection rule as for SSL-AASIST: dev EER
per epoch, five tied checkpoints, 21LA scoring, K under the trial-phase protocol. The
released AASIST checkpoint of the paper is not the baseline here: its training recipe
differs from ours in unknown ways, so both arms are trained.

**Registered reading (frozen):** as for the SSL arms, with three additions.
- AASIST arm 2's K range entirely below arm 1's, medians differing by more than the wider
  within-arm range: channel-matched training materially reduces transport failure on
  AASIST as well.
- Descriptive, not tested: whether the K reduction (arm 1 median minus arm 2 median) is
  larger for AASIST than the 14 observed for SSL-AASIST, as EXP-123 predicts, and whether
  AASIST arm 2's spoof-side cost maximum falls below 1 pp.
- Kill: divergence, or a best dev EER above 2% (the recipe reports 0.83% on 19LA eval;
  a dev EER above 2% means the recipe did not reproduce and the arm is an infrastructure
  failure).

**Derived cost:** no AASIST training anchor exists in calibration.md; AASIST inference
runs at ≥ 77 utts/s (EXP-001 job 63) and training is typically 2–4× slower per utterance,
so one epoch of 25,380 files is estimated at 5–17 min plus ~2 min of dev scoring; 100
epochs ≈ 12–32 h per run, two runs 24–64 h wall-clock. The first epoch's measured time is
recorded in the report and fed back to calibration.md. Submitted behind the seed factorial
at normal priority.

## Scope change (2026-09-11 evening, recorded by decision of the first author)

This experiment was registered and planned as next-version work. That is no longer its
status: if the seed factorial completes in time, the channel-matched training result goes
into the ICASSP 2027 submission, and the manuscript has been built so the two versions can
be compared. The runs are therefore on the submission path, contingent on completing by
14 or 15 September against a 16 September deadline. The readings and kill criteria above are
unchanged; only the scheduling priority is.

A single-seed number is not eligible for the submission: the arm-2 median is 16 at seed 1234
and 20 at seed 1235, so one seed is known not to represent the effect. The paragraph held in
`../../paper/A2/BLOCK-training.tex` carries placeholders until three seeds per arm exist.

## Post-result statistical-interpretation addendum (2026-09-13, after all six runs)

The frozen text above is preserved unchanged. This addendum records a correction to the
inferential interpretation only.

The registered run-level K statistic, development-only checkpoint selection,
complete-separation criterion and spoof-cost ceiling are retained. The observed results
satisfy the two numerical criteria. The registered exact one-sided Mann-Whitney
calculation gives p = 1/20 under unrestricted six-run label exchangeability, but that
assumption was not established for the common-seed design. We therefore withdraw its use
as a significance claim. Preserving seed pairs gives p = 1/8 under the stated within-pair
exchangeability model; this is a post-result conditional sensitivity calculation, not the
registered confirmatory test. The manuscript reports the experiment descriptively. This
correction changes the inferential interpretation, not the observations, exclusions or
registered numerical criteria.

The observed outcome is the registered complete separation, not the registered "not
resolved" category, which was defined by overlapping sets of run medians and did not
occur.

Raised as finding S1 of the exact-PDF audit of 2026-09-13 and adjudicated the same day.
