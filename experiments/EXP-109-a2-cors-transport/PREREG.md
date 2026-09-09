# EXP-109 — A2 extension: detector breadth and a second calibration source

Pre-registered **2026-08-21**, before any new scoring. Protocol:
[../../docs/experiment-protocol.md](../../docs/experiment-protocol.md). Campaign context:
[../../docs/a2-extension-campaign-20260820.md](../../docs/a2-extension-campaign-20260820.md).

**No score is computed until this file is frozen and hashed.** Nothing here selects a
headline; the headline is chosen after the evidence exists (CLAUDE.md rule 4).

## Phase-0 finding that changed the design

CodecFake+ CoRS does not have the structure the campaign plan assumed. In CoRS the codec
**is** the attack: `p225_001.wav` is bona fide and `p225_001_vocos_encodec_6.wav` is its
labelled spoof. The bona-fide pool is therefore **identical across all codec conditions**
(44,455 files total, not per condition), so CoRS cannot supply a bona-fide channel-shift
axis. A threshold set at a fixed FPR on that pool is the same threshold in every condition.

The channel-transport claim therefore stays where it is already earned — the ASVspoof 2021 LA
real-telephony conditions — and CoRS is used for the two things it can actually support:

1. an **attack-family** axis at a fixed operating point, on speakers disjoint from ASVspoof;
2. a **second, independent calibration source** (44,455 VCTK bona-fide recordings) to
   transport *from*, which is what answers the standing objection that 68 of A2's 84 cells
   draw on the same 2,636 recordings.

## Assets, verified 2026-08-21

- `Codecfake_plus_CoRS/` holds **266,730** files = 6 conditions × 44,455: bona fide,
  `vocos_encodec_6`, `vocos_encodec_12`, `sqcodec50dim9`, `spectralcodecs`,
  `FACodec_encodec-decoder-v2_16k`. Speakers: 44,057 VCTK `p*` plus 398 `s5_*`. Paired by
  construction — same speaker, same utterance, across all six.
- The remaining ~26 codec families are inside four 25 GB `.part*` XZ archives, unextracted.
  Not required by this experiment.
- Detectors: SSL-AASIST and AASIST (EXP-001 scores exist for the ASVspoof cells);
  XLS-R+SLS, XLSR-Mamba, XLSR-Conformer local and EER-verified 2026-08-14.

## Cells

**A. Detector breadth on existing A2 cells.** Score 21LA (181,566), 21DF, ITW and BRSpeech
with the three additional detectors, then rerun A2's existing 108-cell transport map per
detector. Derived cost: ~417 k utts × 3 detectors ÷ 64 utts/s = **5.4 GPU-h**.

**B. Fixed-threshold attack-family spread on CoRS.** Calibrate at α = 5% on VCTK bona fide;
report realized FNR per codec family, per detector. 266,730 × 5 detectors ÷ 64 utts/s =
**5.8 GPU-h**.

**C. Cross-source calibration transport.** Calibrate on the CoRS VCTK bona-fide cohort and
deploy on each existing A2 deployment cell, and the reverse. This is the new-recordings arm;
no new scoring beyond A and B.

## Registered readings, symmetric before outcomes

| Outcome | Reading |
|---|---|
| The 108-cell transport failure reproduces on all three new detectors | Detector-general; the $n{=}2$ scope limit is retired |
| It reproduces on some detectors and not others | Detector-conditioned, reported as such; the paper keeps the current scoped claim and names which architectures escape |
| It disappears on the new detectors | The current A2 result is architecture-specific. This is a **negative result about our own paper** and is reported as the headline limitation, not buried |
| CoRS FNR spread at fixed α is wide across families | Attack-family sensitivity at a fixed operating point, on new speakers |
| CoRS FNR spread is narrow | The FNR explosion in A2 is channel-driven rather than attack-driven — a sharper claim than the current one |
| Cross-source calibration transports well | Calibration cohorts are interchangeable across corpora; weakens the "recalibrate per channel" prescription and must be said |
| Cross-source calibration fails | The prescription strengthens and now rests on two independent recording pools |

## Kill criteria

- If the three new detectors cannot reproduce their published EERs on 21LA within the
  0.5-point tolerance used in EXP-001, that detector is dropped before any transport cell is
  computed, and the drop is recorded.
- If CoRS bona-fide scores show the resynthesis-contamination pattern found in
  [EXP-306](../EXP-306-areaI-pilot/REPORT.md) — bona fide scoring more spoof-like than its own
  codec versions — cell B is reported as a corpus-validity finding and its FNR reading is
  withdrawn.

## Label-convention declaration

CoRS labels codec-resynthesised speech `spoof`. Whether that is the right convention is an
open question in the field (*How to Label Resynthesized Audio*, ICASSP 2026, arXiv 2602.16343)
and our own EXP-306 found a 19LA-trained CM flags Vocos copy-synthesis of genuine audio at up
to 100%. **We adopt the corpus's native convention** and report the alternative reading
(resynthesis as bona fide) as a sensitivity row. Neither is selected after seeing outcomes.

## Separation from the WIFS 2026 submission

The WIFS 2026 submission uses the **CoSG** partition, source-holdout AUROC and rank
uncertainty; a text search finds **zero** occurrences of threshold, FPR, false alarm or
operating point in it. EXP-109 uses **CoRS**, a fixed operating point, and no source-holdout
training. The two are cited against each other; neither reuses the other's estimand.

## Prior art established in phase 0

- `2606.21584` — audits threshold transfer and unlabeled corrections; already A2's anchor.
- **`2608.17585` (new, 2026-08)** — *The Last Mile of Deepfake Speech Detection*, an
  industry–academia experience report. Argues A2's thesis from the deployment side (*"EER
  says nothing about the operating point a customer actually runs"*; *"a deployable detector
  must be calibrated, ideally recalibrated for the specific use case"*), with **zero**
  occurrences of `quantile` or `false accept` — it does not propose or evaluate a method.
  **Must-cite, not a scoop.** Postdates A2's 2026-08-14/15 sweeps.
- Speaker-verification calibration backends (Ferrer et al., `2002.03802`, `2102.01760`; and
  the 2004 channel-robust SV line) establish that robustness and calibration are distinct in
  the **parent field**. Consequence recorded below.

## Consequence for the proposed phase 4

The campaign plan proposed the headline *"robust training does not remove the need for
recalibration."* Phase 0 finds that claim is **folklore in speaker verification**, where
trained calibration backends have addressed exactly this separation for two decades. Under
the practitioner test it fails: an SV reviewer says "obviously." A training arm may still be
run, but its admissible claim is a **measurement** — how much of the transport failure
channel-matched training removes, in the bona-fide-only regime where an SV-style calibration
backend is unavailable because there are no spoof labels. Phase 4 is demoted to supporting
and is **not** part of this pre-registration.
