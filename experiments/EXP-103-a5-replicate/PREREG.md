# EXP-103 — A2's drift map replicated on recording-disjoint data

Pre-registered 2026-08-15, before any scoring. Protocol:
[../../docs/experiment-protocol.md](../../docs/experiment-protocol.md).
Supersedes [../EXP-403-a2-detector-families/PREREG.md](../EXP-403-a2-detector-families/PREREG.md)
as A2's strong-accept path; see "Why this and not the detector extension".

## What it fixes

A2's most quotable weakness is written into its own abstract: 84 of the 108
cells are re-orderings of one 2,636-recording, 67-speaker set, so the dependence
is disclosed rather than removed. `ASVspoof5.eval.track_1.tsv` removes it.
**Verified here from the protocol, not relayed:**

| | 21LA (current) | A5 eval track 1 |
|---|---|---|
| conditions | 7 | **12** (`-`, C01–C11) |
| speakers | 67 | **737, present in every condition** |
| bona fide per condition | 2,636 — *the same recordings* | 8,976–9,710 |
| bona-fide utt IDs shared by all conditions | all | **0** |
| **source recordings appearing under >1 condition** | **100%** | **19.5%** (corrected below) |
| totals | — | 138,688 bona fide / 542,086 spoof |

> **Correction, 2026-08-15, made blind — no scores existed (job 127 still
> training, 129/130/131 queued).** The row above originally read "zero shared
> utterance IDs → genuinely disjoint recordings". **The premise was verified and
> the inference was wrong.** Output `utt_id`s are unique per (source, condition)
> pair by construction, so their disjointness is guaranteed and says nothing
> about recordings. The column that matters is the *source* utterance (field 5).
>
> Measured: **6,839 of 35,150 bona-fide sources (19.5%) appear under 11
> conditions each; the other 80.5% appear under exactly one.** So A5 is not
> twin-free — it is 19.5% twinned against 21LA's 100%.
>
> **I verified the axes of the design and inferred the cross.** Twelve
> conditions, 737 speakers in every condition, disjoint output IDs — all true,
> all load-bearing, and none of them establishes how the cross was built. The
> protocol sentence describing the cross is the one I did not read.

Codecs are organizer-applied: `opus_wb/nb`, `arm_wb/nb`, `speex_wb/nb`, `mp3`,
`m4a`, `encodec`, a chained `mp3_encodec`, and `MST`. The AMR conditions also
remove the ffmpeg encoder gap that blocks Area I's channel bank.

**C11/MST is a simulated mobile→PSTN transmission, not real device capture.**
The cost sweep described it as "real device capture over bluetooth and cable";
the TASLP paper's enumeration ends *"or the simulated effects of transmission
from a mobile device across a public switched telephone network"*, and its
Limitations section puts re-recording out of scope (resolved from the full text
by the M1 line). Two consequences for this experiment: **the replicate contains
no physical-capture condition and must not claim one**, and MST is the *same
channel family* as the 21LA `pstn` condition carrying A2's flagship cell — which
is useful, because it gives the replicate a like-for-like comparison on that
family, but it means MST is **not a novel condition** and must not be written as
one. Splitting the 737 speakers 50/50 by seeded
hash makes calibrate→deploy pairs **speaker-disjoint as well**, so the twin
hazard is eliminated by construction rather than disclosed.

## Why this and not the detector extension

I proposed adding XLS-R+SLS and XLSR-Mamba as A2's decisive experiment, on the
reasoning that the paper's ceiling is "measurement, not instrument". **An
independent analysis disagreed and it is right.** The objection a reviewer
actually writes is the sampling one — and the paper hands it to them
pre-written. Adding detector families multiplies cells over *the same 2,636
recordings*; it cannot touch that. My cross-detector bound test
(37.7%/76.0% coverage, `../EXP-403-a2-detector-families/bound_transfer.py`)
becomes a supporting observation, not the experiment.

The goal survives on a better axis: transfer tested across **corpora** rather
than detectors is both harder and more convincing, and it rides free on the
replicate.

## Pre-check, and its branch, frozen before scoring

The detectors are 19LA-trained; A5 carries unseen attacks on different source
speech. Score 1,500 bona fide + 1,500 spoof from the no-codec condition and
compute oracle FNR at 5% FPR, both detectors. ~2 min GPU.

- **If either detector's oracle FNR ≤ 50%:** proceed to 2a and 2b as written.
- **If both exceed 50%:** A5 is overlap-dominated, **the price axis dies, and
  the FPR-severity axis survives** — which is the headline's measurable half, so
  the replicate still runs regardless.

  **The branch, and what it actually costs.** The orchestrator advised recording
  that swapping in XLS-R+SLS or XLSR-Mamba is *free*, because M1's four-detector
  campaign would already have scored them. **That premise is false as of writing
  and is not recorded as true here.** The M1 line states it is under a standing
  freeze from RV and is escalating its A5 arm rather than running it, so those
  scores do not exist and may never. Written down because a pre-registration
  that inherits a stale cross-session assumption is worse than one that admits a
  cost.

  So the branch is, in order: (i) if RV authorises M1's arm, the stronger
  detectors are free and we take them; (ii) if not, scoring one additional
  detector for A2's own account is ~1–1.5 GPU-h and needs a portfolio decision,
  because it partly reinstates an extension that was superseded on merit;
  (iii) otherwise the ITW self-transcoded grid. **Only (i) is free.**

## Reading 2a — primary, restricted to the twin-free subset

Realized FPR at the transported threshold across the 132 ordered
calibrate→deploy condition pairs per detector, **computed only on source
recordings that appear under exactly one condition**, and speaker-disjoint by
seeded hash on top of that. Reported as signed $\log_2(\mathrm{FPR}/\alpha)$,
same estimator as EXP-102.

That subset is **63,459 bona-fide recordings: 2,137–2,871 per codec condition
and 35,149 in no-codec, with all 737 speakers present in every condition.** Per
condition it is comparable to 21LA's 2,636 — but where 21LA's are the *same*
2,636 recordings seven times, these are disjoint. **This is the design the
replicate needed, and it is now constructed rather than assumed.**

**The 19.5% crossed subset is not discarded — it becomes a second arm, and it is
worth more than the error cost.** The same 6,839 recordings under 11 conditions
is a *paired* grid with exactly 21LA's twin structure, on the same corpus,
scored by the same detectors. Running both arms measures **how much the twin
structure inflates severity**, which is the quantity A2 currently can only
disclose and never estimate. Pre-registered here as descriptive: with one corpus
the comparison is n=1 and no general correction factor is claimed from it.

**Pre-registered, and this is the sentence that must exist before the number:**
a *milder* collapse than 21LA is **not a failure**. 21LA's within-corpus cells
share recordings, which plausibly inflates severity. A milder but present effect
converts a disclosed n=1 into **"the mechanism replicates on recording- and
speaker-disjoint data; the rate is corpus-specific"** — which is a stronger
paper than the current one, not a weaker one. Only *absence* of the conservative
collapse refutes.

## Reading 2b — pre-registered secondary, zero marginal compute

Fit the severity→price map on 21LA (Spearman +0.992/+0.984, sign agreement
100%), apply it **entirely held out** to A5: different corpus, speakers, codecs,
unseen attacks. Report held-out $R^2$ against the **0.5** bar the paper already
uses for its W₁ fit.

If it transfers, A2 delivers a deployment-computable estimate of the quantity a
bona-fide-only deployment cannot measure — the gap the paper currently only
names.

**Detectability of the 0.5 bar.** $R^2$ here is over 132 pairs whose dependence
unit is the **condition** (12), not the pair. `bar_check.py corr --effect 0.71
--n1 12 --operational 0.5` — $R^2=0.5$ is $|\rho|\approx0.71$ — reports the
$\rho$ floor at this n. The bar is reported at both levels and the condition-level
interval is printed rather than hidden behind the pooled figure, because three
bars today were decided purely by which n was claimed.

## Dependence unit

The **condition** (12) for severity claims and the **speaker** (737, split
50/50) for within-condition intervals. The ordered pair is the unit of the
*design*, not of the evidence: 132 pairs come from 12 conditions and are not 132
independent observations. n_pairs is never reported as n.

## Scope: two detectors, not four

**Amended 2026-08-15 before scoring.** The orchestrator proposed scoring four
detectors so M1's clustered-inference arm could ride along. **The M1 line
declined, correctly**: it is under a standing freeze from RV to prose and CPU on
existing artifacts, and its A5 arm is a new experiment. Adding XLS-R+SLS and
XLSR-Mamba *on M1's account* would route that line's frozen work through this
session — the same shape as permission laundering even though nothing was
denied. M1 is escalating to RV directly.

So this run is **`ssl` and `aasist` only**, which is what A2 needs. If M1's arm
comes back authorised we decide then whether to extend or re-run; a superset
later is a bonus, not a plan this PREREG relies on. The emitter is unchanged and
already serves both.

## Cost

≈2–3 GPU-h for the two reproduction-scored detectors over the 80 GB local
`flac_E_eval/`, chained behind job 127. No download, no transcode, no DGX.
Analysis CPU-minutes.

## Page budget, which is the real constraint

The replicate needs a table and the paper is at exactly 5 pages. **Table 2 is
the cut**: the paper itself calls it a re-audit rather than a contribution and
concedes cohort z-norm is a category difference rather than a competitor. Three
rows or prose. Decided now so the result is not written into a paper with no
room for it.

## What would make this worthless

- **Quoting 132 pairs as n.** They come from 12 conditions.
- **Reading a milder replicate as a failure**, or a stronger one as vindication
  without checking whether the speaker-disjoint split changed the estimator's
  behaviour.
- **Fitting 2b's map on anything A5 has seen.** The map is fitted on 21LA only,
  and A5 is touched once.
- **Writing a like-for-like provenance sentence across the two grids.** A5's
  bona fide is single-source and carries no source-corpus column; 21DF's spans
  three source corpora. The organizers name the former as a limitation
  themselves — TASLP (arXiv:2601.03944) §D.3, verbatim via the M1 line, which
  holds the full text: *"The acquisition and reliance on a single corpus (e.g.,
  VCTK or MLS/LibriVox) for constructing bona fide speech samples has been a
  recurring criticism in the community."* Citable, and it is *why* A5's speakers
  are recording-disjoint — but it means the two grids cannot support the same
  provenance claim, and neither paper should elide that.

## Corroboration of an existing A2 sentence, from the M1 line

A2 §2 says ASVspoof~5 "reports its cross-dataset tables in EER alone". The M1
line verified at full text, across **both** A5 papers (workshop overview
2408.08739 and TASLP 2601.03944): `confidence interval`, `bootstrap`, `p-value`,
`standard error`, `Holm` and `Bonferroni` occur **zero times**, and the single
`significan` hit in each is ordinary English. The sentence stands and is now
independently sourced. A2 makes no claim that A5 reports significance, so
nothing needs changing — checked rather than assumed, after M1 flagged that
`ideas/M1/variants.md` carries exactly that error.
