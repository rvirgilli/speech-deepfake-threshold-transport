# EXP-403 — does a bona-fide-only bound on spoof-side cost transfer across detector families?

> **SUPERSEDED 2026-08-15, before any scoring, by
> [../EXP-103-a5-replicate/PREREG.md](../EXP-103-a5-replicate/PREREG.md).**
> I argued the detector extension was A2's decisive experiment because the
> ceiling is "measurement, not instrument". An independent analysis showed the
> objection a reviewer actually writes is the *sampling* one — 68 of 84 cells
> are the same 2,636 recordings — and adding detector families multiplies cells
> over those same recordings, so it cannot touch it. That is right, and I was
> optimising the wrong axis.
>
> What survives: the cross-detector bound result below (37.7%/76.0% coverage,
> shape transfers and gain does not) is a real measurement and becomes a
> supporting observation in the paper. The transfer question moves to **corpora**
> instead of detectors, where it is both harder and free — see EXP-103 §2b.
> Kept substantively unedited: a superseded plan is evidence about how the line
> reasons. **Archival terminology clarification, 2026-08-29:** the two historical
> “codec” shorthands below were corrected to the verified 21LA taxonomy—six real
> transmission/channel conditions plus untransmitted `none`. The protocol,
> analysis and result are unchanged.

Pre-registered 2026-08-15, before any new scoring. Protocol:
[../../docs/experiment-protocol.md](../../docs/experiment-protocol.md).
Rationale and pricing: [../../paper/A2/STRONG-ACCEPT.md](../../paper/A2/STRONG-ACCEPT.md).

## Why this is the decisive experiment rather than more coverage

A2's stated limitation is that the bona-fide-only observable
$\log_2(\mathrm{FPR}/\alpha)$ ranks deployment cells by spoof-side cost at
ρ = 0.99/0.98 but cannot give a rate. If a conservative bound transferred, the
limitation would become an instrument.

Measured at n=2 (`bound_transfer.py`, CPU, existing scores): a monotone envelope
fitted on one detector covers **37.7%** and **76.0%** of the other's held-out
cells against a ~95% requirement. It fails — and the asymmetry is the mechanism:
at matched observable, AASIST's price is **1.4–2.4×** SSL-AASIST's, so the
*shape* transfers and the *gain* does not.

**Two explanations remain and n=2 cannot separate them:**

- **(a)** an idiosyncratic difference between this particular pair;
- **(b)** a per-detector scale factor, in which case "shape from the grid, gain
  from a small labelled probe" is a constructible bound and A2 has a method.

Separating them requires leave-one-out, which requires ≥4 families. No further
analysis of the existing grid can do it.

## Pre-committed disposition — both branches, before seeing either

Stated now so this is not a fishing expedition:

- **If a bound transfers at n=4:** it is A2's method contribution, and the
  framing changes from "the cost is invisible" to "here is how to bound it".
- **If it does not:** we submit on the strengthened negative. *"No
  detector-transferable bound exists across four architecture families,
  including a state-space model"* is a materially stronger claim than the
  current two-detector disclaimer, and **no further spend is authorised on this
  question** — I pre-commit to stopping rather than adding a fifth family.

## What runs

Score **XLS-R+SLS** and **XLSR-Mamba** (LA checkpoint) on the drift-map corpora:
`asv21la` (six real transmission/channel conditions plus untransmitted `none`;
the metadata field is historically named `codec`), `asv21df_100k`, `itw`,
`brspeech_test`, `asv19_dev`. Checkpoints and smoke-tested adapters already
exist in `~/projects/voxtech/deepfake-model-assessment`; **no training**.

XLSR-Mamba is the load-bearing addition: it is a state-space back-end, so the
"is this an AASIST artifact?" objection — fair against two detectors sharing a
back-end lineage — does not survive it.

Then `drift_map.py` over four families, then `bound_transfer.py` leave-one-out.

## The bar, with its detectability stated before freezing

**Primary:** a bound transfers iff, for **every** leave-one-out fold, the
envelope fitted on the other three families covers **≥95%** of the held-out
family's cells.

**Detectability — and the first draft of this bar was wrong.** The dependence
unit is not the cell. A detector's ~50 cells are orderings of one recording set
under seven 21LA channel/transmission conditions plus four corpora, so the honest
unit is the **condition**,
n≈11 per fold. At that n, `bar_check.py` says:

```
$ bar_check.py rate --effect 0.95 --n1 11 --baseline 0.80 --operational 0.90
  test         : exact binomial vs baseline 0.8 (Wilson CI)
  95% CI       : [0.6226, 0.9838]
  p at the bar : 0.3221
  min detectable rate at p<0.05: unreachable at this n
  VERDICT      : BELOW ITS OWN FLOOR
```

**Unreachable at any coverage** — even 11/11 gives p = 0.086 against an 80%
alternative. A 95% coverage bar read at the condition level cannot distinguish a
bound from a non-bound, which is the fourth bar tonight below its own floor and
the exact shape the sweep was looking for.

**So the bar is stated at the level where it is decidable, and both levels are
reported:**

- **Primary (decidable):** pooled across all four folds, coverage ≥95% of
  **cells**, n≈200. `bar_check.py rate --effect 0.95 --n1 50 --baseline 0.80`
  clears at p = 0.0013 per fold, and pooling only strengthens it. Cells are
  dependent, so this is a *descriptive* coverage rate over the grid we hold and
  is reported as such — not as an estimate of coverage on unseen detectors.
- **Secondary (reported, not decision-bearing):** per-fold coverage at the
  condition level, with its Wilson interval, precisely so the width is visible
  rather than hidden behind the pooled number.

**A bound that fails is not a null.** Coverage well below 95% with the
shape-transfers/gain-does-not signature is a positive finding about the
mechanism, and it is what the strengthened negative rests on.

## Dependence unit

The **detector family** for the transfer claim (n=4 — small, and every statement
about transferability carries that n explicitly). The **condition** within a
family for coverage intervals. The **cell** only for descriptive pooled rates,
never as an independent n.

## Cost

≈350k utterances per detector against the EXP-001 anchor (417k ≈ 3 GPU-h) →
**≈5 GPU-h total**, chained behind job 127 on the 3090 by queue dependency.
Analysis is CPU-minutes. Nothing writes to bigfour.

## What would make this worthless

- **Scoring a checkpoint trained on different data and calling it a family
  difference.** Both additions must be LA-trained, matching the existing two;
  the XLSR-Mamba DF checkpoint is *not* interchangeable and is not used.
- **Reading the pooled cell-level coverage as if cells were independent.** It is
  descriptive. The transfer claim's n is 4.
- **Adding a fifth family after seeing n=4 fail.** Pre-committed against above.
