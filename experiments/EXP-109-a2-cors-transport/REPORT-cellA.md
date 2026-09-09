# EXP-109 cell A — report: the transport failure reproduces on three further detectors

> **Superseded on 2026-09-09 by [AMENDMENT-2](AMENDMENT-2-trial-phase.md):** the cells below
> include the ASVspoof 2021 hidden (VAD-trimmed) phase. Under the corrected protocol the
> FPR-axis counts are 33 / 25 / 30 of 42 (88 of 126) and the flagship spoof-side cost does
> **not** reproduce on these detectors. Numbers below are kept for the record.

Date: **2026-08-21**, revised after the independent gate-2 audit (verdict **FAIL**,
`reproduced=yes`, 4 material findings). PREREG `a7a8dbec…`;
[AMENDMENT-1](AMENDMENT-1-detector-provenance.md) restated the cells after the phase-0
finding that the three added detectors exist only as released score files.

- **Status: descriptive.** The numerical map is valid and independently reproduced. It is
  **not** an auditable confirmatory verdict — see "What this is not" below.
- **Budget: registered 5.4 GPU-h → actual 0 GPU-h and 51.6 s of CPU.** The estimate assumed
  runnable checkpoints; released scores make the map a join, not a scoring campaign.

## Gate

All three detectors reproduced the EERs recorded in the official-scores README on the
eval-phase subset — `xlsr_sls` 2.868 (2.868), `xlsr_mamba` 0.931 (0.931), `xlsr_conformer`
1.378 (1.378). None was dropped. Transport cells use all 181,566 trials, matching EXP-102's
existing map; only the gate uses the 148,176-trial eval subset, matching the README. The
auditor confirmed the split is implemented as claimed and that all-trial EERs differ
materially (7.717 / 6.379 / 5.934%), so the eval filter genuinely fired.

## Result: 126 cells

Six of the seven 21LA conditions are real transmission — `alaw, ulaw, gsm, g722, opus`
through an Asterisk PBX with one utterance per VoIP channel, and `pstn` over a real Spanish
carrier. `none` is the untransmitted source. Each detector gives 42 ordered pairs.

| Detector | Cells | Miss target by >2× | Conservative | Liberal | Worst uncensored \|log2\| |
|---|---:|---:|---:|---:|---:|
| XLS-R+SLS | 42 | 17 | 10 | 7 | 4.588 |
| XLSR-Mamba | 42 | 14 | 10 | 4 | 2.975 |
| XLSR-Conformer | 42 | 21 | 16 | 5 | 3.735 |
| **Total** | **126** | **52 (41%)** | **36** | **16** | |

A2's existing map is 58 of 108 (54%).

**Censoring.** Four XLS-R+SLS cells — `gsm→{none, g722, opus, alaw}` — are
`resolution_limited`: their realized FPR falls below the `5/2636` resolution boundary, so the
estimator's own rule (`drift_map.py:143`) is that their severities must not be read as point
values. Three of the four have severity fields floor-clamped by the `3/2636` guard; the
fourth (`gsm→alaw`, FPR 0.001469) sits above that floor and is not clamped. Their point
severities are therefore **not reported here** — they are neither a resolved ranking nor an
equal tie. The column above gives the worst **uncensored** value per detector, and the
off-target counts are unaffected, since every one of these cells misses target by a wide
margin under any censor-aware reading.

An earlier revision of this report described all four as "at the floor of `3/2636`" and as a
"censored tie", and then printed their severities anyway. Both statements were wrong; the
independent fresh-perspective audit caught them.

## A2's flagship cell reproduces

Calibrate on real PSTN, deploy on real VoIP-over-Asterisk G.722:

| Detector | Realized FPR | FNR | Oracle FNR | FNR ÷ oracle | log2(FPR/α) |
|---|---:|---:|---:|---:|---:|
| XLS-R+SLS | 0.35% | 36.4% | 8.5% | **4.30×** | −3.83 |
| XLSR-Conformer | 0.74% | 27.4% | 7.0% | **3.89×** | −2.76 |
| XLSR-Mamba | 1.63% | 20.1% | 8.7% | **2.32×** | −1.62 |

The false-alarm rate reads better than the 5% target on every one of them while the
missed-spoof rate runs 2.3–4.3× its deployment-calibrated value. (An earlier revision said
"three to four times" for every detector; the exact ratios are above and Mamba is 2.32×.)

## GSM: a descriptive extremum, materially weaker than first reported

For the selected variants, `gsm` is the worst calibration source for each detector: five of
six GSM-source cells miss target for every detector, and its mean absolute severity across
its six deployments is the highest of any source. Because four SLS GSM cells are
resolution-limited, that mean is reported for SLS as a **censor-aware lower bound of >4.07**
rather than a point value; the next-worst SLS source, `pstn`, is at 2.709. Mamba (1.779) and
Conformer (2.166) have no censored cells. Exact integration by the auditor preserves the
ranking, so it is not a Monte Carlo seed artifact.

**Three qualifications, all from the audit, all load-bearing:**

1. **It is variant-dependent.** With the XLSR-Conformer `eval_lv` variant instead of the
   frozen fixed-size one, the worst source becomes **`pstn`**, not `gsm`.
2. **The detectors are not independent.** Their score Pearson correlations are 0.899–0.971
   and their 42-cell severity Spearman correlations 0.966–0.988. "Independently" was wrong;
   this is descriptive agreement among highly correlated released-score systems.
3. **It is post-hoc.** PREREG lines 7–8 state the headline is selected after the evidence
   exists. An earlier revision of this report said GSM was "not selected after the fact";
   the defensible wording is a **post-hoc descriptive extremum of a pre-specified sweep**.

**Consequence: this finding is not carried into the paper.**

## What this is not

The PREREG registered a reading for "reproduces on all three new detectors", but no
detector-general verdict check exists in the code, and the branch was assigned after the
counts were observed. This run is therefore **descriptive**, not a confirmatory verdict.
Establishing that status requires a new prospective experiment. The scope limitation it
addresses — A2's "$n{=}2$ detectors" — is answered descriptively, which is the same
epistemic footing as everything else A2 reports.

## Independent audit

An independent read-only auditor (gate 2, `INDEPENDENT-AGENT-AUDIT-PROTOCOL.md`) wrote its
own probe importing no repository code, integrating exactly over the hypergeometric rank
distribution of the 25th order statistic rather than sampling. It reproduced 52 (36/16) and
every per-detector count, with **zero >2× classification disagreements**. It authenticated
all three score files byte-identical to their upstream commits (SLS `89a09ac`, Mamba
`10ec738`, Conformer `e8e1959`) and the key against the official ASVspoof package.

A second, fresh-perspective auditor then verified the corrections and returned **REVISE,
2 closed / 2 open**, catching that the first censoring correction was itself factually wrong
and that `drift_map.py` — which supplies the actual cell computation — was unhashed. Both are
now fixed: `drift_map.py` (`1c378f76…`) is bound in the result and the freeze manifest, and
the censoring paragraph above is rewritten. Its four material findings are addressed here: the confirmatory-verdict claim is withdrawn
(above), input hashes are now bound in `results_cellA.json`, `FREEZE.sha256` now covers the
amendment and the code, and censored severities are no longer printed as extrema. The
re-run after adding input binding reproduced `cells` and `sanity` **bit-identically**.

The former `oracle_fpr_check` field was tautological: the oracle threshold was built from
deployment bona-fide scores and checked on those same scores. It is now named
`same_sample_quantile_identity` and documented only as a quantile/tie-convention record, never
as validation evidence. **Provenance debt closed 2026-08-28:** the score README now pins the
three authenticated snapshots by full upstream commit SHA (SLS
`89a09ac4404c5687d96d6c123fcf6db20e4e4b38`, Mamba
`10ec73810d24091028a2aa5454c8dfd1a239e441`, Conformer
`e8e195938b898d21ac0105076f883b1571db2664`) and gives immutable tree URLs.

## Scope

These are **released author scores**, a separate provenance stratum from the EXP-001
reproductions; no claim pools the two, and the auditor confirmed no pooling occurs in code.
Nothing here is population inference: the cells are outputs of one declared calibration
procedure on a fixed released roster.
