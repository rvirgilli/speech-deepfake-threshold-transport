# EXP-102 — A2 full campaign (deployment measurement study)

> **Estimator-label amendment (2026-08-28):** cell 7's historical
> “weighted conformal (Tibshirani 2019)” label is not the estimator that ran.
> See `AMENDMENT-2-estimator-scope.md` for the exact clipped 15-bin
> mixture-score weighting heuristic and the oracle-label adaptive ceiling.
- **Line/idea:** A2 · **Priority:** core campaign (V1 per [variants](../../ideas/A2/variants.md); spec = [paper/A2/main.tex](../../paper/A2/main.tex) — every cell below fills a named table/figure)
- **Hypothesis (V1, margins below pilots):** the conformal bona-fide quantile pins realized FPR within ±2 pp of a 5% target from N=100 across the grid; FNR within 2 pts of oracle on viable corpora; the same-N cohort-norm baseline fails FPR control on the majority of cells; the guarantee's decay under drift is measurable and, on within-corpus channel drift, bounded.
- **Why now:** both load-bearing checks confirmed (EXP-002, EXP-010); fallback ladder validated (V2′/V3, 2026-08-14); spec written. This PREREG freezes the evidence loop before any further cell runs.

## Fixed cell list (evidence loop)

| # | Cell | Fills | Method | Budget |
|---|---|---|---|---|
| 1 | XLS-R+SLS scoring: 19LA dev + 21LA + 21DF-100k + ITW + BRSpeech-DF, scores dumped to EXP-001 format | Tables 1–2 row 3 | gpu-queue chained job; EXP-001 harness + public checkpoint | ~2 GPU-h (EXP-001 anchor: 64 utt/s SSL lane, ~450k utts); skipped without penalty if checkpoint unobtainable → grid stays 2 systems, stated limitation |
| 2 | Unlabeled C-methods: C1 z-norm, C4 CORAL, C5 AS-norm w/ unlabeled target cohort, on all (system, corpus) | Table 2 "unlabeled" row | reimplementation per arXiv 2606.21584 descriptions; CPU on existing scores | CPU <1 h (EXP-002 class); ~2 h |
| 3 | N-sweep: quantile + cohort z-norm, N ∈ {30, 100, 300, 1000, 3000, 10000}, B=1000, both FPR tightness and FNR gap, vs Beta-law prediction | Fig. 2 (cost-of-labels) | extends EXP-010 harness | CPU <1 h |
| 4 | Contamination: 1/2/5% spoofs injected into calibration set, N ∈ {100, 500}, realized-FPR excursion | Fig. 2 panel | extends EXP-010 harness; cite Bashari et al. for theory, claim measurement only | CPU <1 h |
| 5 | Drift map — within-corpus channel axis: calibrate on one 21LA channel/transmission condition's bona fide (N=500), deploy on each other condition; all 7×6 ordered pairs, both/3 systems | Fig. 1 | 21LA `keys/LA/CM/trial_metadata.txt` col-3 condition labels (historical field name: `codec`); six real transmissions plus untransmitted `none` | CPU ~1 h |
| 6 | Drift map — cross-corpus bundle + language (n=1, flagged): calibrate on corpus A bona fide, deploy on corpus B, all ordered pairs of {21LA-nocodec, 21DF-100k, ITW, BRSpeech} | Fig. 1 | same harness | CPU <1 h |
| 7 | Drift baselines in every drift cell: weighted conformal (Tibshirani 2019; weights from a label-free density-ratio estimate on scores) + adaptive conformal (Gibbs & Candès 2021, online on the deployment stream) | Fig. 1 / §5 | required by V2′ panel: the claim under breakage must be "label-free repairs don't restore control" | CPU ~2 h; ~3 h |
| 8 | Excursion-vs-distance curve: realized-FPR excursion vs oracle W1 (calibration-vs-deployment bona-fide scores), fitted on 2 systems, **predicted on the held-out third** (transfer validation); reported R² | Fig. 1 | oracle diagnostic, labelled as such | CPU minutes |
| 9 | Unlabeled monitoring statistics (co-primary): (a) W1 between deployment *mixture* score distribution and calibration-time reference mixture; (b) CDTS-style batch predictive-entropy. Success criterion per statistic: flags cells with excursion > 2 pp at TPR ≥ 0.8, FPR ≤ 0.2 across drift cells, AND Spearman(monitor, excursion) ≥ 0.6. Attack-prevalence confound probed by re-mixing spoof share ±50% in deployment stream | Fig. 1 / §5 | label-free by construction | CPU ~1 h |

Total: ≤2 GPU-h (cell 1 only, skippable), ~6 h CPU, ~10 h. **Contingency budget (≤20%):** α=1% target replication, extra N values, per-codec FNR table, monitor sensitivity sweeps — spent only on mock-review demands, never on new claims.

## Pre-registered decision tree (outcome regions → variant)

Decision cells evaluated at N=500, primary system SSL-AASIST, α=5%; excursion := |realized FPR − 5%|.

1. **V1 stands** iff: quantile within ±2 pp on all main-grid cells (Tables 1–2) AND on the within-corpus channel-drift cells (cell 5) the excursion ≤ 5 pp on > half of the 42 ordered pairs. Cross-corpus/language breakage does not dethrone V1 — it is Fig. 1's expected gradient, reported honestly.
2. **Switch to V2′** ("drift breaks distribution-free thresholds") iff: excursion > 5 pp on ≥ half of the within-corpus channel-drift pairs (the drift no deployment can avoid) AND at least one positive handle exists: a monitoring statistic meets its cell-9 success criterion OR the excursion–W1 curve transfers (held-out system R² ≥ 0.5). Supporting cells are a subset of this campaign (cells 5–9) — no new experiments.
3. **Park (no paper → Interspeech/workshop V3 assets only)** iff: within-corpus breakage per (2) AND no positive handle (both monitors fail their criteria AND transfer R² < 0.5). Pre-priced kill; no improvised rescue.
4. **Switch to V3** (measurement-study re-billing, ASVspoof workshop/Odyssey per its panel constraints) iff: V1's main grid holds but the drift story is too mixed for region 1 or 2 (e.g., breakage on exactly the boundary, or monitors partially succeed) by the draft-review date (~Sep 3). V3's mandatory artifacts (N-sweep, contamination, resource-class grid) are cells 2–4 — already in this campaign.

One reframe maximum (methodology Step 3); tree edits after results exist are prohibited.

## Kill criteria (per protocol — campaign-level)
- Quantile excursion > 2 pp at N=500 on ≥ 2 main-grid corpora (contradicting both pilots) → campaign halted, pilots re-audited before any writing continues.
- Cell 1 checkpoint sanity: XLS-R+SLS must reproduce its published 21DF EER within 1.0 absolute pt, else its rows are dropped (sourcing rule), not debugged past 2 h.

- **Budget:** ≤2 GPU-h + ~6 h CPU + ~10 h; wall-clock ≤3 days including writing integration. Anchors: EXP-001 (scoring throughput), EXP-002/EXP-010 (CPU cell class), EXP-009 (sim class).

## Addendum 2026-08-14 (spec-gate D1/D6)

- **D1 contingency activated (pre-priced):** same-N parametric quantile baseline — Gaussian fit to the N cohort bona-fide scores, threshold t = mean − 1.645·sd (α=5% lower tail) — added as a Table 2 row and N-sweep trace, isolating distribution-freeness from the renormalization-vs-threshold confound. CPU minutes on existing scores; same paired draws as the other policies.
- **D6:** cell 1's GPU remainder (XLS-R+SLS on 19LA dev + BRSpeech-DF for the naive-transfer row and language cell) is now demanded by two reviewers and executes via gpu-queue; fallback on checkpoint-sanity failure stays as pre-registered (sourcing footnote + limitation).
- **Rejected at the gate (decided once):** t-DCF/a-DCF re-scoring of the grid and actually-running CORAL (D11) — new resource class; served by discussion prose + the printed C4-omission reason.
