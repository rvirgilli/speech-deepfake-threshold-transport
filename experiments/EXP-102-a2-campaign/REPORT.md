# EXP-102 — Report (campaign)
- **Status:** confirmed (vs pre-registered decision tree) — **Region 1: V1 stands.** No reframe used.
- **Budget:** estimated ≤2 GPU-h + ~6 h CPU + ~10 agent-h → actual **0.30 GPU-h** (17.7 min for the completed SLS row), ~25 min CPU across the remaining cells, ~5 agent-h. CPU cell class again ran ~10× under budget (anchor appended to calibration.md).
- **What ran:** `n_sweep.py` (cells 3–4), `drift_map.py` (cells 5–9), `c_methods.py` (cell 2), and `a2_sls_complete.py` (cell 1 complete; `sls_official.py` is its historical partial precursor); deepfake-model-assessment uv env, seeds per script; EXP-001 scores + official and locally reproduced XLS-R+SLS scores + 21LA channel-condition metadata. Logs and `results_*.json` in this dir.
- **Results (by cell):**
  - **Cell 3 (N-sweep):** finite-pool held-out realized FPR approaches the iid-continuous Beta reference as N and the pool grow (e.g. ITW: 3.28% vs the 3.23% reference mean at N=30; 4.96% at N=100; 5.0% at N≥300). The Beta law is not claimed as the exact sampling law of the without-replacement finite-pool experiment. Across the eight cells, the largest positive FNR−oracle gap is 2.41 points at N=100 and 0.48 points at N=1000 (and no larger thereafter). Cohort z-norm is flat-wrong at every N (12–24% FPR at N=10,000) — labels don't rescue the non-distribution-free policy.
  - **Cell 4 (contamination):** conservative failure mode on non-overlap cells, as theory predicts — 5% spoof contamination drags FPR down (ITW: 1.5%; FNR up); BRSpeech (overlap regime) nearly inert, marginally above target (5.11–5.19%) — the paper's 'conservative' claim is scoped to non-overlap cells accordingly.
  - **Cell 2 (unlabeled C-methods):** catastrophic at the pinned-FPR operating point: C1 z-norm 21–100% FPR, C2 temp/shift 8–100% (both saturate at 100% on ITW), C5 AS-norm (k=100, 5k cohort) collapses the operating point entirely (FPR ≈ 0, FNR ≈ 99%). Reproduces arXiv 2606.21584's "no unlabeled fix" at our operating point with our checkpoints. *Deviation:* C4 (CORAL) requires re-scoring through the frozen head and was replaced by C2 from the same anchor list; noted for the paper.
  - **Cells 5–6 (drift map):** within-corpus channel drift (six real 21LA transmissions plus the untransmitted source, 42 ordered pairs). SSL-AASIST (the pre-registered decision system): median excursion 2.8 pp, ≤5 pp on **37/42** pairs (worst: none→gsm 5.7 pp). **Detector-dependence (figure-stage finding, added to the paper):** AASIST is far more fragile within-corpus — 11/42 pairs >5 pp, 8/42 >10 pp, max 37.9 pp, all worst cells deploying onto PSTN — so "graceful decay" is a property of the detector's bona-fide score stability, not of the quantile policy itself. Cross-corpus: deployments *onto* BRSpeech-DF break hard (21–24 pp); other cross-corpus cells ≤5 pp for SSL. The decision-tree verdict is unaffected (Region-1 rule was pre-registered on SSL-AASIST), and the detector-dependence strengthens the paper's recalibrate-per-channel prescription.
  - **Cell 7 (heuristic and ceiling):** the clipped 15-bin mixture-score weighting heuristic frequently *worsens* control (up to 28.51% FPR). The outcome-adaptive ceiling uses oracle bona-fide labels and still leaves 7–8 pp excursion on the worst channel cells. Neither is attributed to the cited weighted-conformal construction; see `AMENDMENT-2-estimator-scope.md`.
  - **Cell 8 (excursion vs W1):** correlation without a transferable law — Spearman 0.64 (SSL) / 0.48 (AASIST); cross-system transfer R² = 0.27 / −0.04 (fails the ≥0.5 bar). The paper reports the correlation and does **not** claim excursion is predictable from W1.
  - **Cell 9 (monitors):** mixture-W1 monitor meets both pre-registered criteria on SSL (Spearman 0.64; TPR 0.92 @ FPR 0.19 for flagging >2 pp cells) but fails on AASIST (0.48; no operating point); entropy monitor fails everywhere (Spearman ≤0.07). **Draft-gate amendment (analysis of already-run probe):** the pre-registered ±50% attack-prevalence re-mix shows the mixture-W1 operating point does not survive prevalence shift — every benign cell false-flags under re-mix on both detectors (16/16 SSL, 10/10 AASIST) — and the threshold was selected in-sample. The paper reports unlabeled drift monitoring as an open problem, not a partial success.
  - **Cell 1 (third system, COMPLETE — D6 executed after two spec reviewers demanded it):** gpu-queue job 86 (17.7 min GPU; sanity: per-trial corr 0.9945 vs official 21LA release, kill bar >0.99 passed). Full SLS grid: naive transfer 44.7/58.4/99.1/99.5% FPR (motivation replicated on a third, SOTA system); quantile 4.95–5.02% on all four corpora with FNR−oracle ≤0.43 pts; cohort z-norm off by 12.6–50.8 pp; parametric quantile off by 4.4–9.5 pp (collapses to 0 on BRSpeech). BRSpeech is overlap-dominated for SLS too (oracle FNR 91.1%) — the language-shift regime holds across all three detectors. Scores: `scores/sls_{asv19_dev,brspeech_test}.csv.gz` + official releases; `results_sls_complete.json`.
- **Verdict rationale:** Decision tree region 1 requires quantile within ±2 pp on the main grid (met: 5.0% everywhere incl. third system) AND within-corpus drift excursion ≤5 pp on > half the 42 pairs (met: 37/42). Regions 2–4 not triggered; halt guard not triggered. **A2 proceeds as V1** with the drift map as an honest graceful-decay + failure-regime figure and monitoring as a supporting, explicitly system-dependent analysis.
- **Deviations:** C4→C2 substitution (above). Third-system rows partially sourced from official releases instead of local scoring (strictly better provenance). ITW official-score utt-ids required a `.wav`-suffix strip (join fix, no analytic impact).
- **Artifacts:** `results_nsweep.json`, `results_drift.json`, `results_cmethods.json`, `results_sls_complete.json` (`results_sls_official.json` is the historical partial predecessor), logs (this dir).
- **Consequence:** A2's registered evidence loop is complete, including cell 1. Writing proceeds on V1: fill paper/A2 tables from these JSONs; spec mock-review per calendar (~Aug 18). V2′ stays dormant (its trigger region did not fire); no reframe consumed.

---

## Addendum — 2026-08-14: cells 5–9 superseded by a metric correction (Gate-8 audit)

**Everything above about cells 5, 6, 8 and 9 is superseded.** Cells 1–4 and 7 are
unaffected. The `results_drift.json` in this directory has been regenerated; the
released numbers quoted above are preserved in git at commit `2b7f36b^`.

**What was wrong.** The reported `excursion = |FPR − α|` cannot exceed α on the
conservative side, so the pass rule `excursion ≤ 5 pp` is *vacuous* there: no
amount of conservative collapse can fail it. The spoof-side quantity that would
have exposed the collapse — the deployment FNR at the same threshold — was never
computed. Cells 8 and 9 inherited the defect, cell 9 most damagingly: collapsed
cells entered the monitor ROC as true negatives.

**What replaces it.** Severity is now `log2(FPR/α)` — scale-free, two-sided,
censored at the rule-of-three bound `3/n` with a `resolution_limited` flag for
the 5 cells whose FPR rests on fewer than five deployment utterances. The
spoof-side cost is now `fnr_price = FNR_dep(t_conformal) − FNR_dep(t_oracle,α)`,
referenced to the **oracle** deployment threshold. A first attempt referenced the
calibration condition's FNR instead; the audit showed that conflates
miscalibration with the intrinsic difficulty gap between calibration and
deployment attacks (Spearman 0.46 against the intended quantity, sign-flipped on
four of six headline rows) and it has been demoted to `fnr_shift_diagnostic`. The
self-check now encodes that counterexample as a regression test.

**Corrected findings.**
- 34 of the 84 cells the old rule admitted miss the FPR target by more than a
  factor of two, **all 34 conservatively**. Robust in direction across every bar
  tested (60/84 at 1.5×, 22/84 at 4×); the *counts* are post-hoc and not quotable
  as stable numbers.
- All ten costliest cells on the spoof side are cells the old rule called
  graceful. Worst: `aasist/pstn→none`, FNR 0.181 → 0.731 against its oracle
  threshold; `ssl/brspeech_test→asv21la_nocodec`, price +0.555 — the single worst
  cell in the grid, and one the demoted diagnostic scored as −0.276.
- Cell 8, **pre-registered reading** (excursion): held-out R² 0.265 / −0.039.
  **Corrected** (severity): 0.484 / −0.015. Both miss the pre-registered 0.5 bar,
  so the region-2 decision is unchanged.
- Cell 9, **pre-registered reading**: SSL mixture-W₁ Spearman 0.636 with an
  operating point at TPR 0.921 / FPR 0.188 — a partial pass. **Corrected**:
  Spearman 0.578 and **no operating point on either detector**. Changing the flag
  definition after seeing results is a deviation from PREREG cell 9, so both
  readings are computed and stored (`monitor_eval` and
  `monitor_eval_PREREGISTERED`) and the paper must print both. The correction is
  self-penalising, and the REPORT above had already retracted the monitor claim
  on independent grounds (in-sample threshold, prevalence probe).
- New: the spoof-side price is **uncorrelated with the bona-fide W₁** on both
  detectors (|ρ| < 0.01). Distributional distance on the bona-fide side predicts
  the FPR miss but says nothing about what that miss costs.

**Historical status at 2026-08-14; superseded and closed for the 2026-08-28 A2
submission build.** The paragraph below is retained as an audit trail, not a current task
list. The rebuilt manuscript attaches no interval or significance claim to the reused 21LA
views, states that the seven conditions reuse the same recordings, and reports the
recording- and speaker-disjoint ASVspoof~5 replication. The finite-pool/iid distinction is
also explicit in the manuscript and `n_sweep.py` output.

**Items recorded as open on 2026-08-14** (then carried into the A2 rebuild):
`fnr_price` and the severity have no intervals; `vanilla_fpr_ci` covers
calibration-cohort subsampling only, on a fixed deployment set, with
finite-population shrinkage up to ~21%; and — the finding with the largest
consequence for the rebuild — the seven 21LA channel conditions contain the **same 2,636
source recordings from the same speakers**: six are real transmissions and `none` is
untransmitted. The 42 within-corpus cells are therefore 42 views of one utterance set,
not 42 independent measurements of channel transport.
No interval or "significance" claim may be attached to them. This is direct
evidence for the rebuild's speaker/channel/temporally-disjoint split design.

---

## Addendum 2 — 2026-08-14: the price-vs-W₁ null does not mean what it looks like

Computed before any prose, on instruction, because a null with no interval
cannot lead a paper. `price_w1_correlation.py` → `results_price_w1.json`.

**The interval is tight.** Spearman(oracle bona-fide W₁, signed spoof-side
price), pooled over both detectors: **−0.005**, delete-one-condition jackknife
95% interval **[−0.033, +0.023]**, condition-cluster bootstrap [−0.045, +0.052].
Per detector: SSL +0.009, AASIST −0.007. The most conservative slice
(within-corpus, one detector, 7 conditions) still sits at ±0.15–0.25. Resampling
cells iid gives a *wider* interval (±0.23–0.40) than clustering does, because
the condition-cluster scheme duplicates cells and inflates the effective sample
— reported, and the jackknife is preferred for that reason.

**And the instrument has power on these exact cells:** Spearman(W₁, severity)
= +0.523 (SSL) / +0.883 (AASIST). This is not a null from a dead instrument.

**But the null is an artifact of correlating an unsigned quantity with a signed
one, and the substantive claim it was going to support is false.**

| Relation (same 54 cells per detector) | SSL | AASIST |
|---|---|---|
| W₁ vs **signed** price | +0.009 | −0.007 |
| W₁ vs **\|price\|** | **+0.539** | **+0.850** |
| **signed** $-\log_2(\mathrm{FPR}/\alpha)$ vs signed price | **+0.992** | **+0.984** |
| $\lvert\log_2\rvert$ vs \|price\| | +0.954 | +0.934 |

W₁ is a distance and cannot carry a sign, so its near-zero correlation with a
signed cost is close to uninformative. The observable a deployment actually has
— the *signed* log-ratio of realized to target FPR — predicts the signed
spoof-side price at Spearman ≈ 0.99, and the sign agreement is total
(conservative cells have a positive price in 100% of cases, liberal cells a
negative one in 96–100%).

**Consequences, all against our own interests.**

1. **"The observable carries no information about the cost" is the opposite of
   what the data say.** Leading with it would have been refuted by one line of a
   reviewer's Python. It must not be written.
2. **It also weakens the scope refinement adopted earlier today.** A deployment
   holding only bona-fide labels can rank its cells by cost almost perfectly.
   What it still cannot do is convert that rank into an absolute missed-spoof
   rate, because the map from FPR miss to FNR cost was fitted here *using* spoof
   labels and is detector- and corpus-dependent. The honest form of B1 is
   therefore "the ordering is observable, the units are not", which is a much
   weaker limitation than the one now in the paper.
3. **The price is largely redundant with the severity** (rank correlation 0.93–0.95
   on magnitude, 0.98–0.99 signed). Both are readouts of the same underlying
   quantity, threshold displacement: FPR rises and FNR falls monotonically in the
   threshold, so a cell that misses the FPR target badly must miss on the spoof
   side too. The price's value is that it states the miss in operational units,
   not that it is independent evidence. The paper should say this rather than
   let a reader infer two findings where there is one.

Nothing had been written into the paper on the basis of this at the time; the computation's
conclusion had not yet been adopted.

**Status 2026-08-28: superseded.** The later rebuild adopted the licensed conclusion:
observable bona-fide-side severity is strongly associated with spoof-side price on these
cells but does not identify an absolute missed-spoof rate without spoof labels. The paper
does not market severity and price as independent findings.

---

## Addendum 3 — 2026-08-14: does calibration fragility dissociate from discrimination?

Pre-specified check with pre-committed dispositions, run to decide whether
`main.tex`'s "bona-fide score stability, not the policy, is what fails" is
load-bearing or deletable. `dissociation.py` → `results_dissociation.json`.
Per detector throughout, because "SSL transfers better than non-SSL" is folklore
and a detector *ranking* would be a null for this purpose.

**Outcome: the dissociation is partial, and does not hold in the form that could
lead a paper.**

| | SSL-AASIST | AASIST |
|---|---|---|
| Spearman(\|ΔAUC\|, severity) | **+0.519** | **+0.536** |
| Positive control: Spearman(\|Δ median bona-fide score\|, severity) | +0.538 | +0.880 |
| Cells with discrimination essentially unchanged (\|ΔAUC\| < 0.02) | 44/54 | 24/54 |
| …of those, operating point blown (\|log₂\| > 1) | 17 (0.39) | 7 (0.29) |
| Jackknife 95% on that fraction (unit: condition, n=11) | [0.08, 0.70] | [0.00, 0.75] |

Reading it honestly:

- **Discrimination monitoring is not blind to this failure.** ΔAUC and the
  operating-point miss correlate at ρ ≈ 0.52–0.54. The "everyone is watching the
  wrong number" headline is therefore not available.
- **The rate is not estimable.** With 11 conditions the fraction of
  discrimination-stable cells whose threshold is blown spans [0.08, 0.70]. No
  number may be quoted.
- **Existence is demonstrated and is striking.** `ssl/gsm→none`: AUC *rises*
  0.977 → 0.982 while realized FPR is 44× off target. `aasist/ulaw→none`: ΔAUC
  exactly 0.000, |log₂| 1.41. So the two properties can come apart completely in
  individual cells even though they correlate across cells.
- **The sentence under test is supported but near-mechanical.** Score-location
  shift predicts the miss (ρ 0.54 / 0.88), and the policy is constant across
  cells, so "score stability, not the policy" is true — but a transported
  threshold is wrong precisely when the distribution under it moves, which makes
  this an explanation rather than a finding. It has been rewritten in the caption
  to state both measured correlations and one dissociating cell, with no
  implication that discrimination monitoring would miss the failure.

**Disposition applied** (pre-committed by the coordinator before the check ran):
the dissociation does not hold, so A2 ships as a measurement study of how badly
and how silently calibration transfer fails across 108 cells, under a criterion
adopted from NIST actDCF / ASVspoof 5 / NISTIR 8280 rather than proposed.

Two corrections also applied today, both against our own interest and both found
by us: the abstract's claim that the same absence "causes the blind spot and
limits the repair" was overstated — the signed miss ranks the cost at ρ ≈ 0.99,
so the asymmetry is in *units*, not information — and the price/severity
redundancy (ρ 0.95 / 0.93 on magnitude) is now stated in the paper so no reader
counts two findings where there is one.
