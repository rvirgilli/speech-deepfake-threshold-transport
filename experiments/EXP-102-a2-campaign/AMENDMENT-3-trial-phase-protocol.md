# Amendment 3 — ASVspoof 2021 trial-phase protocol (post-hoc correction)

Date: 2026-09-09. Applies to every A2 analysis that reads the reproduction-scored
21LA and 21DF score tables (EXP-002, EXP-010, EXP-102, EXP-103 cost map, EXP-403)
and to the XLS-R+SLS official-score rows.

## What was wrong

The campaign loaded every trial in the released ASVspoof 2021 LA and DF keys.
Those keys carry three phases: `eval`, `progress` and `hidden`. The hidden
subset is the same material with all non-speech intervals removed by a voice
activity detector; the organisers report that hidden subsets "were not used in
deriving challenge results or rankings" (Liu et al., IEEE/ACM TASLP 2023,
Sec. IV-A). Each 21LA condition therefore mixed 2,356 untrimmed bona-fide
recordings with 280 trimmed ones (10.6%); the 21DF 100k sample held 308 trimmed
bona fide of 3,738. Trimmed speech has a different score distribution: on 21LA,
SSL-AASIST's EER is 0.84% on `eval` and 15.05% on `hidden`; AASIST's 8.15% and
35.31% (EXP-001 score tables joined to the keys).

`PREREG.md` names no phase. EXP-001's `sanity_eer.py` filtered to `eval` and
calls that the official protocol; the calibration and drift scripts did not.
The mismatch was found on 2026-09-09 while adding an EER row to Table 1, after
an external exact-PDF audit of 2026-09-08 had asked for the evaluation-population
description of the official-score rows.

## Correction

`EXP-001-scoring-campaign/code/protocol.py` removes hidden-phase trials from
every A2 loader. `progress` is kept: it is untrimmed, and with it every 21LA
condition still holds the same 2,356 bona-fide recordings of the same 67
speakers and the same 21,164 spoofed trials, the twin structure the drift map
and its dependence statements rely on (`eval` alone is 1,924–2,356 per condition
and breaks it). The XLS-R+SLS rows, previously `eval` only, now use the same
`eval + progress` set, so Table 1 has one protocol.

All artifacts were regenerated on 2026-09-09 (E002 → E010 → SLS → drift map →
dependents → N-sweep, parametric, C-methods, speaker clustering, EER/spread →
EXP-403 → EXP-103 cost map → Fig. 1). The all-phase artifacts survive at
repository commit `3cfc9cf` and in the first three commits of the public
package (`9876ced`, `9416deb`, `208d046`).

## What moved

| quantity | all phases (superseded) | hidden excluded |
|---|---|---|
| cells missing target by >2× | 58/108 | 77/108 (36 liberal, 41 conservative) |
| cells inside the ±5 pp tolerance | 84 | 72 |
| conservative >2× misses inside it | 34 | 41 |
| within-21LA cells >2× | 42/84 | 57/84 |
| AASIST PSTN→G.722: FPR / transported FNR / destination-oracle FNR | 0.47% / 74% / 18.8% | 0.42% / 70.1% / 0.66% |
| AASIST FNR on PSTN spoofs at the transported PSTN threshold (`cal_fnr_at_threshold`, six PSTN-calibrated cells) | 38.2–38.5% | 30.6–30.9% |
| AASIST oracle FNR on PSTN at 5% FPR | 38.4% | 30.0% |
| Table 1 AASIST 21LA / 21DF | 36.4 (+0.8) / 43.5 (+0.4) | 19.1 (+0.4) / 36.8 (+0.3) |
| Table 1 SSL-AASIST 21LA / 21DF | 15.9 (+0.7) / 10.5 (+0.1) | 0.2 (+0.0) / 7.1 (+0.2) |
| Table 1 XLS-R+SLS 21LA / 21DF / ITW | 1.6 (+0.1) / 0.8 (+0.0) / 11.2 (+0.4) | 1.5 (+0.1) / 0.7 (+0.1) / 11.3 (+0.5) |
| mean realized FPR, 12 cells | 4.95–5.02% | 4.91–5.06% |
| weighted heuristic within 2 pp / unweighted / improved | 33 / 26 / 70 | 17 / 20 / 74 |
| ACI oracle-label ceiling within 2 pp | 99/108 | 70/108 (min 0.02%, max 86%) |
| mixture-W1 monitor, corrected reading, SSL-AASIST | no operating point | TPR 0.89 / FPR 0.16 in-sample; ±50% re-mix exceeds threshold on 19/19 and 18/19 benign cells |
| mixture-W1 monitor, pre-registered reading, SSL-AASIST | TPR 0.92 / FPR 0.19 | no operating point |
| speaker-disjoint 90% FPR width, AASIST/PSTN | 7.03 pp (MC SD 0.17) | 10.08 pp (MC SD 0.30); null 3.36 |
| cross-detector transfer R² | 0.48 / −0.02 | 0.53 / 0.25 |
| exploratory A5 cost-map held-out R² | 0.59 (0.45–0.64) | −0.81 (−1.31 to −0.67) |
| contamination 5% at N=500, seven cells below target | 0.99–4.99%; eighth 5.11% | 0.31–4.99%; eighth 5.16% |
| Gaussian quantile worst miss | 6.4 pp (0.66 on ITW) | 7.0 pp (0.68 on ITW) |
| resolution-limited cells (FPR < 5/n) | 5 | 13 |

ASVspoof 5 (EXP-103 replication), ITW and BRSpeech-DF cells do not change.

## Direction

The correction strengthens the central transport result and weakens three
secondary ones. It strengthens it: more cells miss, the flagship's destination
oracle falls from 18.8% to 0.66%, and the same-condition FNRs in Table 1 drop.
It weakens: the exploratory cost map no longer transfers (negative R²); the
importance-weighted heuristic is now worse than the unweighted quantile on the
within-2 pp count; and the adaptive oracle-label result is no longer a ceiling,
because in large-shift cells its level parameter reaches the clip bound and the
threshold cannot leave the support of the 500-score cohort. The paper records
the exclusion as a post-hoc protocol correction, not as the design.

## Addendum, 2026-09-09 (later): full 21DF set

After the correction above, the reproduction-scored detectors were rescored on the full
21DF evaluation release (611,829 trials; 5 undecodable files skipped, listed in
`EXP-001-scoring-campaign/skipped.txt`), replacing the pre-registered 100k subsample so
that every Table 1 row uses one protocol (eval + progress: 20,637 bona fide, 572,611 spoofed
trials for the reproduction-scored detectors, 572,616 for the official XLS-R+SLS scores).
The corpus key `asv21df_100k` became `asv21df_full` in every A2 script. No headline count
moved (77/108, 72 in band, 41 hidden, 57/84 within; eval-only identical). Table 1 21DF
cells moved from 36.8 (+0.3) / 7.1 (+0.2) to 36.7 (+0.1) / 6.9 (+0.1); AASIST 21DF EER
14.5 → 14.2; the policy block's 21DF column and the contamination and Gaussian ranges moved
at the first decimal. The subsample artifacts survive at commit `2ceba26`.
