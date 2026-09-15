# Official per-trial score files (ASVspoof 2021) — provenance

Downloaded 2026-08-14. All EERs verified locally against official keys
(`../keys/{DF,LA}/CM/trial_metadata.txt`, eval-phase subset only).

## xlsr-sls/  — XLS-R + SLS (Zhang et al., ACM MM 2024)
Source snapshot: https://github.com/QiShanZhang/SLSforASVspoof-2021-DF/tree/89a09ac4404c5687d96d6c123fcf6db20e4e4b38/scores
(commit `89a09ac4404c5687d96d6c123fcf6db20e4e4b38`; authenticated against the local files)
- scores_DF.txt   611,829 lines — verified EER 1.916% (paper: 1.92%)
- scores_LA.txt   181,566 lines — verified EER 2.868% (paper: 2.87%)
- scores_Wild.txt  31,779 lines (In-the-Wild)
Format: `<utt_id> <score>` (LLR-like, higher = bonafide)

## xlsr-conformer-rosello/ — XLSR-Conformer (Rosello et al., Interspeech 2023)
Source snapshot: https://github.com/ErosRos/conformer-based-classifier-for-anti-spoofing/tree/e8e195938b898d21ac0105076f883b1571db2664/Scores
(commit `e8e195938b898d21ac0105076f883b1571db2664`; authenticated against the local files)
- Scores_Best_DF_Fixed_size_train.txt          611,829 — EER 2.273%
- Scores_Best_DF_Fixed_size_train_eval_lv.txt  611,829 — EER 2.583%
- Scores_Best_LA_Fixed_size_train.txt          181,566 — EER 1.378%
- Scores_Best_LA_Fixed_size_train_eval_lv.txt  181,566 — EER 0.973%
- Scores_Best_LA_variable_size_train.txt       181,566 — EER 0.871%

## xlsr-mamba/ — XLSR-Mamba (Xiao & Das, IEEE SPL 2025, arXiv 2411.10027)
Source snapshot: https://github.com/swagshaw/XLSR-Mamba/tree/10ec73810d24091028a2aa5454c8dfd1a239e441/Scores
(commit `10ec73810d24091028a2aa5454c8dfd1a239e441`; authenticated against the local files)
- Bmamba3_LA_WCE_1e-06_ES144_NE12.txt (DF)      611,829 — EER 1.884% (paper: 1.88%)
- Bmamba5_LA_WCE_1e-06_ES144_NE12.txt (LA)      181,566 — EER 0.931% (paper: 0.93%)
- Bmamba5_In-the-Wild_WCE_1e-06_ES144_NE12.txt   31,779

## Also available (not copied here)
ASVspoof 2021 official baseline per-trial scores ship inside the keys packages:
`../DF-keys-full/keys/DF/CM/{CQCC-GMM,LFCC-GMM,LFCC-LCNN,RawNet2}/score.txt` (and LA equivalents).
