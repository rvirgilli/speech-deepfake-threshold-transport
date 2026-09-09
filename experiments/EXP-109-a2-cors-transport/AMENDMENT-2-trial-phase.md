# AMENDMENT 2 — cell A recomputed without the ASVspoof 2021 hidden phase

Date: **2026-09-09**. Append-only; the frozen [PREREG.md](PREREG.md) (`a7a8dbec…`) is not
edited. Follows [EXP-102 AMENDMENT-3](../EXP-102-a2-campaign/AMENDMENT-3-trial-phase-protocol.md).

## What changed

`transport_released.py` built its transport cells from all 181,566 21LA trials, including
the 16,926 hidden-phase trials whose non-speech intervals were removed by a voice activity
detector and which the organisers excluded from challenge results. The transport cells now
use the eval and progress phases (164,640 trials; 2,356 bona fide and 21,164 spoofed trials
per condition). The sanity EER gate is unchanged (eval phase, as in the official-scores
README) and still passes for all three detectors (2.868 / 0.931 / 1.378, exact to three
decimals). The script's SHA-256 recorded in `FREEZE.sha256` (`adcace90…`) is superseded by
`a0473d32…`; `results_cellA.json` now carries a `phases` field.

## What moved

| Detector | Cells >2× (all phases) | Cells >2× (hidden excluded) | conservative / liberal |
|---|---:|---:|---|
| XLS-R+SLS | 17 | **33** | 17 / 16 |
| XLSR-Mamba | 14 | **25** | 13 / 12 |
| XLSR-Conformer | 21 | **30** | 16 / 14 |
| **Total** | 52 of 126 | **88 of 126** | |

The FPR-axis finding of REPORT-cellA strengthens: the reproduction stratum (EXP-102) gives
57 of 84 under the same protocol.

## What does not survive

REPORT-cellA's statement that the flagship PSTN→G.722 spoof-side cost "reproduces on three
unseen architectures" was an artifact of the trimmed hidden trials. Under the corrected
protocol the three released-score detectors miss 2.7% / 0.9% / 1.9% of G.722 spoofs at the
PSTN-calibrated threshold against destination oracles of 0.17% / 0.08% / 0.03%: spoof-side
costs of 0.9–2.7 points, against 69 points for AASIST. The FPR-axis failure is
detector-general; the spoof-side magnitude is detector-conditioned. The manuscript states
both.
