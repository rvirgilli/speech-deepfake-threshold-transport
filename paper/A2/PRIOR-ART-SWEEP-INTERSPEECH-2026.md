# Interspeech 2026 prior-art sweep for A2

**Closed:** 2026-08-28  
**Purpose:** test whether the accepted Interspeech 2026 program contains a competitor to
A2's fixed-FPR threshold-transport measurement and target-channel bona-fide calibration
study. This is a search record, not evidence for any empirical claim in the manuscript.

## Source and coverage

The official Interspeech 2026 program search endpoint was queried on 2026-08-28:

- program page: <https://interspeech2026.org/en-AU/pages/programme/program>
- public program endpoint:
  <https://interspeech2026.org/api/program-sheet?spreadsheetId=1bUDt3Bp0ApPATiG3bS7RzHq68tWFZsFOnTz3uBkXoEk>

The response contained **1,416 program rows**, including **1,384 numbered papers**. Every
title and session name was searched case-insensitively. The broad screen used `deepfake`,
`deep fake`, `spoof`, `anti-spoof`, `synthetic speech`, `fake audio`, `generated speech`,
`voice clone`, and `voice cloning`; it returned **77 rows**. The conference's dedicated
spoof/deepfake/synthetic-speech sessions contained **60 papers**, all of which were screened.
A second screen used `conformal`, `false positive`, `false alarm`, `FPR`, `operating point`,
`threshold calibration`, the `calibr*` stem (covering `calibrate`, `calibrated`, and
`calibration`), `quantile`, `order statistic`, `Neyman`, and `type-I control`. It returned
seven calibration/conformal titles in other speech tasks and **zero
deepfake/anti-spoofing titles**. The available preprints or project pages for the closest
title-level candidates were then read.

## Closest papers and disposition

| Program no. | Paper | Disposition for A2 |
|---:|---|---|
| 345 | [*When Spoof Detectors Travel: Evaluation Across 66 Languages in the Low-Resource Language Spoofing Corpus*](https://arxiv.org/abs/2603.02364) | The direct threshold-transport neighbor. It transfers an externally calibrated EER threshold but has spoof-only target data and reports spoof rejection; A2 has target bona fide and studies FPR. Already cited as `lrlspoof26` and contrasted explicitly. |
| 1098 | [*QAMO: Quality-aware Multi-centroid One-class Learning For Speech Deepfake Detection*](https://arxiv.org/abs/2509.20679) | One-class representation learning with EER evaluation, not post-training target-domain threshold calibration or finite-sample FPR control. No claim collision. |
| 1366 | [*DeepFense: A Unified, Modular, and Extensible Framework for Robust Audio Deepfake Detection*](https://arxiv.org/abs/2604.08450) | Toolkit and large architecture/training-data benchmark using cross-domain detection metrics. No fixed operating-point transport or target-bona-fide calibration contribution. |
| 3210 | [*Evidence Subspace Projection: Measuring How Much Evidence Explains Deepfake Detection in Self-Supervised Speech Models*](https://arxiv.org/abs/2607.11538) | Interpretability analysis of evidence factors including codec and transmission. It does not measure transported-threshold FPR or propose A2's calibration regime. |
| 2167 | *Hard Positive-targeted Training for Robust Audio Deepfake Detection under Neural Codec Processing* (program title; no public preprint located) | Detector robustness under codec processing; not real telephony threshold transport. Its scope reinforces the need to keep 21DF compression distinct from 21LA transmission, which A2 already does. |
| 908 | [*Supervised Post-training of Speech Foundation Models for Robust Adaptation in Speech Deepfake Detection*](https://arxiv.org/abs/2606.25328) | Detector post-training/adaptation evaluated by EER, not a fixed-detector threshold-policy study. |
| 3019 | [*SEA-Spoof: Bridging the Gap in Multilingual Audio Deepfake Detection for South-East Asia*](https://arxiv.org/abs/2509.19865) | Multilingual dataset and fine-tuning study; no target-bona-fide operating-point guarantee. |
| 2283 | [*Bridging the Age Gap: Towards Detecting Neural Audio Codec Synthesized Elderly Speech Deepfake*](https://arxiv.org/abs/2606.21735) | Demographic/codec-deepfake benchmark and detector study; no threshold-transport claim. |
| 3212 | [*Quantizer-Aware Hierarchical Neural Codec Modeling for Speech Deepfake Detection*](https://arxiv.org/abs/2603.16914) | Codec-representation detector architecture; no threshold-policy contribution. |
| 157 | [*Exploring the Scale and Diversity of Speech Anti-spoofing Datasets: Experiments and Analysis*](https://arxiv.org/abs/2606.08038) | Training-set diversity study evaluated cross-domain; no fixed-FPR calibration study. |

Other screened papers concern architectures, datasets, multilingual/demographic robustness,
explainability, source attribution, watermarking, efficiency, adversarial robustness, or
synthetic-speech generation. None claims the combination A2 relies on: a fixed detector,
threshold transport over real telephony/channel conditions, realized target FPR at a
predeclared operating point, and target-channel bona-fide order-statistic recalibration with
finite-sample assumption separation.

## Manuscript consequence

No scope, framing, or headline change is required. LRLspoof was already cited and is the only
new accepted paper requiring an explicit contrast. The sweep therefore closes the open marker
in `main.tex`; it does not justify a claim that no unpublished or differently worded work can
exist.
