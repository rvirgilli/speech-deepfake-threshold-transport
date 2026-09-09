# A2 exact-submission review package

This package binds the exact five-page submission PDF, source, bibliography,
numeric guards and score-derived artifacts used for review. It contains no raw
audio and does not rerun detector inference. ICASSP 2027 uses single-anonymous
review, so this package deliberately carries the same named-author manuscript as
the submission; silent PDF/source substitution is not permitted.

Start with `paper/A2/main.pdf`, then `claim_map.json`. `manifest.json` gives a full
SHA-256 digest and size for every packaged file. The EXP-102 directory contains
the primary 108-cell campaign and post-hoc speaker diagnostic; EXP-103 contains
the recording- and speaker-disjoint ASVspoof 5 replication and the descriptive
cost-map reconstruction; EXP-403 contains the two-detector transfer negative.

From `paper/A2/`, run:

```bash
uv run python check_numbers.py
uv run python number_census.py
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The source payload is build-complete: it includes the local `spconf.sty`,
`IEEEbib.bst`, and `figs/drift.pdf` dependencies used by `main.tex`.

From the package root, `python3 verify_manifest.py` checks every packaged file's
size and SHA-256 digest and refuses any file not bound by the manifest.

The current scientific scope is deliberately narrower than the historical
reports. The held-out R² point is descriptive; its former Fisher interval and
p-value are invalid and retired, and its delete-one-A5 range is 0.45--0.64. The
71% versus 50% A5 difference is confounded and is not attributed to
disjointness. Exchangeability without ties gives the quantile's marginal rank
validity; its conditional-FPR Beta law additionally requires iid continuous
sampling from a fixed score distribution. Neither is a per-deployment
guarantee. The speaker-clustering analysis is visibly post-hoc, and its 0.17
value is a Monte Carlo SD rather than population uncertainty.

Per-utterance score tables for every detector--corpus pair are included under
`experiments/EXP-001-scoring-campaign/scores/` and
`experiments/EXP-102-a2-campaign/scores/`, together with the manifest builder
that defines each evaluation set. XLS-R+SLS scores on 21LA and 21DF are the
authors' public release and are not duplicated. Every derived artifact is
accompanied by its source code.

The score-weighting negative is a clipped 15-bin ratio of deployment to
calibration **mixture-score** densities applied to a bona-fide cohort. It is a
heuristic, not Tibshirani et al.'s covariate-shift conformal construction, and
has no test-point weight. The adaptive result uses oracle bona-fide labels and
is a labeled ceiling. See
`experiments/EXP-102-a2-campaign/AMENDMENT-2-estimator-scope.md`.

The experiment's without-replacement finite-pool empirical spread is distinct
from the iid-continuous population Beta reference. The manuscript and serialized
N-sweep state both designs explicitly.

Scope is condition-specific: six processed 21LA conditions are real
transmissions and `none` is the untransmitted source; 21DF is compression-only
and untransmitted.

## Licence

Code is MIT (`LICENSE`). Score tables are derived from ASVspoof 2019, ASVspoof
2021, ASVspoof 5, In-the-Wild and BRSpeech-DF and keep those datasets' terms; no
audio, model weights or third-party code is redistributed.

## Running the analysis code elsewhere

Every script resolves its inputs from the authors' layout by default and accepts
environment overrides: `A2_SCORES` (directory of the EXP-001 score tables, here
`experiments/EXP-001-scoring-campaign/scores`), `A2_DATA` (ASVspoof 2021 keys and
official XLS-R+SLS scores, `keys/{LA,DF}/CM/trial_metadata.txt` and
`official-scores/xlsr-sls/`), `A2_EMB` (EXP-001 embedding arrays, needed only for C5
and not included; `c_methods.py` skips C5 and still computes C1 and C2 when they are
absent), `A2_A5_RAW` (ASVspoof 5 chunk dumps; when absent, `analyze.py` reads the
exported tables in `experiments/EXP-103-a5-replicate/artifacts/`, which carry the same
scores rounded to six decimals), `A2_A5_PROTO` (the ASVspoof 5 evaluation protocol
`ASVspoof5.eval.track_1.tsv`, otherwise under `A2_DATA/asvspoof5`) and
`A2_MANIFESTS` (EXP-001 manifests). Public keys and official scores are downloaded from
asvspoof.org and the SLS authors' repository.
