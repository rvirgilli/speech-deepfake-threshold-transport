# Amendment 4 — the C2 temperature/shift fit diverged; damped Newton installed

Date: 2026-09-09. Applies to `c_methods.py` and `results_cmethods.json` (the unlabeled
correction rows of Table 1 and the "up to 95 pp" statement).

## What was wrong

`fit_temp_shift` fitted the quartile pseudo-label logistic transform by 50 full Newton
steps with no decrease check. On the separable pseudo-labels of the ASVspoof 2019 LA dev
scores the iteration diverged: the mean logistic loss rose from 1.565 to about 6.6×10⁹
(SSL-AASIST source scores; weights of order 10⁹). The transform that reached the paper
was therefore a failed optimization, and its low FPR on 21LA and 21DF (0.3% and 1.4% for
SSL-AASIST) was an accident of that failure. Found by the exact-PDF audit of 2026-09-09
(round 4), which reran the released function on the released scores.

## Correction

Each Newton step is now backtracked (halved, up to 30 times) until the loss decreases,
and the module carries a self-check that the fit separates a synthetic pseudo-label set.
On the real source scores the fitted loss is 6.3×10⁻⁹ (w = 133, b = 807), i.e. a steep
transform whose endpoint after 50 accepted steps defines the method, as in the cited
construction, where the separable objective has no finite optimum.

## What moved (realized FPR at the 5% target, N = 500)

| detector / corpus | C2, diverged fit (superseded) | C2, damped fit |
|---|---|---|
| SSL-AASIST 21LA / 21DF / ITW / BRSpeech | 0.3 / 1.4 / 100 / 100 | 100 / 100 / 100 / 100 |
| AASIST 21LA / 21DF / ITW / BRSpeech | 2.3 / 9.3 / 100 / 100 | 60.8 / 71.1 / 100 / 100 |

C1 and C5 are unchanged. The manuscript's Table 1 now prints both C1 and C2 rows; the
statement that the unlabeled corrections miss the target by up to 95 pp stands. The
superseded file survives at commit `2b97a8a`.

## Note, 2026-09-11: two EER implementations, and why the published one is order-invariant

Raised by the M1 line, which found in its own code an EER computed from `np.argsort`
positions with default tie ordering, and flagged that any such EER depends on input order
rather than on the data. A2 prints twelve EERs in Table 1 plus per-condition ranges, so the
question was whether ours shares that construction. It does not.

`table1_eer_spread.eer` (and the identical `EXP-103/a5_eer.eer`) builds its threshold grid as
`np.unique(concatenate([bona, spoof]))` and reads FPR and FNR by `searchsorted` on the sorted
arrays, so equal scores are collapsed before any comparison and the result is a function of
the score multiset alone. Checked on the real 21LA AASIST cell (16,492 bona fide, 148,148
spoofs): five independent shuffles of both arrays return one distinct value, 8.2647%, the
printed 8.26. Those arrays do contain **1,289 duplicate score values**, so the hazard was
real and was avoided by the grouping, not by the absence of ties.

A second implementation exists in `EXP-109/transport_released.eer`: trial-level, stable
mergesort, `nanargmin` of the frr/far gap. Six permutations under heavy synthetic ties did not
move it, so it is not order-sensitive either, but it differs from the grouped implementation
by 1.6 points on that synthetic data because the two include ties differently at the crossing.
On the real 21LA scores they agree to four decimals on both detectors (8.2647 and 0.8129). No
published number comes from the trial-level one; it backs an internal provenance check. The
divergence is recorded because two implementations that agree on today's data and disagree
under ties are a trap for whoever edits either.
