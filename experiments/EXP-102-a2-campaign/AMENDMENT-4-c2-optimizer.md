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
