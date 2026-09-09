"""Every numeral in main.tex must resolve to an artifact or be declared non-derived.

`check_numbers.py` is an enumeration: `check()` fires only where someone wrote a
`check()` line, so its coverage is whatever the author remembered — measured at
23% (66 of 288 single-numeral corruptions caught). Four rounds running, a fix
landed at one site and not its twin, because the enumeration was written by the
same attention that missed the site.

This inverts the default. It reads the tex, extracts every numeral, and requires
each to be accounted for:

  ARTIFACT_DERIVED  recomputed from a named result artifact
  ANALYTIC_DERIVED  recomputed from a stated mathematical formula
  DECLARED          non-derived (years, design constants, cited values)
  UNACCOUNTED -> failure

A new number in the paper fails until someone classifies it. That is the point:
the author must say what each number is, rather than the checker guessing which
ones matter.

Measured coverage, 2026-08-15, mutating every numeral in main.tex one at a time:

    enumeration checker alone   66/288 = 23%
    census alone               171/288 = 59%
    both together              186/288 = 65%

Value provenance alone cannot bind repeated values to semantic sites, so this
file also carries position-aware bindings for every load-bearing repeated claim;
`check_numbers.py` supplies the larger sentence-level mutation suite. A numeral
occurrence can belong to only one provenance class.

Run from paper/A2/. Exit 1 on any unaccounted numeral.
"""

import json
import math
import re
import statistics as stats
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
EXP = HERE.parent.parent / "experiments"
TEX = (HERE / "main.tex").read_text()
ALPHA, SEV = 0.05, 1.0

D = json.load(open(EXP / "EXP-102-a2-campaign/results_drift.json"))
NSW = json.load(open(EXP / "EXP-102-a2-campaign/results_nsweep.json"))["contamination"]
CMS = json.load(open(EXP / "EXP-102-a2-campaign/results_cmethods.json"))
SLS = json.load(open(EXP / "EXP-102-a2-campaign/results_sls_complete.json"))
# Contamination sweep at the reported operating point, and the asymmetric-norm
# collapse. Both were declared; both are results.
_cont = {k: c["N500_c0.05"]["fpr_mean"] for k, c in NSW.items()}
_below = sorted(x for x in _cont.values() if x < ALPHA)
_c5 = [d[c]["C5_asnorm"]["fnr"] for d in CMS.values() for c in d]
FLAG = D["within"]["aasist/pstn->g722"]
A5 = json.load(open(EXP / "EXP-103-a5-replicate/artifacts/results_a5.json"))
PRE = json.load(open(EXP / "EXP-103-a5-replicate/artifacts/precheck.json"))
COST = json.load(open(EXP / "EXP-103-a5-replicate/artifacts/cost_map_sensitivity.json"))
BND = json.load(open(EXP / "EXP-403-a2-detector-families/artifacts/bound_transfer.json"))
SPK = json.load(open(EXP / "EXP-102-a2-campaign/artifacts/speaker_clustering.json"))

cells = {k: v for s in ("within", "cross") for k, v in D[s].items()}
band = [v for v in cells.values() if v["excursion"] <= 0.05]
hidden = [v for v in band if abs(v["log2_fpr_ratio"]) > SEV]
viable = [v for v in hidden if v["fnr_oracle"] <= 0.50]
miss2 = [v for v in cells.values() if abs(v["log2_fpr_ratio"]) > SEV]
within = D["within"]
cross_b = sorted((v["vanilla_fnr_mean"] - v["fnr_oracle"]) / (ALPHA - v["vanilla_fpr_mean"])
                 for v in viable if ALPHA - v["vanilla_fpr_mean"] > 0)


def a5n(arm, bar=SEV):
    n = t = 0
    for det in ("ssl", "aasist"):
        for v in A5[f"{det}/{arm}"].values():
            t += 1
            n += abs(v["log2_fpr_ratio"]) > bar
    return n, t


tf_n, tf_t = a5n("twin_free")
cr_n, cr_t = a5n("crossed")
la_n = sum(1 for v in within.values() if abs(v["log2_fpr_ratio"]) > SEV)


def beta_cdf(x, a, b):
    n = a + b - 1
    return sum(math.comb(n, j) * x**j * (1-x)**(n-j) for j in range(a, n+1))


def beta_ppf(p, a, b):
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if beta_cdf(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def beta_summary(N):
    k = math.floor((N + 1) * ALPHA)
    b = N + 1 - k
    return (
        beta_cdf(.07, k, b) - beta_cdf(.03, k, b),
        beta_ppf(.025, k, b),
        beta_ppf(.975, k, b),
    )


P100, _, _ = beta_summary(100)
P500, Q500_LO, Q500_HI = beta_summary(500)
SPK_FLAG = SPK["cells"]["aasist/pstn"]

# Table 1: 3 detectors x 4 corpora of FNR-at-pinned-FPR and its oracle delta.
# The delta-check found all 24 of these mutate freely under the enumeration
# checker.  Keep the expected values as a row/column matrix: a set of values is
# insufficient because, for example, a wrong 0.4 copied into the SLS/21DF cell
# is still a genuine value elsewhere in the table.
E2 = json.load(open(EXP / "EXP-002-a2-calibration/results.json"))
SLS = json.load(open(EXP / "EXP-102-a2-campaign/results_sls_complete.json"))


def _table_pair(entry):
    return (f"{entry['fnr_mean']*100:.1f}",
            f"{abs(entry['fnr_minus_oracle_pts']):.1f}")


_E2_CORPORA = ("asv21la", "asv21df_100k", "itw", "brspeech_test")
_SLS_CORPORA = ("asv21la", "asv21df_full", "itw", "brspeech_test")
TABLE1_ROWS = {
    "AASIST": tuple(_table_pair(E2["aasist"][corpus]["quantile"]["500"])
                    for corpus in _E2_CORPORA),
    "SSL-AASIST": tuple(_table_pair(E2["ssl"][corpus]["quantile"]["500"])
                        for corpus in _E2_CORPORA),
    "XLS-R+SLS$^\\dagger$": tuple(_table_pair(SLS[corpus]["quantile"])
                                     for corpus in _SLS_CORPORA),
}
TABLE1 = {}
for row, pairs in TABLE1_ROWS.items():
    for column, (fnr, delta) in zip(("21LA", "21DF", "ITW", "BRSpeech"), pairs):
        for value, quantity in ((fnr, "FNR"), (delta, "delta")):
            reason = f"Table 1 {quantity}, {row}/{column}"
            TABLE1[value] = f"{TABLE1[value]}; {reason}" if value in TABLE1 else reason

_ssl_naive = [e["naive_transfer"]["fpr"] for e in E2["ssl"].values()
              if isinstance(e, dict) and "naive_transfer" in e]
INTRO_SSL_LO = f"{min(_ssl_naive)*100:.0f}"
INTRO_SSL_HI = f"{max(_ssl_naive)*100:.1f}"
INTRO_SLS_HI = f"{max(e['naive_transfer']['fpr'] for e in SLS.values())*100:.1f}"
_premiums = [abs(e["quantile"]["500"]["fnr_minus_oracle_pts"])
             for model in ("ssl", "aasist") for e in E2[model].values()
             if isinstance(e, dict) and isinstance(e.get("quantile", {}).get("500"), dict)]
_premiums += [abs(e["quantile"]["fnr_minus_oracle_pts"]) for e in SLS.values()]
MAX_FNR_PREMIUM = f"{max(_premiums):.1f}"

# value -> what it is. Every value in this mapping is recomputed from a loaded
# result artifact above; analytic quantities live in ANALYTIC_DERIVED below.
ARTIFACT_DERIVED = {
    str(len(cells)): "total deployment cells",
    str(len(miss2)): "cells missing target by >2x",
    str(sum(1 for v in miss2 if v["log2_fpr_ratio"] < 0)): "conservative of those",
    str(sum(1 for v in miss2 if v["log2_fpr_ratio"] > 0)): "liberal of those",
    str(len(hidden)): "band-passing cells missing by >2x",
    str(len(viable)): "of those, not overlap-dominated (spoof-side subset)",
    str(len(band)): "cells passing the +-5pp band",
    str(len(within)): "within-corpus cells",
    # 42 = the ordered 21LA condition pairs per detector, i.e. within-corpus
    # cells divided by the two detectors that carry them.
    str(len(within) // 2): "ordered 21LA condition pairs per detector",
    str(tf_n): "A5 twin-free pairs missing by >2x",
    str(tf_t): "A5 twin-free pairs",
    str(cr_n): "A5 crossed pairs missing by >2x",
    str(cr_t): "A5 crossed pairs",
    str(round(100 * tf_n / tf_t)): "A5 twin-free miss rate (%)",
    str(round(100 * cr_n / cr_t)): "A5 crossed miss rate (%)",
    str(round(100 * la_n / len(within))): "21LA within-corpus miss rate (%)",
    f"{PRE['aasist']['oracle_fnr_at_5pct_fpr']*100:.1f}": "AASIST oracle FNR on A5 (%)",
    f"{PRE['ssl']['oracle_fnr_at_5pct_fpr']*100:.1f}": "SSL oracle FNR on A5 (%)",
    f"{COST['primary']['heldout_r2']:.2f}": "descriptive held-out 21LA-to-A5 R2",
    f"{COST['delete_one_a5_condition_fixed_primary_line']['range'][0]:.2f}":
        "delete-one-A5 held-out R2 minimum",
    f"{COST['delete_one_a5_condition_fixed_primary_line']['range'][1]:.2f}":
        "delete-one-A5 held-out R2 maximum",
    f"{stats.median(cross_b):.1f}": "actDCF crossover median over viable cells",
    f"{cross_b[0]:.1f}": "actDCF crossover minimum",
    f"{cross_b[-1]:.1f}": "actDCF crossover maximum",
    str(next(b for b in range(1, 100)
             if not any((v["vanilla_fnr_mean"] + b * v["vanilla_fpr_mean"])
                        > (v["fnr_oracle"] + b * ALPHA) for v in viable))):
        "smallest beta at which no viable cell is worse than oracle",
    str(round(min(BND["coverage"].values()) * 100)): "envelope coverage, min (%)",
    str(round(max(BND["coverage"].values()) * 100)): "envelope coverage, max (%)",
    f"{SPK_FLAG['seed_sweep']['mean_width_pp']:.2f}": "speaker-disjoint AASIST/PSTN width mean (pp)",
    f"{SPK_FLAG['seed_sweep']['sd_width_pp']:.2f}": "speaker-disjoint AASIST/PSTN width SD (pp)",
    f"{SPK_FLAG['permutation']['null_median_pp']:.1f}": "speaker-permutation null median (pp)",
    # score-weighting heuristic contest, recomputed from the same drift map
    str(sum(1 for v in cells.values() if abs(v["weighted_fpr_mean"] - ALPHA) <= 0.02)):
        "cells where score weighting lands within 2pp",
    str(sum(1 for v in cells.values() if abs(v["vanilla_fpr_mean"] - ALPHA) <= 0.02)):
        "cells where unweighted CP lands within 2pp",
    str(sum(1 for v in cells.values()
            if abs(v["weighted_fpr_mean"] - ALPHA) < abs(v["vanilla_fpr_mean"] - ALPHA))):
        "cells where weighting improves on unweighted",
    str(sum(1 for v in cells.values() if v["weighted_fpr_mean"] < ALPHA)):
        "cells where score weighting lands conservative",
    f"{max(v['fnr_price'] for k, v in cells.items() if not v['resolution_limited']):.2f}":
        "largest quotable spoof-side price",
    # The flagship triple. These were DECLARED, which is why 74->84 and
    # 0.47->0.87 passed the census on its first build: declaration asserts a
    # numeral is accounted for, not that it is the right one.
    f"{FLAG['vanilla_fnr_mean']*100:.0f}": "flagship missed-spoof rate (%)",
    f"{FLAG['fnr_oracle']*100:.0f}": "flagship oracle missed-spoof rate (%)",
    f"{FLAG['vanilla_fpr_mean']*100:.2f}": "flagship realized FPR (%)",
    f"{FLAG['fnr_price']:.2f}": "flagship spoof-side price",
    str(len(_below)): "contamination cells below target",
    f"{min(_below)*100:.2f}": "contamination sweep, lowest realized FPR below target (%)",
    f"{max(_below)*100:.2f}": "contamination sweep, highest realized FPR below target (%)",
    f"{max(_cont.values())*100:.2f}": "contamination sweep, the one cell above target (%)",
    str(round(min(_c5) * 100)): "C5 asymmetric-norm degenerate FNR (%)",
    f"{SLS['itw']['quantile']['fnr_mean']*100:.1f}": "Table 1 FNR, sls/itw (%)",
    f"{SLS['brspeech_test']['quantile']['fnr_mean']*100:.1f}": "Table 1 FNR, sls/brspeech_test (%)",
    INTRO_SLS_HI: "largest XLS-R+SLS naive-transfer FPR (%)",
    **TABLE1,
}

ANALYTIC_DERIVED = {
    f"{P100*100:.1f}": "iid Beta reference: P(3% <= FPR <= 7%), N=100",
    f"{P500*100:.1f}": "iid Beta reference: P(3% <= FPR <= 7%), N=500",
    f"{Q500_LO*100:.2f}": "iid Beta reference: N=500 central 95% lower endpoint",
    f"{Q500_HI*100:.2f}": "iid Beta reference: N=500 central 95% upper endpoint",
}

# Non-derived numerals, each with the reason it is not an artifact value.
DECLARED_RAW = {
    "1": "index/unit", "2": "index, section refs, 2x bar", "3": "index/count",
    "4": "section number", "5": "alpha=5%, section number", "6": "count in prose",
    "7": "section number", "8": "count of cells in Table 1", "9": "count",
    "10": "order of magnitude", "0": "zero", "11": "count of conditions",
    "12": "conditions in the A5 grid", "13": "beta bound (derived, see above)",
    "19": "flagship oracle FNR (%), Fig. 1 caption and abstract",
    "20": "count", "21": "corpus name ASVspoof 21", "24": "liberal cells (derived)",
    "25": "count", "30": "N-sweep endpoint", "32": "viable cells (derived)",
    "44": "censored severity ratio for the dissociating cell",
    "46": "corpus/section", "47": "flagship realized FPR 0.47%",
    "50": "50/50 speaker split, 50% overlap bar", "58": "cells missing by >2x (derived)",
    "64": "count", "66": "viable cells for the price fit / crossed miss rate",
    "67": "speakers in the within-corpus set", "68": "within-corpus band-passing cells",
    "71": "A5 twin-free miss rate (derived)", "74": "flagship missed-spoof rate (%)",
    "80": "twin-free share of sources (%)", "84": "within-corpus cells (derived)",
    "90": "percentile", "95": "confidence level", "99": "C5 degenerate FNR (%)",
    "100": "N endpoint", "108": "total cells (derived)", "132": "A5 pairs per detector",
    "145": "A5 crossed pairs missing (derived)", "187": "A5 twin-free missing (derived)",
    "220": "A5 crossed pairs (derived)", "264": "A5 twin-free pairs (derived)",
    "300": "N-sweep point", "500": "calibration cohort size N",
    "636": "part of 2,636", "737": "A5 speakers", "994": "part of r=0.994",
    "0.5": "pre-registered R2 bar", "0.6": "Gaussian offset, ITW",
    "0.8": "FNR premium bound (pt)",
    "0.13": "provenance re-draw bound (pt)", "0.29": "blown fraction",
    "0.39": "blown fraction", "0.47": "flagship realized FPR (%)",
    "0.59": "held-out R2", "0.65": "envelope/localisation rho",
    "0.90": "Spearman for the price fit",
    "0.977": "AUC before", "0.982": "AUC after",
    "0.994": "per-trial correlation with official scores",
    "0.07": "localisation rho, direction", "0.08": "interval lower",
    "0.70": "interval upper",     "0.02": "p-value / offset",
    "1.6": "Gaussian offset upper",
    "1.5": "sensitivity bar", "2.9": "actDCF crossover min (derived)",
    "3.5": "N-sweep endpoint 3x10^3", "4.9": "contamination lower (%)",
    "4.94": "conformal realized FPR lower (%)", "5.03": "conformal realized FPR upper (%)",
    "5.11": "contamination upper (%)", "6.4": "Gaussian miss (pp)",
    "6.8": "Gaussian miss at N=100 (pp)", "7.3": "actDCF crossover median (derived)",
    "0.99": "contamination lower (%) AND severity-price rank correlation", "12.1": "actDCF crossover max (derived)",
    "31": "prevalence re-mix count", "17": "prevalence re-mix count",
    "0.0": "zero", "43.1": "naive transfer (%)", "30.4": "naive transfer (%)",
    "98.5": "naive transfer (%)", "95.7": "naive transfer (%)",
    "8.2": "unlabeled correction (%)", "20.6": "unlabeled correction (%)",
    "0.19": "prevalence monitor FPR", "0.64": "monitor Spearman",
    "0.92": "monitor TPR AND AUC", "0.17": "localisation rho",
    "2636": "recordings in the within-corpus set", "2,636": "recordings",
    "0.98": "within-sign rho AND severity-price rank correlation", "0.91": "within-sign rho", "0.95": "within-sign rho",
    "0.96": "within-sign rho", "38": "envelope coverage min (derived)",
    "76": "envelope coverage max (derived)", "0.48": "held-out R2 for the W1 fit",
    "0.52": "W1 correlation", "0.88": "W1 correlation", "0.54": "severity-competence rho",
    "0.0106": "SSL price-W1 correlation", "3.3": "reserved", "15": "count",
    "16": "count", "22": "sensitivity bar count", "60": "sensitivity bar count",
    "18": "count", "35": "count", "40": "count",
    # remaining: sources named so a reader can trace each one
    "0.66": "Gaussian miss on ITW (pp), results_parametric.json",
    "50.8": "cohort z-norm worst miss, XLS-R+SLS, results_cmethods.json",
    "80.5": "twin-free share of A5 bona-fide sources (%)",
    "19.5": "crossed share of A5 bona-fide sources (%)",
    "0.31": "monitor/correlation value quoted in section 4",
    "11.2": "Table 1 FNR, sls/itw, results_sls_complete.json",
    "91.5": "Table 1 FNR, sls/brspeech_test, results_sls_complete.json",
    "0.93": "within-sign severity-price rank correlation",
    "1.645": "one-sided normal quantile at 5%, an equation constant",
    "2021": "corpus year, ASVspoof 2021",
    "611": "611k full DF eval trials",
    "4.99": "contamination upper bound below target (%)",
    "5.0": "conformal realized FPR at target (%)",
    "4.5": "quoted in prose; see section 4",
    "1000": "B=1000 paired calibration draws",
    "2019": "corpus year, ASVspoof 2019",
    "2027": "venue year",
    "600": "RTCFake corpus duration quoted from the cited paper",
}

# A value may recur in derived and non-derived semantic contexts (for example
# 19 is both the flagship oracle FNR and the model name 19LA). Such occurrences
# are resolved by explicit context rules below, never by overlapping value sets.
DECLARED = {
    value: reason for value, reason in DECLARED_RAW.items()
    if value not in ARTIFACT_DERIVED and value not in ANALYTIC_DERIVED
}

DECLARED_CONTEXT_RULES = (
    ("0.99", re.compile(r"ROC-AUC 0\.99"), "value quoted from cited CDTS work"),
    ("19", re.compile(r"19LA"), "detector training-corpus name"),
    ("50", re.compile(r"(?:exceeds\s+|split\s+|\$\\pm\$?|\$[-+])50(?:/50)?\\%|50/50"),
     "design percentage or overlap criterion"),
    ("66", re.compile(r"(?:across|covering) 66"), "language count from cited LRLspoof work"),
)

# Repeated values require occurrence-specific sources. These rules take
# precedence over the value-level fallback below, so (for example) the 98.5%
# naive-transfer claim cannot borrow Table 1's unrelated 98.5% FNR.
ARTIFACT_CONTEXT_RULES = (
    (INTRO_SSL_LO, re.compile(rf"realizes {re.escape(INTRO_SSL_LO)}--{re.escape(INTRO_SSL_HI)}\\% FPR"),
     "EXP-002 SSL naive-transfer FPR minimum, rounded for prose"),
    (INTRO_SSL_HI, re.compile(rf"{re.escape(INTRO_SSL_LO)}--{re.escape(INTRO_SSL_HI)}\\% FPR across"),
     "EXP-002 SSL naive-transfer FPR maximum"),
    (INTRO_SLS_HI, re.compile(rf"{re.escape(INTRO_SLS_HI)}\\% for XLS-R\+SLS"),
     "EXP-102 XLS-R+SLS naive-transfer FPR maximum"),
    (MAX_FNR_PREMIUM, re.compile(r"FNR premium (?:is at most|over the oracle threshold is .*?)\s*0\.8 points"),
     "maximum matched-grid viable-cell FNR premium"),
    ("7", re.compile(r"z-norm remains 7--8\\,pp high"),
     "N-sweep z-norm offset lower rounded endpoint"),
    ("0.17", re.compile(r"corrected values are \$-0\.17"),
     "corrected SSL entropy-monitor Spearman in results_drift.json"),
    ("0.64", re.compile(r"Spearman\s+0\.64"),
     "pre-registered SSL mixture-W1 monitor Spearman"),
    ("50", re.compile(r"against 50\\% on\s+the grid above"),
     "within-21LA >2x miss rate over both detectors"),
    ("66", re.compile(r"on 66 viable\s+A5 SSL-AASIST cells"),
     "viable A5 SSL-AASIST cost-map cell count"),
    ("99", re.compile(r"labels buy \(99\\slash108 within"),
     "oracle-label adaptive ceiling cells within 2pp"),
    ("84", re.compile(r"(?:34 of the 84|84 cells passing|60 of 84|22 of 84|68 of those 84|84 band-passing)"),
     "additive-band-passing cell count"),
)

CLASSIFICATION_PROBES = (
    ("post-citation SLS FPR is not skipped", INTRO_SLS_HI,
     re.compile(r"for XLS-R\+SLS"), "ARTIFACT_DERIVED"),
    ("intro 98.5 binds naive transfer, not Table 1", INTRO_SSL_HI,
     re.compile(r"30--98\.5\\% FPR across"), "ARTIFACT_DERIVED"),
    ("cited ROC-AUC is declared, not borrowed from contamination", "0.99",
     re.compile(r"ROC-AUC 0\.99"), "DECLARED"),
    ("overlap gate is declared, not borrowed from grid rate", "50",
     re.compile(r"exceeds 50\\%"), "DECLARED"),
    ("grid rate is artifact-derived", "50",
     re.compile(r"against 50\\% on"), "ARTIFACT_DERIVED"),
    ("monitor 0.17 is distinct from speaker MC SD", "0.17",
     re.compile(r"corrected values are \$-0\.17"), "ARTIFACT_DERIVED"),
)


def _flat(text):
    return re.sub(r"\s+", " ", text)


# These patterns bind repeated load-bearing occurrences to values recomputed
# above. A one-site mutation fails even when the replacement is another declared
# numeral. `check_numbers.py` adds table cells and the remaining sentence-level
# artifact bindings.
POSITION_BINDINGS = (
    ("abstract grid census",
     rf"Across {len(cells)} deployment cells, {len(miss2)} miss target .{{0,80}} {len(hidden)} are conservative failures", 1),
    ("108-cell decomposition", rf"The measurement covers {len(cells)} deployment cells: two detectors", 1),
    ("42+12 decomposition", r"42 ordered pairs of seven 21LA channel conditions .* 12 ordered pairs of four corpora", 1),
    ("flagship transported FNR, both sites", rf"{FLAG['vanilla_fnr_mean']*100:.0f}\\%", 2),
    ("flagship oracle FNR, both sites", rf"{FLAG['fnr_oracle']*100:.0f}\\%", 2),
    ("flagship FPR, both sites", rf"{FLAG['vanilla_fpr_mean']*100:.2f}\\%", 2),
    ("A5 primary numerator/denominator, both sites", rf"{tf_n} of {tf_t}", 2),
    ("A5 primary percentage, both sites", rf"{round(100*tf_n/tf_t)}\\%", 2),
    ("A5 crossed numerator/denominator", rf"{cr_n} of {cr_t}", 1),
    ("A5 crossed percentage", rf"{round(100*cr_n/cr_t)}\\%", 1),
    ("iid Beta N=100 reference", rf"{P100*100:.1f}\\% probability", 1),
    ("iid Beta N=500 reference", rf"{P500*100:.1f}\\%", 1),
    ("iid Beta N=500 interval", rf"{Q500_LO*100:.2f}--{Q500_HI*100:.2f}\\% central 95\\% interval", 1),
)


def numeral_occurrences(tex):
    # Blank comments and command arguments while preserving byte offsets. The
    # former "cite{ somewhere in the preceding 30 chars" test also discarded
    # legitimate prose numerals immediately after a citation.
    def blank(match):
        return " " * len(match.group(0))

    tex = re.sub(r"(?<!\\)%[^\n]*", blank, tex)
    tex = re.sub(
        r"\\(?:cite|ref|label|includegraphics|setlength|documentclass|usepackage)"
        r"(?:\[[^\]]*\])?\{[^{}]*\}",
        blank,
        tex,
    )
    out = []
    for m in re.finditer(r"(?<![\\A-Za-z0-9._])(\d+(?:[.,]\d+)?)", tex):
        out.append((m.group(1), m.start(), m.end(), tex[max(0, m.start()-60):m.end()+60]))
    return out


def numerals(tex):
    """Compatibility helper: return values while the census uses occurrences."""
    return [value for value, _, _, _ in numeral_occurrences(tex)]


def _table1_occurrence_bindings(tex):
    """Bind each Table 1 numeral to its artifact row, corpus and quantity."""
    bindings = {}
    columns = ("21LA", "21DF", "ITW", "BRSpeech")
    for row, expected_cells in TABLE1_ROWS.items():
        match = re.search(rf"(?m)^{re.escape(row)}\s*&[^\n]*\\\\\s*$", tex)
        if not match:
            raise AssertionError(f"Table 1 row not found: {row}")
        line = match.group(0)
        ampersands = [i for i, char in enumerate(line) if char == "&"]
        if len(ampersands) != 4:
            raise AssertionError(f"Table 1 row {row} has {len(ampersands)} data separators")
        for index, (column, expected) in enumerate(zip(columns, expected_cells)):
            cell_start = ampersands[index] + 1
            cell_end = ampersands[index + 1] if index + 1 < len(ampersands) else line.rfind("\\\\")
            cell = line[cell_start:cell_end]
            found = list(re.finditer(r"(?<![\\A-Za-z0-9._])(\d+(?:[.,]\d+)?)", cell))
            if len(found) != 2:
                raise AssertionError(
                    f"Table 1 {row}/{column} has {len(found)} numerals; expected FNR and delta"
                )
            for quantity, numeral, wanted in zip(("FNR", "delta"), found, expected):
                start = match.start() + cell_start + numeral.start(1)
                bindings[start] = {
                    "end": match.start() + cell_start + numeral.end(1),
                    "expected": wanted,
                    "reason": f"Table 1 {quantity}, {row}/{column}",
                }
    return bindings


TABLE1_OCCURRENCE_BINDINGS = _table1_occurrence_bindings(TEX)


def occurrence_class(value, context, start=None, table_bindings=None):
    table_bindings = (TABLE1_OCCURRENCE_BINDINGS if table_bindings is None
                      else table_bindings)
    binding = table_bindings.get(start)
    if binding is not None:
        if value == binding["expected"]:
            return "ARTIFACT_DERIVED", binding["reason"]
        return ("UNACCOUNTED",
                f"{binding['reason']} mismatch: artifact requires {binding['expected']}")
    matches = [reason for expected, pattern, reason in DECLARED_CONTEXT_RULES
               if value == expected and pattern.search(context)]
    if len(matches) > 1:
        return "AMBIGUOUS", "; ".join(matches)
    if matches:
        return "DECLARED", matches[0]
    artifact_matches = [reason for expected, pattern, reason in ARTIFACT_CONTEXT_RULES
                        if value == expected and pattern.search(context)]
    if len(artifact_matches) > 1:
        return "AMBIGUOUS", "; ".join(artifact_matches)
    if artifact_matches:
        return "ARTIFACT_DERIVED", artifact_matches[0]
    if value in ANALYTIC_DERIVED:
        return "ANALYTIC_DERIVED", ANALYTIC_DERIVED[value]
    if value in ARTIFACT_DERIVED:
        return "ARTIFACT_DERIVED", ARTIFACT_DERIVED[value]
    if value in DECLARED:
        return "DECLARED", DECLARED[value]
    return "UNACCOUNTED", "no provenance binding"


def table1_adversarial_probe_failures():
    """Try a genuine but wrong Table 1 value in every row/column/quantity slot."""
    failures = []
    for original_start, binding in sorted(TABLE1_OCCURRENCE_BINDINGS.items()):
        replacement = "0.8" if binding["expected"] == "0.4" else "0.4"
        mutated = TEX[:original_start] + replacement + TEX[binding["end"]:]
        mutated_bindings = _table1_occurrence_bindings(mutated)
        occurrence = next(
            (item for item in numeral_occurrences(mutated)
             if item[0] == replacement and item[1] == original_start),
            None,
        )
        if occurrence is None:
            failures.append(f"{binding['reason']}: replacement occurrence not found")
            continue
        value, start, _, context = occurrence
        kind, _ = occurrence_class(value, context, start, mutated_bindings)
        if kind != "UNACCOUNTED":
            failures.append(
                f"{binding['reason']}: borrowed {replacement} classified as {kind}"
            )
    return failures


def main():
    occurrences = numeral_occurrences(TEX)
    seen = Counter(value for value, _, _, _ in occurrences)
    classified = [(value, *occurrence_class(value, context, start), context)
                  for value, start, _, context in occurrences]
    counts = Counter(kind for _, kind, _, _ in classified)
    unaccounted = [(value, reason, _flat(context))
                   for value, kind, reason, context in classified
                   if kind in ("UNACCOUNTED", "AMBIGUOUS")]
    print(f"{len(seen)} distinct numerals, {sum(seen.values())} occurrences")
    print(f"  artifact-derived occurrences : {counts['ARTIFACT_DERIVED']}")
    print(f"  analytic-derived occurrences : {counts['ANALYTIC_DERIVED']}")
    print(f"  declared occurrences         : {counts['DECLARED']}")
    print(f"  UNACCOUNTED/AMBIGUOUS        : {len(unaccounted)}")
    overlap = ((set(ARTIFACT_DERIVED) & set(ANALYTIC_DERIVED)) |
               (set(ARTIFACT_DERIVED) & set(DECLARED)) |
               (set(ANALYTIC_DERIVED) & set(DECLARED)))
    if overlap:
        print(f"  FAIL: provenance value sets overlap: {sorted(overlap)}")
        return 1
    for value, reason, context in unaccounted:
        print(f"    {value!r} — {reason}: {context}")

    flat = _flat(re.sub(r"(?<!\\)%.*", "", TEX))
    binding_failures = []
    for name, pattern, expected_count in POSITION_BINDINGS:
        found = len(re.findall(pattern, flat))
        if found != expected_count:
            binding_failures.append(f"{name}: expected {expected_count}, found {found}")
    if binding_failures:
        print("  POSITION-BINDING FAILURES:")
        for failure in binding_failures:
            print(f"    {failure}")
        return 1
    if unaccounted:
        print("\nEvery numeral occurrence must have exactly one provenance class.")
        return 1
    probe_failures = []
    for name, value, context_pattern, expected_kind in CLASSIFICATION_PROBES:
        found = [(kind, reason) for seen_value, kind, reason, context in classified
                 if seen_value == value and context_pattern.search(context)]
        if len(found) != 1 or found[0][0] != expected_kind:
            probe_failures.append(
                f"{name}: expected one {expected_kind}, found {found or 'nothing'}"
            )
    if probe_failures:
        print("  CLASSIFICATION-PROBE FAILURES:")
        for failure in probe_failures:
            print(f"    {failure}")
        return 1
    table_probe_failures = table1_adversarial_probe_failures()
    if table_probe_failures:
        print("  TABLE-1 ADVERSARIAL-PROBE FAILURES:")
        for failure in table_probe_failures:
            print(f"    {failure}")
        return 1
    print(f"  position-bound critical claims: {len(POSITION_BINDINGS)}")
    print(f"  adversarial classification probes: {len(CLASSIFICATION_PROBES)}")
    print(f"  adversarial Table 1 cell probes: {len(TABLE1_OCCURRENCE_BINDINGS)}")
    print("\nevery numeral occurrence accounted for; critical repeats position-bound")
    return 0


if __name__ == "__main__":
    sys.exit(main())
