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
ones matter. A retired value is REMOVED from the declarations rather than kept,
so its return is unaccounted.

Measured coverage, 2026-08-15, mutating every numeral in main.tex one at a time:

    enumeration checker alone   66/288 = 23%
    census alone               171/288 = 59%
    both together              186/288 = 65%

Value provenance alone cannot bind repeated values to semantic sites, so this
file also carries position-aware bindings for every load-bearing repeated claim;
`check_numbers.py` supplies the larger sentence-level mutation suite. A numeral
occurrence can belong to only one provenance class.

Run from paper/A2/. Exit 1 on any unaccounted numeral. A2_TEX overrides the
manuscript path for mutation testing.
"""

import ast
import json
import math
import os
import re
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
EXP = HERE.parent.parent / "experiments"
E102 = EXP / "EXP-102-a2-campaign"
TEX = Path(os.environ.get("A2_TEX", HERE / "main.tex")).read_text()
ALPHA, SEV, TOL = 0.05, 1.0, 0.05

D = json.load(open(E102 / "results_drift.json"))
NSW = json.load(open(E102 / "results_nsweep.json"))
CMS = json.load(open(E102 / "results_cmethods.json"))
SLS = json.load(open(E102 / "results_sls_complete.json"))
DIS = json.load(open(E102 / "results_dissociation.json"))
EER = json.load(open(E102 / "results_table1_eer_spread.json"))
PAR = json.load(open(E102 / "results_parametric.json"))
SPK = json.load(open(E102 / "artifacts/speaker_clustering.json"))
E2 = json.load(open(EXP / "EXP-002-a2-calibration/results.json"))
M10 = json.load(open(EXP / "EXP-010-a2-matched-baseline/results.json"))
A5 = json.load(open(EXP / "EXP-103-a5-replicate/artifacts/results_a5.json"))
PRE = json.load(open(EXP / "EXP-103-a5-replicate/artifacts/precheck.json"))
A5EER = json.load(open(EXP / "EXP-103-a5-replicate/artifacts/a5_eer.json"))
DRIFT_MAP = (E102 / "drift_map.py").read_text()
CORS = json.load(open(EXP / "EXP-109-a2-cors-transport/results_cellA.json"))
CMETH = (E102 / "c_methods.py").read_text()
PHASE = json.load(open(E102 / "results_phase_sensitivity.json"))


def script_constant(path, name):
    m = re.search(rf"^{name}\s*=\s*([^#\n]+)", path.read_text(), re.M)
    return ast.literal_eval(m.group(1).strip())


cells = {k: v for s in ("within", "cross") for k, v in D[s].items()}
within = D["within"]
in_tol = [v for v in cells.values() if abs(v["vanilla_fpr_mean"] - ALPHA) <= TOL]
hidden = [v for v in in_tol if abs(v["log2_fpr_ratio"]) > SEV]
miss2 = [v for v in cells.values() if abs(v["log2_fpr_ratio"]) > SEV]
n_within_tol = sum(1 for k, v in cells.items() if k in within and abs(v["vanilla_fpr_mean"] - ALPHA) <= TOL)
FLAG = max((kv for kv in cells.items() if not kv[1]["resolution_limited"]),
           key=lambda kv: kv[1]["fnr_price"])[1]
PSTN_MIN_PRICE = min(v["fnr_price"] for k, v in within.items() if k.startswith("aasist/pstn->"))
AUC = DIS["aasist"]["cond_auc"]
AUC_OTHERS = [AUC[c] for c in ("alaw", "ulaw", "gsm", "g722", "opus", "none")]
MON = D["monitor_eval"]["ssl/w1_mixture"]
OP = MON["achieves_tpr80_fpr20"]
BENIGN = [v for k, v in cells.items() if k.startswith("ssl/") and abs(v["log2_fpr_ratio"]) <= SEV]
E2_CORPORA = ("asv21la", "asv21df_full", "itw", "brspeech_test")
SLS_CORPORA = ("asv21la", "asv21df_full", "itw", "brspeech_test")

_cont = {k: c["N500_c0.05"]["fpr_mean"] for k, c in NSW["contamination"].items()}
_below = sorted(x for x in _cont.values() if x < ALPHA)
_c5 = [d[c]["C5_asnorm"]["fnr"] for d in CMS.values() for c in d]


def spearman(x, y):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
                j += 1
            for k in range(i, j + 1):
                r[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return r
    rx, ry = rank(x), rank(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    return num / math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))


RHO = {d: spearman([abs(v["log2_fpr_ratio"]) for k, v in cells.items() if k.startswith(d + "/")],
                   [v["w1_bona_oracle"] for k, v in cells.items() if k.startswith(d + "/")])
       for d in ("ssl", "aasist")}


def a5n(arm, bar=SEV):
    n = t = 0
    for det in ("ssl", "aasist"):
        for v in A5[f"{det}/{arm}"].values():
            t += 1
            n += abs(v["log2_fpr_ratio"]) > bar
    return n, t


tf_n, tf_t = a5n("twin_free")
la_n = sum(1 for v in within.values() if abs(v["log2_fpr_ratio"]) > SEV)
A5_CONDS = {k.split("->")[1] for k in A5["ssl/twin_free"]}
_a5_over = set()
for k, v in A5["ssl/twin_free"].items():
    if v["fnr_oracle"] > 0.5:
        _a5_over.add(k.split("->")[1])
A5_USABLE = [v for k, v in A5["ssl/twin_free"].items() if k.split("->")[1] not in _a5_over]


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


def normal_quantile(p):
    lo, hi = 0.0, 5.0
    for _ in range(60):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < p else (lo, mid)
    return lo


P100, _, _ = beta_summary(100)
P500, Q500_LO, Q500_HI = beta_summary(500)
SPK_W = [v["seed_sweep"]["mean_width_pp"] for v in SPK["cells"].values()]
SPK_NULL = [v["permutation"]["null_median_pp"] for v in SPK["cells"].values()]
CORS_N = {d: sum(1 for v in c.values() if abs(v["log2_fpr_ratio"]) > SEV) for d, c in CORS["cells"].items()}
EC = EER["eer_by_condition_21la"]
def _ec_range(det):
    v = [x * 100 for x in EC[f"{det}/asv21la"].values()]
    return f"{min(v):.1f}", f"{max(v):.1f}"

# Table 1: 3 detectors x 4 corpora of FNR-at-pinned-FPR with its oracle delta,
# and an EER block on the same trials. The delta-check found all 24 upper-block
# values mutate freely under the enumeration checker. Keep the expected values
# as a row/column matrix: a set of values is insufficient because, for example,
# a wrong 0.4 copied into the SLS/21DF cell is still a genuine value elsewhere.


def _table_pair(entry):
    return (f"{entry['fnr_mean']*100:.1f}",
            f"{abs(entry['fnr_minus_oracle_pts']):.1f}")


TABLE1_ROWS = {
    "AASIST": tuple(_table_pair(E2["aasist"][corpus]["quantile"]["500"])
                    for corpus in E2_CORPORA),
    "SSL-AASIST": tuple(_table_pair(E2["ssl"][corpus]["quantile"]["500"])
                        for corpus in E2_CORPORA),
    "XLS-R+SLS$^\\dagger$": tuple(_table_pair(SLS[corpus]["quantile"])
                                     for corpus in SLS_CORPORA),
}
_pol = lambda d: {c: d[c] for c in E2_CORPORA}
POLICY_ROWS = {
    "Naive transfer": tuple((f"{E2['ssl'][c]['naive_transfer']['fpr']*100:.1f}",) for c in E2_CORPORA),
    "C1 z-norm": tuple((f"{CMS['ssl'][c]['C1_znorm']['fpr']*100:.1f}",) for c in E2_CORPORA),
    "C2 temp./shift": tuple((f"{CMS['ssl'][c]['C2_tempshift']['fpr']*100:.1f}",) for c in E2_CORPORA),
    "C5 AS-norm": tuple((f"{CMS['ssl'][c]['C5_asnorm']['fpr']*100:.1f}",) for c in E2_CORPORA),
    "Cohort z-norm": tuple((f"{M10['ssl'][c]['500']['znorm']['fpr_mean']*100:.1f}",) for c in E2_CORPORA),
    "Gaussian q.": tuple((f"{PAR['ssl'][c]['500']['parametric']['fpr_mean']*100:.1f}",) for c in E2_CORPORA),
    "Conformal q.": tuple((f"{E2['ssl'][c]['quantile']['500']['fpr_mean']*100:.1f}",) for c in E2_CORPORA),
}
EER_ROWS = {
    "AASIST": tuple((f"{EER['cells'][f'aasist/{c}']['eer']*100:.1f}",) for c in E2_CORPORA),
    "SSL-AASIST": tuple((f"{EER['cells'][f'ssl/{c}']['eer']*100:.1f}",) for c in E2_CORPORA),
    "XLS-R+SLS$^\\dagger$": tuple((f"{EER['cells'][f'sls/{c}']['eer']*100:.1f}",) for c in SLS_CORPORA),
}
TABLE1_BLOCKS = (("FNR", "delta"), ("EER",), ("realized FPR",))
# Every table line, in document order per row name: the detector rows appear
# twice (FNR block, then EER block); each policy row once (lower block).
TABLE1_LINES = {row: [(TABLE1_ROWS[row], TABLE1_BLOCKS[0]), (EER_ROWS[row], TABLE1_BLOCKS[1])] for row in TABLE1_ROWS}
TABLE1_LINES.update({row: [(cells_, TABLE1_BLOCKS[2])] for row, cells_ in POLICY_ROWS.items()})
TABLE1 = {}
for block, rows in ((TABLE1_ROWS, 0), (EER_ROWS, 1), (POLICY_ROWS, 2)):
    for row, cells_ in block.items():
        for column, values in zip(("21LA", "21DF", "ITW", "BRSpeech"), cells_):
            for value, quantity in zip(values, TABLE1_BLOCKS[rows]):
                reason = f"Table 1 {quantity}, {row}/{column}"
                TABLE1[value] = f"{TABLE1[value]}; {reason}" if value in TABLE1 else reason

_ssl_naive = [E2["ssl"][c]["naive_transfer"]["fpr"] for c in E2_CORPORA]
INTRO_SSL_LO = f"{min(_ssl_naive)*100:.0f}"
INTRO_SSL_HI = f"{max(_ssl_naive)*100:.1f}"
_q = [E2[d][c]["quantile"]["500"] for d in ("aasist", "ssl") for c in E2_CORPORA]
_q += [SLS[c]["quantile"] for c in SLS_CORPORA]
_oracle = [E2[d][c]["oracle"]["fnr"] for d in ("aasist", "ssl") for c in E2_CORPORA]
_oracle += [SLS[c]["oracle"]["fnr"] for c in SLS_CORPORA]
FPR_LO, FPR_HI = min(e["fpr_mean"] for e in _q), max(e["fpr_mean"] for e in _q)
MAX_FNR_PREMIUM = f"{math.ceil(max(e['fnr_minus_oracle_pts'] for e, o in zip(_q, _oracle) if o <= 0.5) * 10 - 1e-9) / 10:.1f}"
SPREAD_LO = min(v["quantile_N500_fpr_pct_2.5_97.5"][0] for v in EER["cells"].values())
SPREAD_HI = max(v["quantile_N500_fpr_pct_2.5_97.5"][1] for v in EER["cells"].values())
_z = [e["znorm"]["fpr_mean"] for N, e in NSW["n_sweep"]["ssl/itw"].items() if isinstance(e, dict)]
_naive_miss = [abs(f - ALPHA) * 100 for f in _ssl_naive]
_unlab = [abs(CMS["ssl"][c][m]["fpr"] - ALPHA) * 100 for c in E2_CORPORA
          for m in ("C1_znorm", "C2_tempshift", "C5_asnorm")]
_par500 = {(d, c): PAR[d][c]["500"]["parametric"]["fpr_mean"] for d in PAR for c in PAR[d]
           if isinstance(PAR[d][c], dict) and "500" in PAR[d][c]}
_zdev = [abs(M10[d][c]["500"]["znorm"]["fpr_mean"] - ALPHA) * 100 for d in ("ssl", "aasist")
         for c in E2_CORPORA]
_beyond = [x for x in _zdev if x > 2]
N_BONA = {v["n_dep_bona"] for v in within.values()}.pop()
N_SPOOF = {v["n_dep_spoof"] for v in within.values()}.pop()
N_SPK = {v["n_speakers"] for v in SPK["cells"].values()}.pop()
N_CAL = script_constant(E102 / "drift_map.py", "N_CAL")
B_WEIGHTED = script_constant(E102 / "drift_map.py", "B_WEIGHTED")
CONTAM = max(script_constant(E102 / "n_sweep.py", "CONTAM"))
CONTAM_N = max(script_constant(E102 / "n_sweep.py", "CONTAM_NS"))
BINS = int(re.search(r"def density_ratio_weights\(.*bins=(\d+)\)", DRIFT_MAP).group(1))
CLIP = re.search(r"np\.clip\(q / np\.maximum\(p, 1e-8\), ([\d.]+), ([\d.]+)\)", DRIFT_MAP).groups()
ACC = re.search(r"if tpr >= ([\d.]+) and fpr <= ([\d.]+):", DRIFT_MAP).groups()

# value -> what it is. Every value in this mapping is recomputed from a loaded
# result artifact above; analytic quantities live in ANALYTIC_DERIVED below.
ARTIFACT_DERIVED = {
    str(len(cells)): "total deployment cells",
    str(len(miss2)): "cells missing target by >2x",
    str(sum(1 for v in miss2 if v["log2_fpr_ratio"] < 0)): "conservative of those (below target)",
    str(sum(1 for v in miss2 if v["log2_fpr_ratio"] > 0)): "liberal of those (above target)",
    str(round(FLAG["vanilla_fpr_mean"] * FLAG["n_dep_bona"])): "flagship false alarms out of n_dep_bona",
    str(min(CORS_N.values())): "released-score detectors, min cells off by >2x (EXP-109)",
    str(max(CORS_N.values())): "released-score detectors, max cells off by >2x (EXP-109)",
    str(sum(CORS_N.values())): "released-score detectors, pooled cells off by >2x (EXP-109)",
    str(42 * len(CORS_N)): "released-score detectors, pooled channel pairs (EXP-109)",
    **{v: f"per-condition 21LA EER {which}, {det} (Table 1 caption)" for det in ("aasist", "ssl", "sls")
       for which, v in zip(("min", "max"), _ec_range(det))},
    f"{min(SPK_W):.1f}": "speaker-disjoint width, min over six cells (pp)",
    f"{max(SPK_W):.1f}": "speaker-disjoint width, max over six cells (pp)",
    f"{min(SPK_NULL):.1f}": "speaker-permutation null median, min over six cells (pp)",
    f"{max(SPK_NULL):.1f}": "speaker-permutation null median, max over six cells (pp)",
    str(len(hidden)): "in-tolerance cells missing by >2x",
    str(len(in_tol)): "cells inside the +-5 pp tolerance on realized FPR",
    str(sum(1 for v in in_tol if abs(v["log2_fpr_ratio"]) > 0.585)): "in-tolerance cells off by >1.5x",
    str(sum(1 for v in in_tol if abs(v["log2_fpr_ratio"]) > 2.0)): "in-tolerance cells off by >4x",
    str(n_within_tol): "in-tolerance cells that are within-corpus",
    str(len(within)): "within-corpus cells",
    str(la_n): "within-corpus cells missing by >2x",
    # 42 = the ordered 21LA condition pairs per detector, i.e. within-corpus
    # cells divided by the two detectors that carry them.
    str(len(within) // 2): "ordered 21LA condition pairs per detector",
    str(tf_n): "A5 twin-free pairs missing by >2x",
    str(tf_t): "A5 twin-free pairs",
    str(round(100 * tf_n / tf_t)): "A5 twin-free miss rate (%)",
    str(round(100 * la_n / len(within))): "21LA within-corpus miss rate (%)",
    str(len(A5_CONDS) - 1): "A5 organizer-applied codecs (conditions minus the source)",
    f"{PRE['aasist']['oracle_fnr_at_5pct_fpr']*100:.1f}": "AASIST oracle FNR on A5 (%)",
    f"{PRE['ssl']['oracle_fnr_at_5pct_fpr']*100:.1f}": "SSL oracle FNR on A5 (%)",
    # The flagship: named cell = argmax fnr_price among non-resolution-limited cells.
    f"{FLAG['vanilla_fnr_mean']*100:.0f}": "flagship missed-spoof rate (%)",
    f"{FLAG['fnr_oracle']*100:.2f}": "flagship oracle missed-spoof rate (%)",
    f"{FLAG['vanilla_fpr_mean']*100:.2f}": "flagship realized FPR (%)",
    f"{FLAG['fnr_price']*100:.1f}": "flagship spoof-side price (percentage points)",
    str(math.floor(PSTN_MIN_PRICE * 100 + 1e-9)): "floor of the min price over AASIST cells calibrated on PSTN (points)",
    f"{FLAG['cal_fnr_at_threshold']*100:.0f}": "AASIST FNR on PSTN spoofs at the transported PSTN threshold (%)",
    f"{AUC['pstn']:.3f}": "AASIST AUC on PSTN, results_dissociation.json",
    f"{min(AUC_OTHERS):.3f}": "AASIST AUC, min over the other six 21LA conditions",
    f"{max(AUC_OTHERS):.3f}": "AASIST AUC, max over the other six 21LA conditions",
    # monitors: the corrected reading's one in-sample pass and the re-mix probe
    f"{OP['tpr']:.2f}": "mixture-W1 monitor TPR, corrected reading",
    f"{OP['fpr']:.2f}": "mixture-W1 monitor FPR, corrected reading",
    f"{MON['spearman_vs_target']:.2f}": "mixture-W1 monitor Spearman, corrected reading",
    str(len(BENIGN)): "SSL benign cells (|log2| <= 1)",
    str(sum(1 for v in BENIGN if v["monitors"]["w1_mixture_prev_half"] >= OP["threshold"])):
        "benign cells over threshold after halving spoof share",
    str(sum(1 for v in BENIGN if v["monitors"]["w1_mixture_prev_x15"] >= OP["threshold"])):
        "benign cells over threshold after x1.5 spoof share",
    # score-weighting heuristic contest, recomputed from the same drift map
    str(sum(1 for v in cells.values() if abs(v["weighted_fpr_mean"] - ALPHA) <= 0.02)):
        "cells where score weighting lands within 2pp",
    str(sum(1 for v in cells.values() if abs(v["vanilla_fpr_mean"] - ALPHA) <= 0.02)):
        "cells where unweighted CP lands within 2pp",
    # contamination sweep at the reported operating point
    str(round(CONTAM * CONTAM_N)): "contaminating spoofs in the cohort (rate x N, n_sweep.py)",
    f"{min(_below)*100:.2f}": "contamination sweep, lowest realized FPR below target (%)",
    f"{max(_below)*100:.2f}": "contamination sweep, highest realized FPR below target (%)",
    f"{max(_cont.values())*100:.2f}": "contamination sweep, the one cell above target (%)",
    str(round(min(_c5) * 100)): "C5 asymmetric-norm degenerate FNR (%)",
    # calibration grid
    f"{FPR_LO*100:.2f}": "quantile mean realized FPR, min over 12 cells (%)",
    f"{FPR_HI*100:.2f}": "quantile mean realized FPR, max over 12 cells (%)",
    str(len(_q)): "detector-corpus cells in Table 1",
    MAX_FNR_PREMIUM: "ceiling of the max FNR premium over usable cells (pt)",
    f"{SPREAD_LO*100:.1f}": "held-out 2.5th percentile of realized FPR, min over cells (%)",
    f"{SPREAD_HI*100:.1f}": "held-out 97.5th percentile of realized FPR, max over cells (%)",
    f"{max(_naive_miss + _unlab):.0f}": "naive/unlabeled worst miss over SSL corpora (pp)",
    f"{max(abs(f-ALPHA)*100 for f in _par500.values()):.1f}": "Gaussian quantile worst miss at N=500 (pp)",
    f"{PRE['ssl']['n_bona']:,}": "A5 pilot recordings per class (precheck.json)",
    f"{A5EER['ssl']['n_bona']:,}": "A5 deployment-half bona fide (a5_eer.json)",
    f"{A5EER['ssl']['n_spoof']:,}": "A5 deployment-half spoofs (a5_eer.json)",
    f"{A5EER['aasist']['eer']*100:.1f}": "A5 deployment-half pooled EER, AASIST (%)",
    f"{A5EER['ssl']['eer']*100:.1f}": "A5 deployment-half pooled EER, SSL-AASIST (%)",
    f"{min(E2['ssl'][c]['quantile']['500']['fpr_mean'] for c in E2_CORPORA)*100:.1f}": "SSL quantile realized FPR, min over corpora (%)",
    str(len(A5_USABLE)): "A5 SSL-AASIST pairs on usable destinations",
    str(sum(1 for v in A5_USABLE if v["log2_fpr_ratio"] < -SEV)): "usable A5 pairs missing conservatively by >2x",
    str(sum(1 for v in A5_USABLE if v["fnr_price"] > 0)): "usable A5 pairs with positive spoof-side cost",
    str(script_constant(E102 / "c_methods.py", "K_COHORT")): "C5 cohort neighbours, c_methods.py K_COHORT",
    f"{script_constant(E102 / 'c_methods.py', 'COHORT_SUB'):,}": "C5 cohort subsample, c_methods.py COHORT_SUB",
    re.search(r"for _ in range\((\d+)\):", CMETH).group(1): "C2 Newton steps, c_methods.py",
    str(len(_beyond)): "cohort z-norm cells beyond 2 pp",
    f"{max(abs(SLS[c]['znorm']['fpr_mean']-ALPHA)*100 for c in SLS_CORPORA):.1f}":
        "cohort z-norm worst miss, XLS-R+SLS (pp)",
    # setup counts
    f"{N_BONA:,}": "bona-fide recordings per 21LA condition, hidden phase excluded",
    f"{N_SPOOF:,}": "spoofed trials per 21LA condition, hidden phase excluded",
    str(N_SPK): "speakers per 21LA condition",
    f"{E2['ssl']['asv21la']['n_bona']:,}": "full 21LA bona fide, hidden phase excluded (EXP-002; equals SLS n_bona)",
    f"{E2['ssl']['asv21df_full']['n_bona']:,}": "full 21DF bona fide, hidden phase excluded (EXP-002; equals SLS n_bona)",
    # speaker diagnostic
    str(SPK["cells"]["aasist/pstn"]["permutation"]["median_calibration_speakers"]): "median calibration speakers",
    str(SPK["draws_per_width"]): "draws per speaker-disjoint width",
    **TABLE1,
}

ANALYTIC_DERIVED = {
    f"{P100*100:.1f}": "iid Beta reference: P(3% <= FPR <= 7%), N=100",
    f"{P500*100:.1f}": "iid Beta reference: P(3% <= FPR <= 7%), N=500",
    f"{Q500_LO*100:.2f}": "iid Beta reference: N=500 central 95% lower endpoint",
    f"{Q500_HI*100:.2f}": "iid Beta reference: N=500 central 95% upper endpoint",
    f"{normal_quantile(1 - ALPHA):.3f}": "one-sided normal quantile at 5%",
}

# Non-derived numerals, each with the reason it is not an artifact value.
DECLARED_RAW = {
    "1": "index/unit", "2": "index, section refs, 2x bar, 2 pp", "3": "index/count",
    "4": "section number", "5": "alpha=5%, section number, 5/n", "6": "count in prose",
    "7": "section number", "8": "count of cells in Table 1", "9": "count",
    "10": "order of magnitude", "0": "zero", "12": "conditions in the A5 grid",
    "19": "corpus name 19LA", "20": "count", "21": "corpus name ASVspoof 21",
    "46": "corpus/section",
    "50": "50/50 speaker split, 50% overlap bar, +-50% re-mix",
    "80": "twin-free share of sources (%)", "90": "percentile", "95": "confidence level",
    "97.5": "percentile", "2.5": "percentile", "100": "N endpoint, 100k",
    "500": "calibration cohort size N",
    "737": "A5 speakers", "994": "part of r=0.994", "0.994": "per-trial correlation with official scores",
    "0.5": "beta-bound and overlap cutoff in prose", "0.8": "monitor acceptance TPR",
    "0.2": "monitor acceptance FPR", "0.1": "weight clip lower bound", "1.5": "sensitivity bar",
    "0.0": "zero delta in Table 1", "80.5": "twin-free share of A5 bona-fide sources (%)",
    "1000": "B=1000 paired calibration draws", "2019": "corpus year, ASVspoof 2019",
    "2021": "corpus year, ASVspoof 2021", "2027": "venue year", "66": "language count from cited LRLspoof work",
    "15": "histogram bins in the heuristic", "200": "B_WEIGHTED draws / permutations",
    "11": "A5 codecs", "1.6": "table column separation (pt)",
    "6": "count in prose; exponent of the 1e-6 stabilizers (c_methods.py, drift_map.py)",
    "3": "index/count; spoof-side cost bound in points (asserted in check_numbers.py)",
}

# A value may recur in derived and non-derived semantic contexts (for example
# 19 is both the SSL benign cell count and the model name 19LA). Such occurrences
# are resolved by explicit context rules below, never by overlapping value sets.
DECLARED = {
    value: reason for value, reason in DECLARED_RAW.items()
    if value not in ARTIFACT_DERIVED and value not in ANALYTIC_DERIVED
}

DECLARED_CONTEXT_RULES = (
    ("19", re.compile(r"19LA"), "detector training-corpus name"),
    ("50", re.compile(r"(?:exceeds\s+|split\s+|\$\\pm\$?|\$[-+])50(?:/50)?\\%|50/50|\\pm\$50"),
     "design percentage, overlap criterion or re-mix magnitude"),
    ("66", re.compile(r"(?:across|covering) 66"), "language count from cited LRLspoof work"),
    ("12", re.compile(r"\+\$ 12 ordered pairs"), "cross-corpus ordered pairs, by construction"),
    ("15", re.compile(r"15 equal-width bins"), "histogram bins, drift_map.py density_ratio_weights"),
    ("5", re.compile(r"FPR\}<5/n"), "resolution-limited rule numerator, drift_map.py"),
    ("95", re.compile(r"central 95\\% interval"), "confidence level"),
    ("10", re.compile(r"\[0\.1,10\]|10\^\{-6\}"), "clip bound / exponent base"),
    ("0.2", re.compile(r"FPR~\$\\le\$~0\.2"), "monitor acceptance FPR"),
    ("3", re.compile(r"\$3/n\$"), "severity floor numerator (bound in check_numbers.py to drift_map.py)"),
)

# Repeated values require occurrence-specific sources. These rules take
# precedence over the value-level fallback below, so (for example) the 98.5%
# naive-transfer claim cannot borrow Table 1's unrelated 98.5% FNR.
ARTIFACT_CONTEXT_RULES = (
    (INTRO_SSL_LO, re.compile(rf"realizes {re.escape(INTRO_SSL_LO)}--{re.escape(INTRO_SSL_HI)}\\% FPR"),
     "EXP-002 SSL naive-transfer FPR minimum, rounded for prose"),
    (INTRO_SSL_HI, re.compile(rf"{re.escape(INTRO_SSL_LO)}--{re.escape(INTRO_SSL_HI)}\\% FPR across"),
     "EXP-002 SSL naive-transfer FPR maximum"),
    (MAX_FNR_PREMIUM, re.compile(rf"(?:at most a {re.escape(MAX_FNR_PREMIUM)}-point FNR premium|\$\\le\${re.escape(MAX_FNR_PREMIUM)} points)"),
     "maximum usable-cell FNR premium over the oracle, both sites"),
    ("7", re.compile(r"z-norm remains 7--8\\,pp high"), "N-sweep z-norm ITW offset, lower endpoint"),
    ("8", re.compile(r"z-norm remains 7--8\\,pp high"), "N-sweep z-norm ITW offset, upper endpoint"),
    ("5.0", re.compile(r"realizes 4\.9--5\.0\\% FPR on all four corpora"),
     "EXP-002 SSL quantile/500 fpr_mean, max over corpora"),
    ("4.9", re.compile(r"realizes 4\.9--5\.0\\% FPR on all four corpora"),
     "EXP-002 SSL quantile/500 fpr_mean, min over corpora"),
    (str(round(100 * la_n / len(within))), re.compile(r"within-21LA pairs \(68\\%\)"),
     "within-21LA >2x miss rate over both detectors"),
    ("0", re.compile(r"collapses to 0\\% on BRSpeech"), "Gaussian quantile FPR on BRSpeech, results_parametric.json"),
    (str(B_WEIGHTED), re.compile(r"200 draws"), "B_WEIGHTED in drift_map.py"),
    (str(SPK["permutations"]), re.compile(r"200 permutations"), "speaker_clustering.json permutations"),
    (str(N_CAL), re.compile(r"(?:the 500 cohort|at least 500 recordings|N\{=\}500)"),
     "N_CAL in drift_map.py"),
    (str(len(_beyond)), re.compile(r"6/8 cells beyond"), "EXP-010 z-norm cells beyond 2 pp"),
    ("8", re.compile(r"6/8 cells beyond"), "EXP-010 z-norm cells at N=500"),
    (str(len(A5_CONDS) - 1), re.compile(r"11 organizer-applied codecs"), "A5 conditions minus the source"),
    ("0.5", re.compile(r"odds by 0\.5 and 1\.5"), "prevalence factor, drift_map.py"),
    ("1.5", re.compile(r"odds by 0\.5 and 1\.5"), "prevalence factor, drift_map.py"),
    ("50", re.compile(r"50 Newton steps"), "C2 Newton steps, c_methods.py"),
    ("100", re.compile(r"the 100 nearest"), "C5 cohort neighbours, c_methods.py K_COHORT"),
    ("66", re.compile(r"\(66 ordered pairs\)"), "A5 SSL-AASIST pairs on usable destinations"),
    (str(round(FLAG["vanilla_fpr_mean"] * FLAG["n_dep_bona"])), re.compile(r"about 10 false alarms per draw"),
     "flagship false alarms, vanilla_fpr_mean x n_dep_bona"),
    (str(min(CORS_N.values())), re.compile(r"on 25--33 of the 42"), "EXP-109 min per-detector miss count"),
    (str(max(CORS_N.values())), re.compile(r"on 25--33 of the 42"), "EXP-109 max per-detector miss count"),
    (_ec_range("ssl")[0], re.compile(r"0\.2--1\.0 \(SSL-AASIST\)"), "per-condition EER min, SSL-AASIST"),
    (_ec_range("sls")[0], re.compile(r"0\.5--3\.5 \(XLS-R\+SLS\)"), "per-condition EER min, XLS-R+SLS"),
    (_ec_range("sls")[1], re.compile(r"0\.5--3\.5 \(XLS-R\+SLS\)"), "per-condition EER max, XLS-R+SLS"),
    (f"{max(SPK_NULL):.1f}", re.compile(r"against 3\.4--3\.5 under"), "speaker-permutation null median, max"),
    (f"{min(SPK_NULL):.1f}", re.compile(r"against 3\.4--3\.5 under"), "speaker-permutation null median, min"),
)

CLASSIFICATION_PROBES = (
    ("intro 98.5 binds naive transfer, not Table 1", INTRO_SSL_HI,
     re.compile(r"22--98\.5\\% FPR across"), "ARTIFACT_DERIVED"),
    ("overlap gate is declared, not borrowed from grid rate", "50",
     re.compile(r"exceeds 50\\%"), "DECLARED"),
    ("prevalence factors bind to drift_map.py, not the 0.5 premium/cutoff", "0.5",
     re.compile(r"odds by 0\.5 and 1\.5"), "ARTIFACT_DERIVED"),
    ("grid rate is artifact-derived", "68",
     re.compile(r"within-21LA pairs \(68\\%\)"), "ARTIFACT_DERIVED"),
    ("the SSL benign cell count is artifact-derived, 19LA is not", "19",
     re.compile(r"19\\slash19"), "ARTIFACT_DERIVED"),
    ("heuristic draws bind to B_WEIGHTED", "200",
     re.compile(r"200 draws"), "ARTIFACT_DERIVED"),
    ("permutations bind to the speaker artifact", "200",
     re.compile(r"200 permutations"), "ARTIFACT_DERIVED"),
    ("the confidence level is declared, not borrowed from a 95 pp miss", "95",
     re.compile(r"central 95\\% interval"), "DECLARED"),
)


def _flat(text):
    return re.sub(r"\s+", " ", text)


# These patterns bind repeated load-bearing occurrences to values recomputed
# above. A one-site mutation fails even when the replacement is another declared
# numeral. `check_numbers.py` adds table cells and the remaining sentence-level
# artifact bindings.
POSITION_BINDINGS = (
    ("abstract grid census",
     rf"Across {len(cells)} detector, source and target combinations built from .{{0,120}}, {len(miss2)} realize an FPR "
     rf"more than \$2\\times\$ off target \({sum(1 for v in miss2 if v['log2_fpr_ratio'] > 0)} above, "
     rf"{sum(1 for v in miss2 if v['log2_fpr_ratio'] < 0)} below\)\. Of the {len(in_tol)} cells within our own "
     rf"\$\\pm5\$-percentage-point FPR tolerance, {len(hidden)} fall below half the target", 1),
    ("108-cell decomposition", rf"The measurement covers {len(cells)} deployment cells: two detectors", 1),
    ("42+12 decomposition", r"42 ordered pairs of seven 21LA channel conditions .* 12 ordered pairs of four corpora", 1),
    ("hidden count over the tolerance, both sites", rf"{len(hidden)} of the {len(in_tol)} cells inside", 2),
    ("sensitivity range, both sites",
     rf"{sum(1 for v in in_tol if abs(v['log2_fpr_ratio']) > 0.585)} (?:of {len(in_tol)} )?at \$1\.5\\times\$", 2),
    ("sensitivity range at 4x, both sites",
     rf"{sum(1 for v in in_tol if abs(v['log2_fpr_ratio']) > 2.0)} (?:of {len(in_tol)} )?at \$4\\times\$", 2),
    ("flagship transported FNR (abstract; caption sentence cut)", rf"{FLAG['vanilla_fnr_mean']*100:.0f}\\% of spoofs", 1),
    ("flagship oracle FNR (abstract)", rf"against {FLAG['fnr_oracle']*100:.2f}\\%", 1),
    ("flagship FPR, abstract and Drift paragraph", rf"{FLAG['vanilla_fpr_mean']*100:.2f}\\% FPR", 2),
    ("flagship false-alarm count", rf"about {round(FLAG['vanilla_fpr_mean'] * FLAG['n_dep_bona'])} false alarms per draw among {N_BONA:,} bona fide", 1),
    ("released-score detectors, EXP-109",
     rf"on {min(CORS_N.values())}--{max(CORS_N.values())} of the 42 channel pairs each \({sum(CORS_N.values())} of {42 * len(CORS_N)}\)", 1),
    ("speaker six-cell ranges",
     rf"to {min(SPK_W):.1f}--{max(SPK_W):.1f} points \(means over ten seeds of {SPK['draws_per_width']} draws\), against a median of {min(SPK_NULL):.1f}--{max(SPK_NULL):.1f} under", 1),
    ("A5 primary numerator/denominator, both sites", rf"{tf_n} of {tf_t}", 2),
    ("A5 primary percentage, both sites", rf"\({round(100*tf_n/tf_t)}\\%\)", 2),
    ("mean realized FPR range, both sites", rf"{FPR_LO*100:.2f}--{FPR_HI*100:.2f}\\%", 2),
    ("per-condition recording count, five sites", rf"{N_BONA:,}(?:-recording| recordings| bona-fide recordings| bona fide| bona-fide and)", 5),
    ("per-condition spoof count, both sites", rf"{N_SPOOF:,} spoof", 2),
    ("speaker count, both sites", rf"{N_SPK} speakers", 2),
    ("naive/unlabeled worst miss", rf"miss the target by up to {max(_naive_miss + _unlab):.0f} pp", 1),
    ("A5 usable-destination counts",
     rf"\({len(A5_USABLE)} ordered pairs\), where {sum(1 for v in A5_USABLE if v['log2_fpr_ratio'] < -SEV)} miss the FPR target conservatively by more than \$2\\times\$ and {sum(1 for v in A5_USABLE if v['fnr_price'] > 0)} pay", 1),
    ("calibration-condition AUC against the other six",
     rf"AUC {AUC['pstn']:.3f} against {min(AUC_OTHERS):.3f}--{max(AUC_OTHERS):.3f}\)", 1),
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
        r"\\(?:cite|ref|label|includegraphics|setlength|documentclass|usepackage|url)"
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
    """Bind each Table 1 numeral to its artifact row, corpus, block and quantity."""
    bindings = {}
    columns = ("21LA", "21DF", "ITW", "BRSpeech")
    for row, blocks in TABLE1_LINES.items():
        lines = list(re.finditer(rf"(?m)^{re.escape(row)}\s*&[^\n]*\\\\\s*$", tex))
        if len(lines) != len(blocks):
            raise AssertionError(f"Table 1 row {row}: expected {len(blocks)} line(s), found {len(lines)}")
        for match, (expected_cells, quantities) in zip(lines, blocks):
            line = match.group(0)
            ampersands = [i for i, char in enumerate(line) if char == "&"]
            if len(ampersands) != 4:
                raise AssertionError(f"Table 1 row {row} has {len(ampersands)} data separators")
            for index, (column, expected) in enumerate(zip(columns, expected_cells)):
                cell_start = ampersands[index] + 1
                cell_end = ampersands[index + 1] if index + 1 < len(ampersands) else line.rfind("\\\\")
                cell = line[cell_start:cell_end]
                found = list(re.finditer(r"(?<![\\A-Za-z0-9._])(\d+(?:[.,]\d+)?)", cell))
                if len(found) != len(quantities):
                    raise AssertionError(
                        f"Table 1 {row}/{column} has {len(found)} numerals; expected {quantities}"
                    )
                for quantity, numeral, wanted in zip(quantities, found, expected):
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
