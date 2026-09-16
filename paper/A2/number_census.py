"""Every numeral in VERSION B's main.tex must resolve to an artifact or be declared.

A fork of the frozen release candidate's census for the manuscript in this
directory: the monitoring and heuristic values are gone with their
experiments, and the cost-mechanism and scalar-cost values take their place.

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
import hashlib
import json
import math
import os
import re
import statistics as stats
import sys
from collections import Counter
from pathlib import Path

HERE = Path(__file__).parent
EXP = next(p / "experiments" for p in HERE.parents if (p / "experiments").is_dir())
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
_proto = Path(os.environ.get("A2_A5_PROTO", Path(os.environ.get("A2_DATA", Path.home() / "data/corpora/anti-spoofing"))
                             / "asvspoof5/ASVspoof5.eval.track_1.tsv"))
_prows = [l.split() for l in open(_proto)]
_processed = {}
for _r in _prows:
    if _r[8] == "bonafide" and _r[5] != "-":
        _processed.setdefault(_r[5], set()).add(_r[3])
_one = {s for s, c in _processed.items() if len(c) == 1}
A5_UNPROC = sum(1 for _r in _prows if _r[8] == "bonafide" and _r[5] == "-")
A5_KEPT = sum(1 for _r in _prows if _r[8] == "bonafide" and _r[5] in _one)
A5_SHARE = 100 * len(_one) / len(_processed)
A5_SPK = len({_r[0] for _r in _prows})
A5DIAG = json.load(open(EXP / "EXP-103-a5-replicate/artifacts/a5_diagonal.json"))
DRIFT_MAP = (E102 / "drift_map.py").read_text()
CORS = json.load(open(EXP / "EXP-109-a2-cors-transport/results_cellA.json"))
MECH = json.load(open(EXP / "EXP-123-a2-cost-mechanism/results.json"))["detectors"]
MECH_A5 = json.load(open(EXP / "EXP-123-a2-cost-mechanism/results_a5.json"))["detectors"]["ssl"]
DEGR = json.load(open(EXP / "EXP-125-a2-dcf-axis/results_degradation.json"))
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
# Cells whose spoof-side cost exceeds the illustrated one; both are below the 5/n
# resolution floor, which is why the illustrated cell is not the grid maximum.
ABOVE_FLAG = [v for v in cells.values() if v["fnr_price"] > FLAG["fnr_price"]]
B_DRAWS = script_constant(E102 / "drift_map.py", "B")
N_CAL_DRAW = script_constant(E102 / "drift_map.py", "N_CAL")
# The seeded speaker-ID hash of EXP-103 analyze.py half(), recomputed: an independent
# assignment, so the two groups of the twin-free roster are unequal.
_a5seed = script_constant(EXP / "EXP-103-a5-replicate/analyze.py", "SEED")
_a5assign = {}
A5_GRP_TRIALS = {0: 0, 1: 0}
for _r in _prows:
    if _r[8] != "bonafide" or not (_r[5] == "-" or _r[5] in _one):
        continue
    _h = int(hashlib.md5(f"{_a5seed}:{_r[0]}".encode()).hexdigest(), 16) & 1
    _a5assign[_r[0]] = _h
    A5_GRP_TRIALS[_h] += 1
A5_GRP = {0: sum(1 for _h in _a5assign.values() if _h == 0),
          1: sum(1 for _h in _a5assign.values() if _h == 1)}
# Table 2 splits the 66 usable-destination pairs on whether the calibration condition is
# itself usable. Recomputed here from the released pair results, exactly as the manuscript
# defines usability: oracle FNR at or below the bar for every source into that destination.
_a5tf = A5["ssl/twin_free"]
_a5dest = {}
for _k, _v in _a5tf.items():
    _a5dest.setdefault(_k.split("->")[1], []).append(_v["fnr_oracle"])
_a5usable = {_c: max(_o) <= 0.50 for _c, _o in _a5dest.items()}
_a5rows = [(_a5usable.get(_k.split("->")[0], False), _v)
           for _k, _v in _a5tf.items() if _a5usable.get(_k.split("->")[1])]
_A5_SPLIT = {}
for _flag, _tag in ((True, "usable"), (False, "overlap-dominated")):
    _g = [_v for _f, _v in _a5rows if _f == _flag]
    _cons = sum(1 for _v in _g if _v["log2_fpr_ratio"] < -SEV)
    _med = stats.median(_v["fnr_price"] * 100 for _v in _g)
    _A5_SPLIT[str(len(_g))] = f"A5 usable-destination pairs with a {_tag} source"
    _A5_SPLIT[str(_cons)] = f"conservative misses among them ({_tag} source)"
    _A5_SPLIT[f"{_med:.1f}"] = f"median spoof-side cost, {_tag} source (pp)"
_A5_SPLIT[str(sum(1 for _f, _v in _a5rows if _f and _v["fnr_price"] * 100 > 10))] = \
    "usable-source pairs above +10 pp"
assert sum(A5_GRP_TRIALS.values()) == A5_UNPROC + A5_KEPT, (A5_GRP_TRIALS, A5_UNPROC + A5_KEPT)
# The cost mechanism over the 42 ordered pairs, and the scalar-cost axis.
MECH_RHO = {d: MECH[d]["rho"]["source_oracle_fnr"] for d in MECH}
MECH_P = {d: MECH[d]["exact_p"]["source_oracle_fnr"] for d in MECH}
MECH_SIGN = {d: MECH[d]["sign_table"] for d in MECH}
MECH_APPLICABLE = MECH_SIGN["aasist"]["pred_cons_and_cons"] + MECH_SIGN["aasist"]["pred_cons_and_lib"]
PRIORS = [p for p in DEGR["priors"] if p <= 0.1]


def _degradation_counts():
    """Conservative cells, and how many accept more spoofs than their calibration
    condition. The threshold rule is prior-independent, so the two low priors
    recover FPR and FNR separately: DCF/pi = FNR + ((1-pi)/pi)*FPR."""
    cons = more = 0
    for dv in DEGR["detectors"].values():
        for cv in dv["cells"].values():
            if cv["log2_fpr_ratio"] >= -SEV:
                continue
            cons += 1
            rates = {}
            for side in ("deployment", "calibration"):
                low = cv["by_prior"][f"{PRIORS[0]}"][f"v2_dcf_{side}"]
                half = cv["by_prior"]["0.5"][f"v2_dcf_{side}"]
                fpr = (low - half) / (1 / PRIORS[0] - 2)
                rates[side] = half - fpr
            more += rates["deployment"] > rates["calibration"]
    return cons, more


CONS_CELLS, CONS_MORE_SPOOFS = _degradation_counts()
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


DETECTOR_ROWS = ("AASIST", "SSL-AASIST", "XLS-R+SLS$^\\dagger$")
_DET_KEY = {"AASIST": "aasist", "SSL-AASIST": "ssl", "XLS-R+SLS$^\\dagger$": "sls"}


def _corpora(row):
    return SLS_CORPORA if row.startswith("XLS-R+SLS") else E2_CORPORA


def _quantile_entry(row, corpus):
    return SLS[corpus]["quantile"] if row.startswith("XLS-R+SLS") else E2[_DET_KEY[row]][corpus]["quantile"]["500"]


# Table 1, version B: four blocks, each a \multicolumn header followed by its
# rows. The policy block prints two errors per cell as "FPR/FNR", so a cell is an
# ordered pair: swapping its halves is a different table, and the parser binds
# each half to its own artifact.
def _policy_pair(fpr, fnr, fnr_places):
    return (f"{fpr*100:.1f}", f"{fnr*100:.{fnr_places}f}")


TABLE1_BLOCKS = (
    ("Pooled EER (\\%)", ("EER",),
     {row: tuple((f"{EER['cells'][f'{_DET_KEY[row]}/{c}']['eer']*100:.2f}",) for c in _corpora(row))
      for row in DETECTOR_ROWS}),
    ("Conformal FNR (\\%) and oracle difference (pp)", ("FNR", "oracle difference"),
     {row: tuple(_table_pair(_quantile_entry(row, c)) for c in _corpora(row)) for row in DETECTOR_ROWS}),
    ("Realized FPR (\\%) at the conformal quantile, same run", ("realized FPR",),
     {row: tuple((f"{_quantile_entry(row, c)['fpr_mean']*100:.2f}",) for c in _corpora(row))
      for row in DETECTOR_ROWS}),
    ("SSL-AASIST policies: FPR/FNR (\\%)", ("policy FPR", "policy FNR"), {
        "Naive transfer": tuple(_policy_pair(E2["ssl"][c]["naive_transfer"]["fpr"],
                                             E2["ssl"][c]["naive_transfer"]["fnr"], 2) for c in E2_CORPORA),
        "C1 z-norm": tuple(_policy_pair(CMS["ssl"][c]["C1_znorm"]["fpr"],
                                        CMS["ssl"][c]["C1_znorm"]["fnr"], 2) for c in E2_CORPORA),
        "Cohort z-norm": tuple(_policy_pair(M10["ssl"][c]["500"]["znorm"]["fpr_mean"],
                                            M10["ssl"][c]["500"]["znorm"]["fnr_mean"], 1) for c in E2_CORPORA),
        "Gaussian q.": tuple(_policy_pair(PAR["ssl"][c]["500"]["parametric"]["fpr_mean"],
                                          E2["ssl"][c]["oracle"]["fnr"]
                                          + PAR["ssl"][c]["500"]["parametric"]["fnr_minus_oracle_pts"] / 100,
                                          1) for c in E2_CORPORA),
    }),
)
TABLE1 = {}
for _header, _quantities, _rows in TABLE1_BLOCKS:
    for _row, _cells in _rows.items():
        for _column, _values in zip(("21LA", "21DF", "ITW", "BRSpeech"), _cells):
            for _value, _quantity in zip(_values, _quantities):
                _reason = f"Table 1 {_quantity}, {_row}/{_column}"
                TABLE1[_value] = f"{TABLE1[_value]}; {_reason}" if _value in TABLE1 else _reason

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
# C2 and C5 were retired from the manuscript on 2026-09-12; the claim they supported is
# now about C1 alone.
_unlab = [abs(CMS["ssl"][c]["C1_znorm"]["fpr"] - ALPHA) * 100 for c in E2_CORPORA]
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
# The conclusion's one-clause factorial statement, recomputed from the six run readings.
_F110C = EXP / "EXP-110-a2-channel-training"
_f110c = {r: json.load(open(_F110C / f"results_{r}.json"))
          for r in ("arm1", "arm1_s1235", "arm1_s1236", "arm2", "arm2_s1235", "arm2_s1236")}
_m1c = [int(_f110c[r]["K_median"]) for r in ("arm1", "arm1_s1235", "arm1_s1236")]
_m2c = [int(_f110c[r]["K_median"]) for r in ("arm2", "arm2_s1235", "arm2_s1236")]
# The AASIST arms, one seed each, recomputed the same way as the seed factorial.
_AAC = {a: json.load(open(_F110C / f"results_aasist_{a}_s1234.json")) for a in ("arm1", "arm2")}
_aacost = {a: max(c.get("fnr_price", 0) * 100 for ep in v["epochs"].values() for c in ep["cells"].values())
           for a, v in _AAC.items()}
_aaover = {a: 100 * sum(1 for ep in v["epochs"].values() for c in ep["cells"].values()
                        if c.get("fnr_price", 0) * 100 > 10)
              / sum(len(ep["cells"]) for ep in v["epochs"].values()) for a, v in _AAC.items()}
_sslcost = {}
for _arm, _rs in (("arm1", ("arm1", "arm1_s1235", "arm1_s1236")), ("arm2", ("arm2", "arm2_s1235", "arm2_s1236"))):
    _sslcost[_arm] = max(c.get("fnr_price", 0) * 100 for r in _rs
                         for ep in json.load(open(_F110C / f"results_{r}.json"))["epochs"].values()
                         for c in ep["cells"].values())
_F110_AASIST = {
    str(int(_AAC["arm1"]["K_median"])): "AASIST control K median (one seed)",
    str(int(_AAC["arm2"]["K_median"])): "AASIST matched K median (one seed)",
    f"{_sslcost['arm1']:.2f}": "SSL-AASIST control, largest excess FNR (pp)",
    f"{_sslcost['arm2']:.2f}": "SSL-AASIST matched, largest excess FNR (pp)",
    f"{_aacost['arm1']:.1f}": "AASIST control, largest excess FNR (pp)",
    f"{_aacost['arm2']:.1f}": "AASIST matched, largest excess FNR (pp)",
    f"{_aaover['arm1']:.0f}": "AASIST control, share of cells above 10 pp (%)",
    f"{_aaover['arm2']:.0f}": "AASIST matched, share of cells above 10 pp (%)",
}

_F110_CLAUSE = {str(min(_m1c)): "baseline arm K median, lowest of three seeds",
                str(max(_m1c)): "baseline arm K median, highest of three seeds",
                str(min(_m2c)): "matched arm K median, lowest of three seeds",
                str(max(_m2c)): "matched arm K median, highest of three seeds",
                "3": "seeds per arm in the released factorial"}

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
    str(len(cells) - len(within)): "cross-corpus cells",
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
    # the cost mechanism, EXP-123
    f"{MECH_RHO['aasist']:.2f}": "cost/source-oracle-FNR Spearman, AASIST",
    f"{MECH_RHO['ssl']:.2f}": "cost/source-oracle-FNR Spearman, SSL-AASIST",
    f"{MECH_RHO['xlsr_sls']:.2f}": "cost/source-oracle-FNR Spearman, XLS-R+SLS",
    f"{abs(MECH_RHO['xlsr_mamba']):.2f}": "cost/source-oracle-FNR Spearman, XLSR-Mamba (absent)",
    f"{MECH_RHO['xlsr_conformer']:.2f}": "cost/source-oracle-FNR Spearman, XLSR-Conformer (below the bar)",
    f"{MECH_P['aasist']:.4f}": "exact permutation p, AASIST",
    f"{MECH_P['ssl']:.4f}": "exact permutation p, SSL-AASIST",
    f"{MECH_P['xlsr_sls']:.4f}": "exact permutation p, XLS-R+SLS",
    f"{MECH_P['xlsr_conformer']:.3f}": "exact permutation p, XLSR-Conformer",
    f"{MECH_A5['rho']['source_oracle_fnr']:.2f}": "cost/source-oracle-FNR Spearman on the A5 pairs",
    str(MECH_SIGN["aasist"]["pred_cons_and_cons"]): "direction predicted and observed, AASIST",
    str(MECH_SIGN["ssl"]["pred_cons_and_cons"]): "direction predicted and observed, SSL-AASIST",
    str(MECH_SIGN["xlsr_sls"]["pred_cons_and_cons"]): "direction predicted and observed, XLS-R+SLS",
    str(MECH_APPLICABLE): "cells the direction prediction applies to, per detector",
    # the scalar-cost axis, EXP-125
    f"{PRIORS[0]:.2f}": "spoof prior, results_degradation.json",
    f"{PRIORS[1]:.2f}": "second spoof prior, results_degradation.json",
    str(CONS_CELLS): "conservative cells on the cost axis",
    str(CONS_MORE_SPOOFS): "of those, accepting more spoofs than the calibration condition",
    # score-weighting heuristic contest, recomputed from the same drift map
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
    f"{max(abs(f-ALPHA)*100 for f in _par500.values()):.1f}": "Gaussian quantile worst miss at N=500 (pp)",
    f"{PRE['ssl']['n_bona']:,}": "A5 pilot recordings per class (precheck.json)",
    f"{A5_UNPROC:,}": "A5 unprocessed bona-fide recordings (protocol via analyze.py build())",
    f"{A5_KEPT:,}": "A5 processed versions of one-codec sources (protocol)",
    f"{A5_UNPROC + A5_KEPT:,}": "A5 twin-free bona-fide trials (protocol)",
    f"{A5_SHARE:.1f}": "A5 share of one-codec sources (%)",
    str(A5_SPK): "A5 speakers (protocol)",
    f"{A5EER['ssl']['n_bona']:,}": "A5 deployment-half bona fide (a5_eer.json)",
    f"{A5EER['ssl']['n_spoof']:,}": "A5 deployment-half spoofs (a5_eer.json)",
    f"{A5EER['aasist']['eer']*100:.1f}": "A5 deployment-half pooled EER, AASIST (%)",
    f"{A5EER['ssl']['eer']*100:.1f}": "A5 deployment-half pooled EER, SSL-AASIST (%)",
    f"{min(E2['ssl'][c]['quantile']['500']['fpr_mean'] for c in E2_CORPORA)*100:.1f}": "SSL quantile realized FPR, min over corpora (%)",
    str(len(A5_USABLE)): "A5 SSL-AASIST pairs on usable destinations",
    str(sum(1 for v in A5_USABLE if v["log2_fpr_ratio"] < -SEV)): "usable A5 pairs missing conservatively by >2x",
    str(sum(1 for v in A5_USABLE if v["fnr_price"] > 0)): "usable A5 pairs with positive spoof-side cost",
    f"{stats.median(100 * v['fnr_price'] for v in A5_USABLE):.1f}": "median spoof-side cost over usable A5 pairs (pp)",
    str(sum(1 for v in A5_USABLE if 100 * v["fnr_price"] > 10)): "usable A5 pairs with cost above 10 pp",
    str(len(A5["ssl/twin_free"])): "ASVspoof 5 SSL-AASIST pairs (figure caption)",
    **{f"{x*100:.1f}": f"A5 same-condition control FPR {w}, {d} (a5_diagonal.json)" for d in ("ssl", "aasist")
       for w, x in zip(("min", "max"), A5DIAG[d]["fpr_range"])},
    str(len(_beyond)): "cohort z-norm cells beyond 2 pp",
    f"{max(abs(SLS[c]['znorm']['fpr_mean']-ALPHA)*100 for c in SLS_CORPORA):.1f}":
        "cohort z-norm worst miss, XLS-R+SLS (pp)",
    # setup counts
    f"{N_BONA:,}": "bona-fide recordings per 21LA condition, hidden phase excluded",
    f"{N_SPOOF:,}": "spoofed trials per 21LA condition, hidden phase excluded",
    str(N_SPK): "speakers per 21LA condition",
    f"{E2['ssl']['asv21la']['n_bona']:,}": "full 21LA bona fide, hidden phase excluded (EXP-002; equals SLS n_bona)",
    f"{E2['ssl']['brspeech_test']['n_bona']:,}": "bona fide read from the CML-TTS test manifest (EXP-002)",
    f"{E2['ssl']['brspeech_test']['n_spoof']:,}": "BRSpeech-DF test spoofs (EXP-002)",
    f"{E2['ssl']['asv21df_full']['n_bona']:,}": "full 21DF bona fide, hidden phase excluded (EXP-002; equals SLS n_bona)",
    # speaker diagnostic
    str(SPK["cells"]["aasist/pstn"]["permutation"]["median_calibration_speakers"]): "median calibration speakers",
    str(SPK["draws_per_width"]): "draws per speaker-disjoint width",
    # the two resolution-limited cells whose spoof-side cost exceeds the illustrated one
    **{f"{v['fnr_price'] * 100:.1f}": "resolution-limited cell above the illustrated spoof-side cost (pp)"
       for v in ABOVE_FLAG},
    # the realized groups of the A5 seeded speaker-ID assignment
    str(A5_GRP[0]): "A5 twin-free calibration group, speakers",
    str(A5_GRP[1]): "A5 twin-free deployment group, speakers",
    f"{A5_GRP_TRIALS[0]:,}": "A5 twin-free calibration group, bona-fide trials",
    f"{A5_GRP_TRIALS[1]:,}": "A5 twin-free deployment group, bona-fide trials",
    # The source-usability split of the 66 usable-destination A5 pairs (Table 2).
    **_A5_SPLIT,
    **_F110_CLAUSE,
    **_F110_AASIST,
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
    "1234": "training seed, EXP-110 run configuration",
    "1235": "training seed, EXP-110 run configuration",
    "1236": "training seed, EXP-110 run configuration",
    "0.5": "beta-bound and overlap cutoff in prose", "1.5": "sensitivity bar",
    "0.0": "zero delta in Table 1",
    "1000": "B=1000 paired calibration draws", "2019": "corpus year, ASVspoof 2019",
    "2021": "corpus year, ASVspoof 2021", "2027": "venue year", "66": "language count from cited LRLspoof work",
    "200": "speaker-permutation count",
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

# The acknowledgment carries a grant identifier, not a measurement. It is
# declared for that block alone -- docs/icassp2027-submission-format.md S4 keeps
# the funding sentence verbatim -- and the rule is keyed on its own words so the
# exemption cannot cover a numeral anywhere else.
GRANT_RULE = re.compile(r"MCTI grant 057/2023, signed with EMBRAPII")
DECLARED_CONTEXT_RULES = (
    ("057", GRANT_RULE, "AKCIT/PPI IoT grant number, format document S4"),
    ("2023", GRANT_RULE, "AKCIT/PPI IoT grant year, format document S4"),
    ("19", re.compile(r"19LA"), "detector training-corpus name"),
    ("21", re.compile(r"21LA|21DF|ASVspoof 2021"), "corpus or condition name"),
    ("50", re.compile(r"(?:exceeds\s+|split\s+|\$\\pm\$?|\$[-+])50(?:/50)?\\%|50/50|\\pm\$50"),
     "design percentage, overlap criterion or re-mix magnitude"),
    ("66", re.compile(r"(?:across|covering) 66"), "language count from cited LRLspoof work"),
    ("12", re.compile(r"\+\$ 12 ordered pairs"), "cross-corpus ordered pairs, by construction"),
    ("15", re.compile(r"15 equal-width bins"), "histogram bins, drift_map.py density_ratio_weights"),
    ("5", re.compile(r"FPR\}<5/n"), "resolution-limited rule numerator, drift_map.py"),
    ("95", re.compile(r"central 95\\% interval"), "confidence level"),
    ("040", re.compile(r"7!=5\{,\}040"), "7! = 5,040 source-condition permutations (analytic)"),
    ("5", re.compile(r"7!=5\{,\}040"), "7! = 5,040 source-condition permutations (analytic)"),
    ("0.8", re.compile(r"\{0\.8pt\}"), "table column separation (pt), \\tabcolsep"),
    ("10", re.compile(r"\[0\.1,10\]|10\^\{-6\}|above \$\+10\$"), "clip bound / exponent base / cost threshold in a5_usable_cost.py"),
    ("12", re.compile(r"10/12 including XLS-R\+SLS"), "z-norm cells over three detectors"),
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
    (str(SPK["permutations"]), re.compile(r"200 permutations"), "speaker_clustering.json permutations"),
    (f"{script_constant(E102 / 'drift_map.py', 'B'):,}", re.compile(r"1,000 random cohorts of"),
     "B in drift_map.py (abstract flagship average)"),
    (str(N_CAL), re.compile(r"(?:the 500 cohort|at least 500 recordings|N\{=\}500|cohorts of 500 recordings)"),
     "N_CAL in drift_map.py"),
    (str(len(_beyond)), re.compile(r"6/8 AASIST and SSL-AASIST cells"), "EXP-010 z-norm cells beyond 2 pp"),
    ("8", re.compile(r"6/8 AASIST and SSL-AASIST cells"), "EXP-010 z-norm cells at N=500"),
    (str(len(_beyond) + sum(1 for c in SLS_CORPORA if abs(SLS[c]["znorm"]["fpr_mean"] - ALPHA) > 0.02)),
     re.compile(r"10/12 including XLS-R\+SLS"), "z-norm cells beyond 2 pp over three detectors"),
    (str(len(A5_CONDS) - 1), re.compile(r"eleven codec conditions"), "A5 conditions minus the source"),
    # Version B uses several of these values for two different quantities; each
    # occurrence is bound by its own sentence, never by the value alone.
    (str(MECH_APPLICABLE), re.compile(r"applicable cells"),
     "direction count or the cells the prediction applies to, EXP-123 sign_table"),
    (f"{FLAG['vanilla_fpr_mean']*100:.2f}", re.compile(r"0\.42\\% FPR"), "flagship realized FPR (%)"),
    (f"{MECH_RHO['xlsr_conformer']:.2f}", re.compile(r"XLSR-Conformer"), "XLSR-Conformer cost/weakness Spearman"),
    (f"{FLAG['cal_fnr_at_threshold']*100:.0f}", re.compile(r"misses 31\\% of PSTN"),
     "AASIST FNR on PSTN spoofs at the transported threshold (%)"),
    (str(CONS_CELLS), re.compile(r"of those 31 cells"), "conservative cells on the cost axis"),
    (str(sum(1 for v in in_tol if abs(v["log2_fpr_ratio"]) > 2.0)), re.compile(r"29 of 72|of 72 at \$4"),
     "in-tolerance cells off by >4x"),
    (str(CONS_MORE_SPOOFS), re.compile(r"while 29 of those"),
     "conservative cells that accept more spoofs than their calibration condition"),
    (f"{PRIORS[0]:.2f}", re.compile(r"spoof priors \$\\pi=0\.05\$ and"), "spoof prior, results_degradation.json"),
    (f"{abs(MECH_RHO['xlsr_mamba']):.2f}", re.compile(r"XLSR-Mamba \(\$-0\.05\$\)"),
     "XLSR-Mamba cost/weakness Spearman (absent)"),
    ("0.5", re.compile(r"odds by 0\.5 and 1\.5"), "prevalence factor, drift_map.py"),
    ("1.5", re.compile(r"odds by 0\.5 and 1\.5"), "prevalence factor, drift_map.py"),
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
    ("grid rate is artifact-derived", "68",
     re.compile(r"within-21LA pairs \(68\\%\)"), "ARTIFACT_DERIVED"),
    ("the direction count is artifact-derived, 19LA is not", "19",
     re.compile(r"on 18, 19 and 21"), "ARTIFACT_DERIVED"),
    ("the corpus name 21LA is declared, not the applicable-cell count", "21",
     re.compile(r"Each 21LA condition is then"), "DECLARED"),
    ("the flagship FPR is not the XLSR-Conformer correlation", "0.42",
     re.compile(r"0\.42\\% FPR is about"), "ARTIFACT_DERIVED"),
    ("the PSTN spoof-miss rate is not the cost-axis cell count", "31",
     re.compile(r"misses 31\\% of PSTN"), "ARTIFACT_DERIVED"),
    # The 4x sensitivity count went with the withdrawn tolerance analysis, so 29 now
    # has one meaning left: the cells whose target FNR exceeds their source FNR.
    ("the cost-axis spoof-acceptance count binds to the drift artifact", "29",
     re.compile(r"29 have higher target than source FNR"), "ARTIFACT_DERIVED"),
    ("the spoof prior is not alpha and not the XLSR-Mamba correlation", "0.05",
     re.compile(r"spoof priors \$\\pi=0\.05\$ and 0\.10"), "ARTIFACT_DERIVED"),
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
     rf"Across {len(cells)} combinations of two detectors and source--target conditions, {len(miss2)} realize an FPR "
     rf"more than \$2\\times\$ off target \({sum(1 for v in miss2 if v['log2_fpr_ratio'] > 0)} above, "
     rf"{sum(1 for v in miss2 if v['log2_fpr_ratio'] < 0)} below\)", 1),
    # the transfer-family counts moved from the abstract into section 4
    ("within-corpus and cross-corpus counts (section 4)",
     rf"{sum(1 for v in within.values() if abs(v['log2_fpr_ratio']) > SEV)} of the {len(within)} within-corpus "
     rf"cells; the other {len(cells) - len(within)} are cross-corpus", 1),
    # r5 deleted the numbered contribution list that decomposed the grid; the
    # count is now bound in the abstract, in section 2's scope sentence and in
    # section 4's drift paragraph.
    ("108-cell scope in the related-work positioning", rf"we add a {len(cells)}-cell fixed-FPR map", 1),
    ("108 cells in the drift paragraph", rf"Over the {len(cells)} cells", 1),
    # The in-tolerance count and its 1.5x/4x sensitivity range were withdrawn with
    # the additive-tolerance side analysis; nothing in the manuscript reports them.
    ("usable-pair median cost (section 4; abstract clause cut)",
     rf"median cost is \$\+{stats.median(100 * v['fnr_price'] for v in A5_USABLE):.1f}\$ points", 1),
    ("usable-pair count (section 4; abstract clause cut)", rf"across all {len(A5_USABLE)} pairs", 1),
    ("flagship transported FNR and spoof count (abstract)",
     rf"misses \\textbf\{{{FLAG['vanilla_fnr_mean']*100:.0f}\\%\}} of its {FLAG['n_dep_spoof']:,} spoofs", 1),
    ("flagship oracle FNR (abstract)",
     rf"an oracle threshold on all target bona fide misses \\textbf\{{{FLAG['fnr_oracle']*100:.2f}\\%\}}", 1),
    ("flagship calibration budget and pool (abstract)",
     rf"Over {B_DRAWS:,} random cohorts of {N_CAL_DRAW} recordings drawn from {N_BONA:,} PSTN bona-fide trials", 1),
    ("the two higher resolution-limited cells (section 4)",
     "the two larger increases are "
     + " and ".join(rf"\$\+{p:.1f}\$" for p in sorted((v['fnr_price'] * 100 for v in ABOVE_FLAG), reverse=True))
     + " points", 1),
    ("A5 realized speaker groups and their trial counts (section 4)",
     rf"two groups of {A5_GRP[0]} and {A5_GRP[1]} speakers \({A5_GRP_TRIALS[0]:,} and {A5_GRP_TRIALS[1]:,} trials\)", 1),
    # The 2026-09-16 conclusion rewrite leads with the flagship cell, so the value now
    # also appears in section 6; the artifact binding is unchanged.
    ("flagship FPR, abstract, Drift paragraph and conclusion", rf"{FLAG['vanilla_fpr_mean']*100:.2f}\\% FPR", 3),
    ("flagship false-alarm count", rf"about {round(FLAG['vanilla_fpr_mean'] * FLAG['n_dep_bona'])} false alarms per draw among {N_BONA:,} bona fide", 1),
    ("released-score detectors, EXP-109",
     rf"on {min(CORS_N.values())}--{max(CORS_N.values())} of the 42 channel pairs each \({sum(CORS_N.values())} of {42 * len(CORS_N)}\)", 1),
    ("speaker six-cell ranges",
     rf"to {min(SPK_W):.1f}--{max(SPK_W):.1f} points \(means over ten seeds of {SPK['draws_per_width']} draws\), against a median of {min(SPK_NULL):.1f}--{max(SPK_NULL):.1f} under", 1),
    ("A5 primary numerator/denominator (section 4; abstract now carries the spoof-side result)",
     rf"{tf_n} of {tf_t}", 1),
    ("A5 primary percentage (section 4; abstract site cut)", rf"\({round(100*tf_n/tf_t)}\\%\)", 1),
    ("mean realized FPR range, both sites", rf"{FPR_LO*100:.2f}--{FPR_HI*100:.2f}\\%", 2),
    # One of the five recording-count sites was the withdrawn dependence repetition.
    ("per-condition recording count, four sites", rf"{N_BONA:,}(?:-recording| recordings| bona-fide recordings| bona fide| bona-fide and)", 4),
    ("per-condition spoof count, three sites", rf"{N_SPOOF:,} spoof", 3),
    ("speaker count, Setup only", rf"{N_SPK} speakers", 1),
    ("A5 usable-destination counts",
     rf"\({len(A5_USABLE)} ordered pairs\), where {sum(1 for v in A5_USABLE if v['log2_fpr_ratio'] < -SEV)} miss the FPR target conservatively by more than \$2\\times\$ and {sum(1 for v in A5_USABLE if v['fnr_price'] > 0)} pay", 1),
    ("cost mechanism: the three correlations and their p-values",
     rf"is \$\+{MECH_RHO['aasist']:.2f}\$, \$\+{MECH_RHO['ssl']:.2f}\$ and \$\+{MECH_RHO['xlsr_sls']:.2f}\$ for "
     rf"AASIST, SSL-AASIST and XLS-R\+SLS \(\$p={MECH_P['aasist']:.4f},{MECH_P['ssl']:.4f},"
     rf"{MECH_P['xlsr_sls']:.4f}\$, respectively\)", 1),
    ("cost mechanism: the two detectors without it",
     rf"it is \$-{abs(MECH_RHO['xlsr_mamba']):.2f}\$ for XLSR-Mamba and \$\+{MECH_RHO['xlsr_conformer']:.2f}\$ "
     rf"for XLSR-Conformer \(\$p={MECH_P['xlsr_conformer']:.3f}\$\)", 1),
    ("cost mechanism: the A5 correlation over all twin-free pairs",
     rf"On the {MECH_A5['n_pairs']} ASVspoof~5 SSL-AASIST pairs, \$\\rho=\+"
     rf"{MECH_A5['rho']['source_oracle_fnr']:.2f}\$", 1),
    ("the permutation space is 7!", rf"\$7!=5\{{,\}}040\$ source-condition label permutations", 1),
    ("cost mechanism: the direction counts against the applicable cells",
     rf"on {MECH_SIGN['aasist']['pred_cons_and_cons']}, {MECH_SIGN['ssl']['pred_cons_and_cons']} and "
     rf"{MECH_SIGN['xlsr_sls']['pred_cons_and_cons']} of the {MECH_APPLICABLE} applicable cells", 1),
    ("scalar cost: priors, the within-21LA subset and the two counts",
     rf"spoof priors \$\\pi={PRIORS[0]:.2f}\$ and {PRIORS[1]:.2f}, all {CONS_CELLS} within-21LA cells with mean "
     rf"transported FPR below 2\.5\\% have negative degradation across both detectors, although "
     rf"{CONS_MORE_SPOOFS} have higher target than source FNR", 1),
    ("the 42 pairs, in the mechanism and in the p definition", r"42 (?:ordered |dependent )?pairs", 2),
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
    """Bind each Table 1 numeral to its block, row, column and quantity.

    The table is read in document order: a \\multicolumn header opens a block,
    and the row lines that follow belong to it. A row that drifts into another
    block, a missing row, a duplicated row or a cell with the wrong number of
    numerals all raise, so the layout is part of the binding.
    """
    body = tex[tex.index("\\label{tab:fnr}"):tex.index("\\bottomrule")]
    offset = tex.index("\\label{tab:fnr}")
    columns = ("21LA", "21DF", "ITW", "BRSpeech")
    headers = {h: (q, r) for h, q, r in TABLE1_BLOCKS}
    bindings, seen, block = {}, set(), None
    for line in re.finditer(r"(?m)^(.*)$", body):
        text = line.group(1)
        head = re.match(r"\\multicolumn\{5\}\{l\}\{\\emph\{(.+?)\}\}", text)
        if head:
            if head.group(1) not in headers:
                raise AssertionError(f"Table 1 carries an unknown block: {head.group(1)!r}")
            block = head.group(1)
            continue
        row = re.match(r"([^&]+?)\s*&", text)
        if not row or block is None or row.group(1).strip() not in headers[block][1]:
            continue
        name = row.group(1).strip()
        quantities, rows = headers[block]
        if (block, name) in seen:
            raise AssertionError(f"Table 1 row {name!r} appears twice in block {block!r}")
        seen.add((block, name))
        ampersands = [i for i, char in enumerate(text) if char == "&"]
        if len(ampersands) != 4:
            raise AssertionError(f"Table 1 row {name} in {block!r} has {len(ampersands)} data separators")
        for index, (column, expected) in enumerate(zip(columns, rows[name])):
            cell_start = ampersands[index] + 1
            cell_end = ampersands[index + 1] if index + 1 < len(ampersands) else text.rfind("\\\\")
            cell = text[cell_start:cell_end]
            found = list(re.finditer(r"(?<![\\A-Za-z0-9._])(\d+(?:[.,]\d+)?)", cell))
            if len(found) != len(quantities):
                raise AssertionError(f"Table 1 {name}/{column} in {block!r} has {len(found)} numerals; "
                                     f"expected {quantities}")
            for quantity, numeral, wanted in zip(quantities, found, expected):
                start = offset + line.start(1) + cell_start + numeral.start(1)
                bindings[start] = {
                    "end": offset + line.start(1) + cell_start + numeral.end(1),
                    "expected": wanted,
                    "reason": f"Table 1 {quantity}, {name}/{column}",
                }
    missing = {(h, r) for h, _, rows in TABLE1_BLOCKS for r in rows} - seen
    if missing:
        raise AssertionError(f"Table 1 is missing rows: {sorted(missing)}")
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


def table1_pair_swap_probe_failures():
    """Swap the two halves of every pair cell and require the census to notice.

    The policy block prints FPR/FNR and the conformal block FNR (difference);
    a swapped pair keeps both numerals in the document, so only a
    position-anchored binding can catch it.
    """
    failures, probed = [], 0
    bindings = sorted(TABLE1_OCCURRENCE_BINDINGS.items())
    for (start_a, a), (start_b, b) in zip(bindings, bindings[1:]):
        if a["reason"].split(", ", 1)[1] != b["reason"].split(", ", 1)[1] or a["end"] > start_b:
            continue  # not the two halves of one cell
        probed += 1
        if a["expected"] == b["expected"]:
            failures.append(f"{a['reason']}: the two halves are equal, so a swap cannot be detected")
            continue
        swapped = (TEX[:start_a] + b["expected"] + TEX[a["end"]:start_b] + a["expected"] + TEX[b["end"]:])
        marks = _table1_occurrence_bindings(swapped)
        caught = False
        for value, pos, _, context in numeral_occurrences(swapped):
            if pos in marks and occurrence_class(value, context, pos, marks)[0] == "UNACCOUNTED":
                caught = True
        if not caught:
            failures.append(f"{a['reason']}: swapping the halves of the cell is not detected")
    return failures, probed


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
    swap_failures, swap_probes = table1_pair_swap_probe_failures()
    if swap_failures:
        print("  TABLE-1 PAIR-SWAP PROBE FAILURES:")
        for failure in swap_failures:
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
    print(f"  Table 1 pair-swap probes: {swap_probes}")
    print("\nevery numeral occurrence accounted for; critical repeats position-bound")
    return 0


if __name__ == "__main__":
    sys.exit(main())
