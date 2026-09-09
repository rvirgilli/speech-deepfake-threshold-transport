"""Check A2's printed numbers against the result JSONs.

Two properties this has that a naive checker does not:

POSITION-ANCHORED. Matching a literal anywhere in the document lets a corrupted
or deleted value pass, because the same digits usually occur somewhere else.
Every check names an anchor and the value must appear in the window that anchor
opens, so a number that migrates or vanishes from its own sentence fails.

REQUIRED-PRESENCE. Every check that verifies a value only fires when the value
is there; a claim silently dropped from the paper passes such a suite trivially.
The PRESENCE list asserts that specific statements exist at all -- it is the
half that catches a control measured, recorded in JSON, and never written up.

Every artifact-derived literal below is RECOMPUTED from the artifact and
formatted with the rounding the sentence uses: nearest for point values and
ranges, floor for "at least" lower bounds, ceiling for "at most" upper bounds.

Run from paper/A2/. Exit 1 on any failure. A2_TEX and A2_DOCS override the
manuscript path and the reader-file directory for mutation testing; the live
files are never edited by the harness.
"""

import ast
import json
import hashlib
import math
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
EXP = HERE.parent.parent / "experiments"
E102 = EXP / "EXP-102-a2-campaign"
TEX = Path(os.environ.get("A2_TEX", HERE / "main.tex")).read_text()
DOCS = Path(os.environ.get("A2_DOCS", HERE))

drift = json.load(open(E102 / "results_drift.json"))
DIS = json.load(open(E102 / "results_dissociation.json"))
E2 = json.load(open(EXP / "EXP-002-a2-calibration/results.json"))
SLS = json.load(open(E102 / "results_sls_complete.json"))
EER = json.load(open(E102 / "results_table1_eer_spread.json"))
M10 = json.load(open(EXP / "EXP-010-a2-matched-baseline/results.json"))
CMS = json.load(open(E102 / "results_cmethods.json"))
NSW = json.load(open(E102 / "results_nsweep.json"))
PAR = json.load(open(E102 / "results_parametric.json"))
SPK = json.load(open(E102 / "artifacts/speaker_clustering.json"))
A5 = json.load(open(EXP / "EXP-103-a5-replicate/artifacts/results_a5.json"))
PRE = json.load(open(EXP / "EXP-103-a5-replicate/artifacts/precheck.json"))
CORS = json.load(open(EXP / "EXP-109-a2-cors-transport/results_cellA.json"))
PHASE = json.load(open(E102 / "results_phase_sensitivity.json"))

# The tolerance is the paper's own +-5 pp band on realized FPR; the severity
# bar is |log2(FPR/alpha)| > 1, a factor of two either side.
ALPHA, SEV_BAR, TOL = 0.05, 1.0, 0.05
cells = {k: v for sec in ("within", "cross") for k, v in drift[sec].items()}
within = drift["within"]
in_tol = [v for v in cells.values() if abs(v["vanilla_fpr_mean"] - ALPHA) <= TOL]
hidden = [v for v in in_tol if abs(v["log2_fpr_ratio"]) > SEV_BAR]
E2_CORPORA = ("asv21la", "asv21df_100k", "itw", "brspeech_test")
SLS_CORPORA = ("asv21la", "asv21df_full", "itw", "brspeech_test")

failures = []


def window(anchor, chars=1200):
    """Text following an anchor; the anchor must itself be unique."""
    n = _flat(TEX).count(_flat(anchor)) if "\n" in anchor else TEX.count(anchor)
    if n != 1:
        failures.append(f"ANCHOR {'missing' if n == 0 else f'ambiguous (x{n})'}: {anchor!r}")
        return ""
    i = TEX.index(anchor)
    return TEX[i:i + chars]


def _flat(s):
    """Collapse whitespace: a value split by a line wrap is still in its sentence."""
    return " ".join(s.split())


def check(name, anchor, literal, expected=None, source="", chars=1200):
    """`literal` must appear in the window `anchor` opens (not just anywhere)."""
    w = window(anchor, chars)
    if not w:
        return
    if _flat(literal) not in _flat(w):
        loc = ("elsewhere in the document" if _flat(literal) in _flat(TEX)
               else "nowhere in the document")
        failures.append(f"{name}: {literal!r} not in the window after {anchor!r} "
                        f"(found {loc}){f'; source {source}' if source else ''}")
    elif expected is not None:
        print(f"  ok  {name}: {literal}  [{source} = {expected}]")
    else:
        print(f"  ok  {name}: {literal}")


def script_constant(path, name):
    """A module-level literal assignment `NAME = <literal>` in an analysis script."""
    m = re.search(rf"^{name}\s*=\s*([^#\n]+)", path.read_text(), re.M)
    if m is None:
        raise KeyError(f"{name} not assigned in {path}")
    return ast.literal_eval(m.group(1).strip())


def floor_to(x, nd):
    return math.floor(x * 10**nd + 1e-9) / 10**nd


def ceil_to(x, nd):
    return math.ceil(x * 10**nd - 1e-9) / 10**nd


def spearman(x, y):
    """Spearman rank correlation with average ranks for ties (no scipy here)."""
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
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den


# --- the grid: counts recomputed from the drift map ---------------------------
print("position-anchored value checks:")
n_miss = sum(1 for v in cells.values() if abs(v["log2_fpr_ratio"]) > SEV_BAR)
n_within_miss = sum(1 for v in within.values() if abs(v["log2_fpr_ratio"]) > SEV_BAR)
n_within_tol = sum(1 for k, v in cells.items()
                   if k in within and abs(v["vanilla_fpr_mean"] - ALPHA) <= TOL)
n15 = sum(1 for v in in_tol if abs(v["log2_fpr_ratio"]) > 0.585)
n4 = sum(1 for v in in_tol if abs(v["log2_fpr_ratio"]) > 2.0)
assert abs(2 ** 0.585 - 1.5) < 0.001 and 2 ** 2.0 == 4.0
n_conservative = sum(1 for v in cells.values() if v["log2_fpr_ratio"] < -SEV_BAR)
n_liberal = sum(1 for v in cells.values() if v["log2_fpr_ratio"] > SEV_BAR)
assert n_conservative + n_liberal == n_miss, "conservative/liberal split must partition the misses"
assert all(v["log2_fpr_ratio"] < 0 for v in hidden), "an in-tolerance >2x miss is liberal"
check("108-cell decomposition is stated", "The measurement covers 108 deployment cells",
      "two detectors $\\times$ (42 ordered pairs of seven 21LA channel "
      "conditions $+$ 12 ordered pairs of four corpora)",
      f"{len(within)} within + {len(cells) - len(within)} cross", "results_drift.json cell keys")
assert len(cells) == 108 and len(within) == 84, (len(cells), len(within))
check("total miss count and its sign split (abstract)", "Across 108 deployment cells",
      f"{n_miss} realize an FPR more than $2\\times$ off target ({n_liberal} above, {n_conservative} below)",
      (n_miss, n_liberal, n_conservative), "results_drift.json log2_fpr_ratio > 1 / < -1")
# A count is only correct together with the set it was computed over. The audit
# of 2026-08-15 found the abstract asserting "all conservative" of every cell
# missing by >2x, when only the ones inside the band are -- so every headline
# count is bound to its scope, and the scoping words are part of the check.
check("in-tolerance count and hidden count, scoped (abstract)", "Across 108 deployment cells",
      f"Of the {len(in_tol)} cells inside our own $\\pm5$-point tolerance on realized FPR, "
      f"{len(hidden)} are among them, and every one is conservative",
      f"{len(hidden)}/{len(in_tol)}; {n_conservative} conservative / {n_liberal} liberal",
      "results_drift.json")
check("miss count and within-corpus share (experiments)", "Over the 108 cells",
      f"{n_miss} miss target by more than $2\\times$ ({n_within_miss} of the {len(within)} "
      "within-corpus cells)", f"{n_miss}; {n_within_miss}/{len(within)}", "results_drift.json")
check("hidden-cell count (experiments)", "Over the 108 cells",
      f"{len(hidden)} of the {len(in_tol)} cells inside our own $\\pm$5\\,pp tolerance on "
      "realized FPR miss target by over $2\\times$",
      f"{len(hidden)}/{len(in_tol)}", "results_drift.json")
check("the conservative direction is explained as forced, not reported as a finding",
      "Over the 108 cells",
      "necessarily conservatively, since that tolerance excludes a liberal $2\\times$ miss "
      "by construction")
# The paper deliberately attaches NO interval to the count: the within-corpus
# cells share one recording set, so a bootstrap over calibration draws (kept in
# results_count_bootstrap.json, labelled a lower bound) does not cover the
# dominant dependence. What must appear instead is the threshold-sensitivity
# range, at both sites, and the dependence disclosure.
check("sensitivity range (experiments)", "Over the 108 cells",
      f"ranges from {n15} of {len(in_tol)} at $1.5\\times$ to {n4} of {len(in_tol)} at "
      "$4\\times$. We therefore attach no interval",
      f"{n15}/{n4} of {len(in_tol)}", "results_drift.json")
check("within-corpus share of the in-tolerance cells", "Over the 108 cells",
      f"{n_within_tol} of those {len(in_tol)} cells are within-corpus",
      f"{n_within_tol}/{len(in_tol)}", "results_drift.json")
check("hidden-cell count and sensitivity range (figure caption)", "\\caption{Drift map.",
      f"{len(hidden)} of the {len(in_tol)} cells inside the $\\pm$5\\,pp tolerance are off "
      f"target ({n15} at $1.5\\times$, {n4} at $4\\times$; no interval, \\S4), every one "
      "conservative", f"{len(hidden)}/{len(in_tol)}; {n15}/{n4}", "results_drift.json")

# The two sets of similar size must each name itself: 84 within-corpus cells and
# 72 in-tolerance cells are different sets overlapping on n_within_tol.
for m in re.finditer(r"\b84 (?:\w+[- ])?cells\b", _flat(TEX)):
    if "within-corpus" not in m.group(0):
        failures.append(f"an unqualified {m.group(0)!r}: 84 within-corpus and 72 in-tolerance "
                        f"cells are different sets overlapping on {n_within_tol} -- name which")
for m in re.finditer(r"\b72 cells\b(.{0,30})", _flat(TEX)):
    if not re.match(r"\s*(inside|are within-corpus)", m.group(1)):
        failures.append(f"an unqualified '72 cells' at {m.group(0)!r}: say 'inside the tolerance'")

for claim, forcedness in (
        ("realizes 5.0\\% FPR on all four corpora",
         "as expected from marginal rank validity"),):
    w = _flat(TEX)
    i = w.find(_flat(claim))
    if i == -1:
        failures.append(f"MISSING: the by-construction claim {claim!r} is gone; if it was "
                        "removed deliberately, drop this rule too")
    elif _flat(forcedness) not in w[i:i + 400]:
        failures.append(f"by-construction claim stated without its forcedness: {claim!r} "
                        f"needs {forcedness!r} within the same passage")

for phrase, need in ((r"no (?:\w+ )?unlabeled (?:drift )?monitor", "tested"),
                     (r"(?:the|every) (?:\w+ )?label-free (?:heuristic|correction)", "tested")):
    for m in re.finditer(phrase + r"(.{0,40})", _flat(TEX)):
        if need not in m.group(0):
            failures.append(f"unscoped universal claim: {m.group(0)!r} -- needs {need!r}; "
                            "we tested some monitors and one heuristic, not all of them")

# --- the flagship cell, at every site -----------------------------------------
# The cell the paper NAMES must be the costliest one it is allowed to name:
# resolution-limited cells (FPR resting on fewer than five expected false
# alarms, FPR < 5/n in drift_map.py) are censored, so citing the raw argmax
# would be wrong. Position anchors are the only discriminator for these values:
# a value-based census accepts 70 -> 77 as a legitimately accounted numeral.
quotable = max((kv for kv in within.items() if not kv[1]["resolution_limited"]),
               key=lambda kv: kv[1]["fnr_price"])
assert quotable[0] == max((kv for kv in cells.items() if not kv[1]["resolution_limited"]),
                          key=lambda kv: kv[1]["fnr_price"])[0]
det, pair = quotable[0].split("/")
src, dst = pair.split("->")
CODEC_TEX = {"g722": "G.722", "none": "none", "pstn": "PSTN", "gsm": "GSM",
             "opus": "Opus", "alaw": "A-law", "ulaw": "$\\mu$-law"}
_F = quotable[1]
above = [v for v in cells.values() if v["fnr_price"] > _F["fnr_price"]]
assert above and all(v["resolution_limited"] for v in above), "a higher price is quotable"
print("\nflagship cell, every site:")
check("worst spoof-side price (value and cell)", "Their spoof-side cost",
      f"reaches $+{_F['fnr_price']:.2f}$ ({det.upper()}, {CODEC_TEX[src]}$\\to${CODEC_TEX[dst]}",
      quotable[0], "argmax fnr_price among non-resolution-limited cells")
check("flagship FPR as a count of bona fide", "Their spoof-side cost",
      f"whose {_F['vanilla_fpr_mean']*100:.2f}\\% FPR is {round(_F['vanilla_fpr_mean'] * _F['n_dep_bona'])} "
      f"of {_F['n_dep_bona']:,} bona fide", round(_F['vanilla_fpr_mean'] * _F['n_dep_bona']),
      "vanilla_fpr_mean x n_dep_bona")
check("the censored higher cells are counted and the rule stated", "Their spoof-side cost",
      f"{ {1: 'one', 2: 'two', 3: 'three'}[len(above)] } cells score marginally higher but "
      "their FPR rests on fewer than five expected false alarms, $\\mathrm{FPR}<5/n$",
      len(above), "resolution_limited cells with price above the named cell")
for site, anc in (("abstract", "In the costliest resolvable cell"),
                  ("Fig. 1 caption", "The costliest on the spoof side are among them")):
    check(f"flagship named ({site})", anc,
          f"{det.upper()} " + ("calibrated on a PSTN transmission" if site == "abstract"
                               else f"{CODEC_TEX[src]}$\\to${CODEC_TEX[dst]}"),
          quotable[0], "results_drift.json")
    check(f"flagship missed-spoof rate ({site})", anc,
          f"{_F['vanilla_fnr_mean']*100:.0f}\\% of spoofs",
          f"{_F['vanilla_fnr_mean']:.4f}", "results_drift.json")
    check(f"flagship oracle rate ({site})", anc,
          f"against {_F['fnr_oracle']*100:.2f}\\%", f"{_F['fnr_oracle']:.4f}", "results_drift.json")
    check(f"flagship realized FPR ({site})", anc,
          f"{_F['vanilla_fpr_mean']*100:.2f}\\% FPR", f"{_F['vanilla_fpr_mean']:.4f}",
          "results_drift.json")
pstn_cells = {k: v for k, v in within.items() if k.startswith(f"{det}/{src}->")}
assert len(pstn_cells) == 6
# "at least" is a lower bound, so the printed value is the floor of the minimum.
check("every cell calibrated on the flagship source pays a floor price",
      "This cell is not isolated",
      f"every {det.upper()} cell calibrated on {CODEC_TEX[src]} pays at least "
      f"$+{floor_to(min(v['fnr_price'] for v in pstn_cells.values()), 2):.2f}$",
      min(v['fnr_price'] for v in pstn_cells.values()), "min fnr_price over the six cells")
check("the same threshold's FNR on the calibration condition", "This cell is not isolated",
      f"on {CODEC_TEX[src]} itself the same threshold misses "
      f"{_F['cal_fnr_at_threshold']*100:.0f}\\% of spoofs",
      _F["cal_fnr_at_threshold"], "cal_fnr_at_threshold")
auc = DIS[det]["cond_auc"]
others = [auc[c] for c in ("alaw", "ulaw", "gsm", "g722", "opus", "none")]
assert auc[src] < min(others), "the flagship source is no longer the least separable condition"
check("calibration-condition AUC against the other six", "least separable 21LA condition",
      f"(AUC {auc[src]:.3f} against {min(others):.3f}--{max(others):.3f} elsewhere)",
      (auc[src], min(others), max(others)), "results_dissociation.json cond_auc")
sev = drift["severity_vs_w1_transfer"]
rho = {d: spearman([abs(v["log2_fpr_ratio"]) for k, v in cells.items() if k.startswith(d + "/")],
                   [v["w1_bona_oracle"] for k, v in cells.items() if k.startswith(d + "/")])
       for d in ("ssl", "aasist")}
check("Fig. 1 right: severity/W1 Spearman per detector", "\\caption{Drift map.",
      f"($\\rho$ {rho['ssl']:.2f}/{rho['aasist']:.2f})", rho,
      "Spearman(|log2_fpr_ratio|, w1_bona_oracle) over 54 cells each", chars=2200)
check("Fig. 1 right: cross-detector transfer R2", "\\caption{Drift map.",
      f"(held-out $R^2$ {sev['fit_ssl_test_aasist']['r2_transfer']:.2f}/"
      f"{sev['fit_aasist_test_ssl']['r2_transfer']:.2f})", None, "severity_vs_w1_transfer",
      chars=2200)

# --- the setup: the twin structure every 21LA condition shares ------------------
# 2,356 / 67 / 21,164 are derived from the LA key (column 8 = phase, `hidden`
# excluded; column 7 is the trim flag, not the phase) and carried by the
# artifacts: every within cell records n_dep_bona/n_dep_spoof, and the speaker
# audit records n_speakers per condition. Binding to the artifacts keeps the
# guard runnable from the extracted package, which has no key file.
print("\nsetup and dependence unit:")
n_bona = {v["n_dep_bona"] for v in within.values()} | {v["n_cal_bona"] for v in within.values()}
n_spoof = {v["n_dep_spoof"] for v in within.values()}
n_spk = {v["n_speakers"] for v in SPK["cells"].values()}
assert len(n_bona) == len(n_spoof) == len(n_spk) == 1, "21LA conditions no longer twin"
n_bona, n_spoof, n_spk = n_bona.pop(), n_spoof.pop(), n_spk.pop()
assert {v["n_utterances"] for v in SPK["cells"].values()} == {n_bona}
check("per-condition bona-fide / speaker / spoof counts", "Each 21LA condition is then the same",
      f"{n_bona:,} bona-fide recordings of {n_spk} speakers and {n_spoof:,} spoofs",
      (n_bona, n_spk, n_spoof), "results_drift.json n_dep_*; speaker_clustering.json n_speakers")
check("dependence unit is disclosed for the within-corpus cells", "Over the 108 cells",
      f"same {n_bona:,} recordings from the same {n_spk} speakers under seven 21LA channel "
      "conditions, so they are not independent observations and no significance claim is made",
      (n_bona, n_spk), "results_drift.json")
check("the replicate names the reordered set", "\\textbf{Replication on recording-disjoint data.}",
      f"reorders one {n_bona:,}-recording set", n_bona, "results_drift.json")
df = EER["cells"]["ssl/asv21df_100k"]
assert df["n_bona"] == E2["ssl"]["asv21df_100k"]["n_bona"] == EER["cells"]["aasist/asv21df_100k"]["n_bona"]
check("21DF 100k sample size after the exclusion", "use a random 100k-trial subsample",
      f"({df['n_bona'] + df['n_spoof']:,} trials, {df['n_bona']:,} bona fide, after the exclusion)",
      (df["n_bona"], df["n_spoof"]), "results_table1_eer_spread.json")
check("official-score evaluation population", "official XLS-R+SLS scores are evaluated on the full",
      f"({SLS['asv21la']['n_bona']:,} and {SLS['asv21df_full']['n_bona']:,} bona fide)",
      None, "results_sls_complete.json n_bona")
check("the protocol correction is disclosed as post-hoc", "The 2021 keys carry",
      "A first version of this study did so; as a post-hoc protocol correction, every result "
      "below excludes it and keeps the two untrimmed phases", chars=1600)
check("the trimmed subset is framed as a shortcut-removed benchmark, not a channel", "The 2021 keys carry",
      "the trimmed subset is a shortcut-removed benchmark \\cite{dao26} rather than a deployment channel")
_pk = ("miss_2x", "within_miss_2x", "in_band", "hidden_in_band")
assert all(PHASE["eval_only"][k] == PHASE["eval+progress"][k] for k in _pk), \
    {k: (PHASE["eval_only"][k], PHASE["eval+progress"][k]) for k in _pk}
assert (PHASE["eval+progress"]["miss_2x"], PHASE["eval+progress"]["hidden_in_band"]) == (n_miss, len(hidden))
check("eval-only sensitivity is stated and holds", "The 2021 keys carry",
      "restricting to the eval phase alone changes no count reported below",
      {k: PHASE["eval_only"][k] for k in _pk}, "results_phase_sensitivity.json eval_only == eval+progress",
      chars=1600)
check("the hidden phase is described", "The 2021 keys carry",
      "the hidden phase removes non-speech by voice-activity detection")
check("ASVspoof 5 is scoped out of the correction", "\\textbf{Setup.}",
      "ASVspoof~5 has no hidden phase", chars=2600)
n_grid = [int(N) for N in NSW["n_sweep"]["ssl/itw"] if N.isdigit()]
assert max(n_grid) == 10_000
check("budget sweep endpoints", "\\textbf{Setup.}",
      f"Budget sweeps cover $N{{=}}{min(n_grid)}$--$10^4$; full-grid comparisons use "
      f"$N{{=}}{script_constant(E102 / 'drift_map.py', 'N_CAL')}$",
      (min(n_grid), max(n_grid)), "results_nsweep.json keys; drift_map.py N_CAL", chars=2600)
B = script_constant(E102 / "drift_map.py", "B")
assert B == script_constant(E102 / "n_sweep.py", "B") == EER["B"]
check("paired draws B", "\\textbf{Setup.}", f"paired cohort draws ($B{{=}}{B}$)", B,
      "drift_map.py / n_sweep.py B", chars=2600)
check("spoof-positive convention (section 3)", "\\section{Threshold policies}",
      "We treat spoof as the positive class")
# 1.645 is the one-sided normal 95% quantile: solve Phi(z) = 0.95 by bisection.
lo, hi = 0.0, 5.0
for _ in range(60):
    mid = (lo + hi) / 2
    lo, hi = (mid, hi) if 0.5 * (1 + math.erf(mid / math.sqrt(2))) < 1 - ALPHA else (lo, mid)
check("Gaussian quantile constant", "\\section{Threshold policies}",
      f"$t=\\bar s_N - {lo:.3f}\\,\\hat\\sigma_N$", lo, "Phi^-1(0.95)", chars=2600)

# --- the calibration grid ------------------------------------------------------
print("\ncalibration grid:")
q = [E2[d][c]["quantile"]["500"] for d in ("aasist", "ssl") for c in E2_CORPORA]
q += [SLS[c]["quantile"] for c in SLS_CORPORA]
oracles = [E2[d][c]["oracle"]["fnr"] for d in ("aasist", "ssl") for c in E2_CORPORA]
oracles += [SLS[c]["oracle"]["fnr"] for c in SLS_CORPORA]
fpr_lo, fpr_hi = min(e["fpr_mean"] for e in q), max(e["fpr_mean"] for e in q)
# "at most" is an upper bound, so the printed premium is the ceiling of the max
# over usable cells (oracle FNR <= 50%).
premium = ceil_to(max(e["fnr_minus_oracle_pts"] for e, o in zip(q, oracles) if o <= 0.5), 1)
check("mean realized FPR range (abstract)", "A target-channel bona-fide order statistic",
      f"mean realized FPR is {fpr_lo*100:.2f}--{fpr_hi*100:.2f}\\% over twelve detector--corpus "
      f"cells, with at most a {premium:.1f}-point FNR premium",
      (fpr_lo, fpr_hi, premium), "EXP-002 results.json + results_sls_complete.json")
check("mean realized FPR range (experiments)", "\\textbf{FPR control and FNR cost.}",
      f"mean realized FPR is {fpr_lo*100:.2f}--{fpr_hi*100:.2f}\\% across the {len(q)} "
      "detector--corpus cells", (fpr_lo, fpr_hi), "EXP-002 results.json + results_sls_complete.json")
sp_lo = min(v["quantile_N500_fpr_pct_2.5_97.5"][0] for v in EER["cells"].values())
sp_hi = max(v["quantile_N500_fpr_pct_2.5_97.5"][1] for v in EER["cells"].values())
check("held-out spread envelope", "\\textbf{FPR control and FNR cost.}",
      f"lies within {sp_lo*100:.1f}--{sp_hi*100:.1f}\\% on every cell", (sp_lo, sp_hi),
      "results_table1_eer_spread.json quantile_N500_fpr_pct_2.5_97.5")
check("FNR premium on usable cells", "\\textbf{FPR control and FNR cost.}",
      f"FNR premium over the oracle threshold is $\\le${premium:.1f} points", premium,
      "max fnr_minus_oracle_pts over cells with oracle FNR <= 0.5")
_z = [e["znorm"]["fpr_mean"] for N, e in NSW["n_sweep"]["ssl/itw"].items() if isinstance(e, dict)]
_g = [e["parametric"]["fpr_mean"] for N, e in PAR["ssl"]["itw"].items() if isinstance(e, dict)]
check("z-norm's ITW offset, recomputed", "\\textbf{FPR control and FNR cost.}",
      f"cohort z-norm stays {round((min(_z)-ALPHA)*100)}--{round((max(_z)-ALPHA)*100)}\\,pp high",
      f"{min(_z):.4f}-{max(_z):.4f}", "results_nsweep.json")
# The upper endpoint is 1.45 pp exactly in decimal; the artifact's float is just
# below the tie and formats as 1.4. If the artifact moves, the literal moves.
check("the Gaussian's ITW offset, recomputed", "\\textbf{FPR control and FNR cost.}",
      f"the Gaussian quantile {(min(_g)-ALPHA)*100:.1f}--{(max(_g)-ALPHA)*100:.1f}\\,pp high",
      f"{min(_g):.4f}-{max(_g):.4f}", "results_parametric.json")

ssl_naive = [E2["ssl"][c]["naive_transfer"]["fpr"] for c in E2_CORPORA]
sls_naive = [SLS[c]["naive_transfer"]["fpr"] for c in SLS_CORPORA]
check("intro naive-transfer FPR range (SSL-AASIST)", "A threshold set to 5\\% FPR on ASVspoof~2019~LA dev",
      f"realizes {min(ssl_naive)*100:.0f}--{max(ssl_naive)*100:.1f}\\% FPR across",
      (min(ssl_naive), max(ssl_naive)), "EXP-002 results.json naive_transfer")
check("intro naive-transfer FPR (XLS-R+SLS)", "A threshold set to 5\\% FPR on ASVspoof~2019~LA dev",
      f"it reaches {max(sls_naive)*100:.1f}\\%", max(sls_naive), "results_sls_complete.json")
assert all(round(E2["ssl"][c]["quantile"]["500"]["fpr_mean"] * 100, 1) == 5.0 for c in E2_CORPORA)
check("the quantile hits target on all four SSL corpora", "\\textbf{Matched-resource policy comparison.}",
      "realizes 5.0\\% FPR on all four corpora", None, "EXP-002 results.json quantile/500")
naive_miss = [abs(f - ALPHA) * 100 for f in ssl_naive]
check("naive transfer miss range (SSL-AASIST)", "\\textbf{Matched-resource policy comparison.}",
      f"Naive transfer misses target by {min(naive_miss):.0f}--{max(naive_miss):.0f} pp",
      (min(naive_miss), max(naive_miss)), "EXP-002 results.json naive_transfer, |fpr-0.05|")
best = [min(abs(CMS["ssl"][c][m]["fpr"] - ALPHA) for m in ("C1_znorm", "C2_tempshift")) * 100
        for c in E2_CORPORA]
check("best unlabeled correction miss range", "\\textbf{Matched-resource policy comparison.}",
      f"best unlabeled correction misses by {min(best):.0f}--{max(best):.0f} pp",
      (min(best), max(best)), "results_cmethods.json C1/C2, |fpr-0.05|")
c5 = min(CMS[d][c]["C5_asnorm"]["fnr"] for d in CMS for c in CMS[d])
check("C5 degeneracy level", "\\textbf{Matched-resource policy comparison.}",
      f"driving FNR to ${{\\approx}}{c5*100:.0f}\\%", c5, "results_cmethods.json C5_asnorm min fnr")
par = {(d, c): PAR[d][c]["500"]["parametric"]["fpr_mean"] for d in PAR for c in PAR[d]
       if isinstance(PAR[d][c], dict) and "500" in PAR[d][c]}
assert len(par) == 8
check("Gaussian worst miss and ITW miss at N=500", "\\textbf{Matched-resource policy comparison.}",
      f"misses by up to {max(abs(f-ALPHA)*100 for f in par.values()):.1f} pp "
      f"({(par[('ssl', 'itw')]-ALPHA)*100:.2f} pp on ITW)", par, "results_parametric.json")
assert par[("ssl", "brspeech_test")] == 0.0
check("Gaussian collapse on BRSpeech", "\\textbf{Matched-resource policy comparison.}",
      f"collapses to {par[('ssl', 'brspeech_test')]*100:.0f}\\% on BRSpeech", None,
      "results_parametric.json")
zdev = [abs(M10[d][c]["500"]["znorm"]["fpr_mean"] - ALPHA) * 100 for d in ("ssl", "aasist")
        for c in E2_CORPORA]
beyond = [x for x in zdev if x > 2]
sls_z = max(abs(SLS[c]["znorm"]["fpr_mean"] - ALPHA) * 100 for c in SLS_CORPORA)
check("cohort z-norm miss range and cell count", "\\textbf{Matched-resource policy comparison.}",
      f"({min(beyond):.0f}--{max(beyond):.0f} pp, {len(beyond)}/{len(zdev)} cells beyond "
      f"$\\pm$2 pp, and up to {sls_z:.1f} pp for XLS-R+SLS)", (beyond, sls_z),
      "EXP-010 results.json 500/znorm; results_sls_complete.json znorm")

# --- contamination and the label-free heuristic ---------------------------------
print("\ncontamination and heuristic:")
c_rate, c_n = max(script_constant(E102 / "n_sweep.py", "CONTAM")), max(script_constant(E102 / "n_sweep.py", "CONTAM_NS"))
_cont = [e[f"N{c_n}_c{c_rate}"]["fpr_mean"] for e in NSW["contamination"].values()]
_below = sorted(v for v in _cont if v < ALPHA)
assert len(_below) == 7 and len(_cont) == 8, f"contamination: {len(_below)}/{len(_cont)} below target"
check("contamination: rate and cohort size", "In a separate pre-specified test",
      f"replacing {round(c_rate * c_n)} of the {c_n} cohort recordings with spoofs",
      (c_rate, c_n), "n_sweep.py CONTAM / CONTAM_NS")
check("contamination: how many cells fall below target", "In a separate pre-specified test",
      f"on seven of eight cells ({_below[0]*100:.2f}--{_below[-1]*100:.2f}\\%)",
      f"{len(_below)}/{len(_cont)} below {ALPHA}", "results_nsweep.json")
check("contamination: the cell above target is named as near-inert", "In a separate pre-specified test",
      f"nearly unchanged at {max(_cont)*100:.2f}\\%", f"max {max(_cont)}", "results_nsweep.json")
n_w = sum(1 for v in cells.values() if abs(v["weighted_fpr_mean"] - ALPHA) <= 0.02)
n_u = sum(1 for v in cells.values() if abs(v["vanilla_fpr_mean"] - ALPHA) <= 0.02)
check("heuristic contest, both counts", "\\textbf{Label-free heuristic.}",
      f"It holds {n_w}\\slash{len(cells)} cells within 2\\,pp of target, compared with "
      f"{n_u}\\slash{len(cells)} unweighted", (n_w, n_u), "results_drift.json weighted_fpr_mean")
dm = (E102 / "drift_map.py").read_text()
bins = int(re.search(r"def density_ratio_weights\(.*bins=(\d+)\)", dm).group(1))
clip = re.search(r"np\.clip\(q / np\.maximum\(p, 1e-8\), ([\d.]+), ([\d.]+)\)", dm).groups()
check("heuristic design constants", "\\textbf{Label-free heuristic.}",
      f"({bins} equal-width bins over the pooled score range, ratio clipped to "
      f"$[{float(clip[0]):g},{float(clip[1]):g}]$, {script_constant(E102 / 'drift_map.py', 'B_WEIGHTED')} draws)",
      (bins, clip), "drift_map.py density_ratio_weights / B_WEIGHTED")

# --- monitors: both readings ----------------------------------------------------
print("\nmonitors:")
acc = re.search(r"if tpr >= ([\d.]+) and fpr <= ([\d.]+):", dm).groups()
check("monitor acceptance criterion", "\\textbf{Monitoring} was pre-specified",
      f"acceptance TPR~$\\ge$~{acc[0]} and FPR~$\\le$~{acc[1]}", acc, "drift_map.py")
pre = drift["monitor_eval_PREREGISTERED"]
cor = drift["monitor_eval"]
assert all(v["achieves_tpr80_fpr20"] is None for v in pre.values()), "a pre-specified pass exists"
check("pre-specified reading: no operating point", "\\textbf{Monitoring} was pre-specified",
      "under the pre-specified definition no monitor has an operating point", None,
      "monitor_eval_PREREGISTERED all null")
passing = [k for k, v in cor.items() if v["achieves_tpr80_fpr20"] is not None]
assert passing == ["ssl/w1_mixture"], passing
op = cor["ssl/w1_mixture"]["achieves_tpr80_fpr20"]
assert op is not None and cor["aasist/w1_mixture"]["achieves_tpr80_fpr20"] is None
check("corrected reading: the one in-sample pass, scoped to SSL-AASIST", "\\textbf{Monitoring} was pre-specified",
      "has an in-sample operating point on SSL-AASIST only (threshold selected on the same cells)",
      {k: v["achieves_tpr80_fpr20"] for k, v in cor.items() if "w1" in k}, "monitor_eval *.achieves_tpr80_fpr20")
benign = [v for k, v in cells.items() if k.startswith("ssl/") and abs(v["log2_fpr_ratio"]) <= SEV_BAR]
half = sum(1 for v in benign if v["monitors"]["w1_mixture_prev_half"] >= op["threshold"])
x15 = sum(1 for v in benign if v["monitors"]["w1_mixture_prev_x15"] >= op["threshold"])
check("prevalence re-mix pushes benign cells over the threshold",
      "\\textbf{Monitoring} was pre-specified",
      f"re-mixing the deployment spoof share by $\\pm$50\\% pushes {half}\\slash{len(benign)} and "
      f"{x15}\\slash{len(benign)} benign cells over it", (half, x15, len(benign)),
      "monitors.w1_mixture_prev_half / prev_x15 vs achieves_tpr80_fpr20.threshold")
assert all(v["achieves_tpr80_fpr20"] is None for k, v in cor.items() if "entropy" in k)
check("entropy monitor fails under both readings", "\\textbf{Monitoring} was pre-specified",
      "fails on both detectors under either definition")

# --- the ASVspoof 5 replicate, recomputed from its own artifacts ---------------
def _miss_frac(section, bar=1.0):
    n = tot = 0
    for d in ("ssl", "aasist"):
        for v in A5[f"{d}/{section}"].values():
            tot += 1
            n += abs(v["log2_fpr_ratio"]) > bar
    return n, tot


_n_tf, _tot_tf = _miss_frac("twin_free")
conds = {k.split("->")[1] for k in A5["ssl/twin_free"]}
print("\nASVspoof 5 replicate (recomputed):")
check("replicate miss count and rate", "Over both detectors the transported threshold",
      f"{_n_tf} of {_tot_tf} pairs ({round(100*_n_tf/_tot_tf)}\\%)",
      f"{_n_tf}/{_tot_tf}", "results_a5.json")
check("the grid it is contrasted against, at the SAME scope",
      "Over both detectors the transported threshold",
      f"against {round(100*n_within_miss/len(within))}\\% on\nthe grid above",
      f"{n_within_miss}/{len(within)} pooled over both detectors", "results_drift.json")
check("abstract carries the pooled A5 count and denominator",
      "The failure persists on ASVspoof~5", f"{_n_tf} of {_tot_tf} ordered pairs",
      f"{_n_tf}/{_tot_tf}", "results_a5.json")
check("abstract carries the pooled A5 rate",
      "The failure persists on ASVspoof~5", f"({round(100*_n_tf/_tot_tf)}\\%)",
      f"{_n_tf}/{_tot_tf}", "results_a5.json")
check("A5 speaker count and twin-free share (declared; no artifact carries them)",
      "\\textbf{Replication on recording-disjoint data.}",
      "737 speakers. It is restricted to the 80.5\\% of bona-fide sources appearing under one condition")
check("A5 condition count", "\\textbf{Replication on recording-disjoint data.}",
      f"The grid contains {len(conds) - 1} organizer-applied codecs plus the unprocessed source",
      len(conds), "results_a5.json condition keys")
assert "hidden excluded" in CORS["phases"], CORS["phases"]
_cors_n = {d: sum(1 for v in c.values() if abs(v["log2_fpr_ratio"]) > SEV_BAR) for d, c in CORS["cells"].items()}
assert set(_cors_n) == {"xlsr_sls", "xlsr_mamba", "xlsr_conformer"} and all(len(c) == 42 for c in CORS["cells"].values())
_cors_price = max(c["pstn->g722"]["fnr_price"] for c in CORS["cells"].values())
CORS_ANCHOR = "On three further detectors with released 21LA scores"
check("released-score detectors: per-detector miss range and pooled count", CORS_ANCHOR,
      f"on {min(_cors_n.values())}--{max(_cors_n.values())} of the 42 channel pairs each "
      f"({sum(_cors_n.values())} of {42 * len(_cors_n)})", _cors_n, "EXP-109 results_cellA.json |log2_fpr_ratio|>1")
check("released-score detectors: spoof-side cost bound on the flagship pair", CORS_ANCHOR,
      f"stays below {math.ceil(_cors_price * 100)} points", _cors_price, "max fnr_price of pstn->g722 over the three")
assert _cors_price * 100 < math.ceil(_cors_price * 100) and _cors_price * 100 > math.ceil(_cors_price * 100) - 1
for tag, name in (("ssl", "SSL-AASIST"), ("aasist", "AASIST")):
    check(f"gate oracle FNR ({tag})", "AASIST is\noverlap-dominated on this corpus",
          f"{PRE[tag]['oracle_fnr_at_5pct_fpr']*100:.1f}\\%", tag, "precheck.json")
assert PRE["aasist"]["oracle_fnr_at_5pct_fpr"] > 0.5 > PRE["ssl"]["oracle_fnr_at_5pct_fpr"]

# --- dispersion: exact Beta and the speaker diagnostic ---------------------------
# Integer-parameter Beta CDF equals a binomial upper tail, so no scipy is needed.
def _beta_cdf(x, a, b):
    n = a + b - 1
    return sum(math.comb(n, j) * x**j * (1-x)**(n-j) for j in range(a, n+1))


def _beta_ppf(p, a, b):
    lo, hi = 0.0, 1.0
    for _ in range(100):
        mid = (lo + hi) / 2
        if _beta_cdf(mid, a, b) < p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def _beta_summary(N):
    k = math.floor((N + 1) * ALPHA)
    b = N + 1 - k
    inside = _beta_cdf(.07, k, b) - _beta_cdf(.03, k, b)
    return inside, _beta_ppf(.025, k, b), _beta_ppf(.975, k, b)


print("\ndispersion:")
_p100, _, _ = _beta_summary(100)
_p500, _q500lo, _q500hi = _beta_summary(500)
check("N=100 probability within +/-2 FPR points", "\\textbf{Calibration and detection are distinct.}",
      f"{_p100*100:.1f}\\% probability of landing within $\\pm2$ FPR points", _p100, "exact Beta law")
check("N=500 probability within +/-2 FPR points", "\\textbf{Calibration and detection are distinct.}",
      f"the probability is {_p500*100:.1f}\\%", _p500, "exact Beta law")
_p450, _, _ = _beta_summary(450)
check("N=450 probability within +/-2 FPR points (operational prescription)",
      "\\textbf{Operational implications.}",
      f"$N{{=}}450$ gives {_p450*100:.1f}\\% probability of landing within $\\pm2$ points", _p450, "exact Beta law")
check("N=500 central 95% Beta interval", "\\textbf{Calibration and detection are distinct.}",
      f"{_q500lo*100:.2f}--{_q500hi*100:.2f}\\% central 95\\% interval",
      (_q500lo, _q500hi), "exact Beta law")
_spk = SPK["cells"]["aasist/pstn"]
assert (SPK["nominal_N_utterances"], SPK["draws_per_width"], SPK["permutations"], len(SPK["seed_sweep"])) == (
    script_constant(E102 / "speaker_clustering_audit.py", "N"),
    script_constant(E102 / "speaker_clustering_audit.py", "B"),
    script_constant(E102 / "speaker_clustering_audit.py", "PERMUTATIONS"), 10)
SPK_ANCHOR = "A post-hoc speaker-disjoint diagnostic"
check("speaker diagnostic design: cohort floor and median speakers", SPK_ANCHOR,
      f"holds at least {SPK['nominal_N_utterances']} recordings (median "
      f"{_spk['permutation']['median_calibration_speakers']} speakers)",
      None, "speaker_clustering.json")
_w = [v["seed_sweep"]["mean_width_pp"] for v in SPK["cells"].values()]
_null = [v["permutation"]["null_median_pp"] for v in SPK["cells"].values()]
assert len(_w) == 6
check("speaker-disjoint width range over the six cells, seeds and draws", SPK_ANCHOR,
      f"on all six tested detector--condition cells to {min(_w):.1f}--{max(_w):.1f} points "
      f"(means over ten seeds of {SPK['draws_per_width']} draws)", (min(_w), max(_w)),
      "speaker_clustering.json cells.*.seed_sweep.mean_width_pp")
check("speaker-permutation null range and count", SPK_ANCHOR,
      f"against {min(_null):.1f}--{max(_null):.1f} under a speaker-permutation null "
      f"that preserves group sizes ({SPK['permutations']} permutations)", (min(_null), max(_null)),
      "speaker_clustering.json cells.*.permutation.null_median_pp")
if any(v["permutation"]["null_exceedances"] != 0 for v in SPK["cells"].values()):
    failures.append("speaker clustering: a cell no longer exceeds every permutation null")
assert all(v["seed_sweep"]["mean_width_pp"] > v["permutation"]["null_median_pp"] for v in SPK["cells"].values())
check("the abstract's 'every tested cell' is the six-cell claim", "The guarantee is marginal, not per deployment",
      "speaker clustering widens it on every tested cell", None, "all six widths exceed their null")

# --- table and figure furniture -------------------------------------------------
print("\ntable and figure furniture:")
check("Table 1 caption states the estimand and B", "\\caption{Upper block:",
      f"mean FNR (\\%) over $B{{=}}{B}$ paired $N{{=}}{script_constant(E102 / 'drift_map.py', 'N_CAL')}$ cohorts")
_ec = EER["eer_by_condition_21la"]
def _rng(det):
    v = list(_ec[f"{det}/asv21la"].values())
    assert len(v) == 7
    return f"{min(v)*100:.1f}--{max(v)*100:.1f}"
check("Table 1 caption: per-condition 21LA EER ranges", "\\caption{Upper block:",
      f"per condition it is {_rng('aasist')} (AASIST), {_rng('ssl')} (SSL-AASIST) and {_rng('sls')} (XLS-R+SLS)",
      {d: _rng(d) for d in ("aasist", "ssl", "sls")}, "results_table1_eer_spread.json eer_by_condition_21la")
check("Table 1 caption resolves to the release", "\\caption{Upper block:",
      "\\url{https://github.com/rvirgilli/speech-deepfake-threshold-transport")
check("Table 1 EER block header", "\\label{tab:fnr}", "\\emph{EER (\\%) on the same trials}")
check("the 2x bar is disclosed as post-hoc in the figure caption", "\\caption{Drift map.",
      "post-hoc $2\\times$ two-sided bar")

# --- SCOPE WORDS AND DISQUALIFICATIONS -----------------------------------------
# Every item below is something a previous round RESTORED after it had silently
# vanished, so each can vanish again identically.
SCOPE_CRITICAL = [
    ("C5's disqualification, lost once with Table 2's footnote",
     "C5 AS-norm is excluded as degenerate"),
    ("the overlap cutoff is named as ours, not borrowed",
     "our own cutoff excludes it"),
    ("the axis boundary itself -- the sentence the whole argument rests on",
     "the spoof-side axis here rests on one detector"),
    ("the A5 rate contrast is not attributed to disjointness",
     "numerical rate difference is confounded by corpus, codecs and attacks"),
    ("the finite-sample claim is marginal",
     "\\emph{marginal expected} FPR control"),
    ("the additive band's blind spot is stated as arithmetic, not as a finding",
     "cannot distinguish a collapsed threshold from a controlled one"),
]
print("\nscope words and disqualifications:")
check("the realized-threshold limitation is explicit (abstract)", "The guarantee is marginal, not per deployment",
      "its Beta dispersion law assumes iid sampling")
check("exchangeability is scoped to marginal rank validity (section 1)", "We study the deployment behavior",
      "has marginal rank validity under exchangeability without ties, and under iid continuous sampling "
      "its conditional FPR follows an exact finite-sample Beta law")
check("exchangeability is scoped to marginal rank validity (section 3)", "\\section{Threshold policies}",
      "Exchangeability without ties gives the last policy's marginal rank guarantee. Under iid continuous "
      "sampling from a fixed score distribution, its conditional FPR across calibration draws additionally "
      "follows $\\mathrm{Beta}(k, N{+}1{-}k)$", chars=2600)
check("the positioned residual is stated after the prior-art paragraph", "\\section{Related work}",
      "what remains open is how a threshold set on one channel behaves on another when only bona fide "
      "is available to reset it", chars=2200)
check("fixed-threshold auditing is credited as standard practice", "\\section{Related work}",
      "standard biometric practice \\cite{nistir8280,tdcf,asvspoof5}", chars=2200)
for why, needle in SCOPE_CRITICAL:
    if _flat(needle) not in _flat(TEX):
        failures.append(f"SCOPE WORD GONE — {why}: {needle!r}")
    else:
        print(f"  ok  {why}")

# --- statements that must EXIST (the half that catches silent omissions) ----
PRESENCE = [
    ("the paper disclaims both the rule and the audit practice as prior art",
     r"We claim neither the threshold rule nor\s+the practice of auditing both error rates"),
    ("the direct 2026 quantile prior art is cited", r"zhao26ca"),
    ("the direct 2026 fixed-threshold audit is cited", r"schaefer26reality"),
    ("the quantitative delta from the closest threshold-transfer audit is explicit",
     r"Relative to that audit's one detector and two target corpora,.{0,180}108-cell.{0,180}264-pair"),
    ("the 2x severity bar is disclosed as post-hoc in the main text",
     r"The \$2\\times\$ bar is post-hoc"),
    ("the +-5pp tolerance is identified as OUR choice, not the field's",
     r"our own \$\\pm\$5\\,pp tolerance"),
    ("spoof-side cost is oracle-referenced, not calibration-referenced",
     r"against the oracle\s*\n?\s*threshold for that deployment"),
    ("no predictive law is claimed", r"no predictive law is claimed"),
    ("limitations section exists", r"\\textbf\{Limitations\.\}"),
    ("detector-conditioning is scoped, not claimed", r"\$n\{=\}2\$ cannot separate the explanations"),
    ("the cost is not called identifiable from bona-fide scores",
     r"does not make spoof-side cost identifiable from bona-fide scores alone"),
    ("the speaker-unit failure is disclosed",
     r"speaker-disjoint diagnostic on 21LA, which calibrates.{0,200}widened"),
    ("the flag-definition change is disclosed as post-hoc",
     r"The flag definition was changed after results existed"),
    ("the monitor pass is scoped to one detector and in-sample",
     r"in-sample operating point on SSL-AASIST only"),
]
print("\nrequired-presence checks:")
TEX_FLAT = _flat(TEX)
for name, pat in PRESENCE:
    if re.search(pat, TEX_FLAT) or re.search(pat, TEX):
        print(f"  ok  {name}")
    else:
        failures.append(f"MISSING STATEMENT — {name} (/{pat}/ matches nothing)")

# --- retired numbers and phrasings that must NOT reappear -----------------------
RETIRED = [
    (r"37 of 42", "the pre-correction graceful count"),
    (r"37/42", "the pre-correction graceful count"),
    (r"\\le\$?5 points on 37", "the pre-correction abstract claim"),
    (r"CI 33--35", "an interval on a count whose cells are not independent"),
    (r"all (of them )?conservatively", "an algebraic identity of the one-sided rule"),
    (r"costliest.{0,40}all pass the one-sided test", "a second identity (price>0 forces passing)"),
    (r"solved,? (and )?nearly.free", "a 'solved problem' claim against 77 of 108 cells off target"),
    (r"says noth-?\s*\n?ing about what the miscalibration costs",
     "the refuted 'observable carries no information' claim"),
    (r"carries no information about the cost", "the refuted 'no information' claim in any phrasing"),
    (r"one-sided (test|criterion|target|rule)",
     "the retired mis-description: |FPR-alpha|<=eps is two-sided, just low-resolution below target"),
    (r"criterion defect", "the retired framing of the criterion as this paper's contribution"),
    (r"We show this criterion cannot fail",
     "claiming the criterion argument as a result (it is prior art: NIST/ASVspoof/NISTIR 8280)"),
    (r"inherits the (same )?blind spot", "the actDCF category error"),
    (r"more prevalent without", "causal attribution to disjointness"),
    (r"shape transfers and the\s+gain does not", "two-detector overgeneralization"),
    (r"mostly PSTN", "false destination-majority claim"),
    (r"Beta law under exchangeability", "exchangeability alone asserted to imply the Beta law"),
    (r"even under exchangeability, the induced FPR has a Beta law",
     "conditional-FPR Beta law attributed to exchangeability alone"),
    (r"threshold pinning", "residual per-deployment pinning language"),
    (r"Code and all result JSONs are released on publication",
     "reviewer artifact deferred until after review"),
    (r"p\{=\}0\.002", "invalid p-value"),
    (r"\[0\.12,\\,0\.87\]", "invalid R2 interval"),
    (r"cost becomes estimable", "unsupported estimability claim"),
    # --- cut with the 2026-09-09 protocol correction (AMENDMENT-3) ------------
    (r"R\^2[^0-9]{0,12}0\.59", "the exploratory cost map (now negative R2; cut)"),
    (r"0\.45--0\.64", "the cost map's delete-one range (cut)"),
    (r"\b66 (viable|cells)", "the cost map's cell count (cut)"),
    (r"220 (pairs|ordered)", "the crossed twin arm (cut)"),
    (r"145 (of|/) ?220", "the crossed twin arm count (cut)"),
    (r"Exploratory cost-map", "the cost-map paragraph (cut)"),
    (r"38--76\\%", "the detector-transfer envelope (cut)"),
    (r"99\\slash108", "the ACI oracle-label ceiling (no longer a ceiling; cut)"),
    (r"44\\times", "the GSM->none censored ratio (cut)"),
    (r"existence proof", "the GSM->none framing (cut)"),
    (r"GSM\$\\to\$none", "the GSM->none dissociating cell (cut)"),
    (r"-0\.65|-0\.07", "the weight-localisation correlations (cut)"),
    (r"54/108|54\\slash108", "the weight-localisation cell count (cut)"),
    (r"quadrupl", "the fold-ratio title, which one cell cannot support"),
    (r"(?i)manifest-bound", "release wording superseded by the URL in the Table 1 caption"),
    (r"pre-registered", "the text now says pre-specified"),
    (r"viability gate", "the text now says cutoff"),
    (r"measurable for SSL-AASIST only", "the retired abstract clause"),
    (r"confirm transport failure", "the retired replication phrasing"),
    (r"adding 5\\% spoofs", "the retired contamination phrasing"),
    (r"re-draw.{0,60}0\.13", "the provenance re-draw bound (cut)"),
    (r"first calibration run", "the retired sweep phrasing"),
    (r"\b4\.94\b|\b5\.03\b", "the all-phase mean FPR range"),
    (r"\+0\.55", "the all-phase flagship price"),
    (r"\b7\.03\b", "the all-phase speaker-disjoint width"),
    (r"MC SD 0\.17", "the all-phase speaker-disjoint SD"),
    (r"\b38\.4\b|\b36\.4\b|\b43\.5\b|\b15\.9\b|\b10\.5\b", "all-phase Table 1 values"),
    (r"(?<![\d.])74\\%|(?<![\d.])0\.47\\%|(?<![\d.])19\\% ", "the all-phase flagship triple"),
    (r"58 of 108|34 of the 84|60 of 84|22 of 84", "the all-phase grid counts"),
    (r"2,636", "the all-phase recording count"),
    (r"\b10\.08\b|SD 0\.30", "the single-cell speaker-diagnostic literals (now a six-cell range)"),
    (r"TPR 0\.89|FPR 0\.16|Spearman 0\.70", "the in-sample monitor operating point (cut; the pass is stated, not its numbers)"),
    (r"covers the eight cells", "the sweep-coverage sentence (cut)"),
    (r"We treat unlabeled drift monitoring as open", "the open-problem sentence (cut for space)"),
    (r"\\cite\{[^}]*\b(firc26|leroux25|rtcfake26|leong26|mcp25|negroni26|falsesafety26|cdts26)\b",
     "a citation dropped on 2026-09-09 for the page budget"),
]
print("\nretired-number checks:")
for pat, why in RETIRED:
    if re.search(pat, TEX):
        failures.append(f"RETIRED NUMBER back in the paper: /{pat}/ — {why}")
    else:
        print(f"  ok  absent: {why}")

# --- reader files: RV opens these first and they are pure prose ------------
# They were written once and the paper moved underneath them: on 2026-08-14 both
# still described the criterion framing as an outside reader's validated choice,
# three hours after that framing died on prior art. Prose that states a VERDICT
# needs a currency check; these assertions are it.
print("\nreader-file currency checks:")
READER_STALE = [
    (r"one-sided criterion", "retired mis-description of the additive band"),
    (r"identified the criterion defect.{0,40}as the transferable result",
     "the criterion framing presented as still live"),
    (r"criterion(?: defect)? (?:is|as) (?:the )?(?:transferable|main) (?:result|contribution)",
     "the criterion claimed as this paper's contribution"),
    (r"gpt-5\.6|xhigh|reviewer model", "a review named by its tool rather than its role"),
    (r"74/19/0\.47|0\.19 → 0\.74|\+0\.55", "the all-phase flagship triple"),
    (r"58 of 108|58/108|34 of the 84|34 of them|60/84|22/84", "the all-phase grid counts"),
    (r"4\.94|5\.03", "the all-phase mean FPR range"),
    (r"\b7\.03\b|MC SD 0\.17", "the all-phase speaker-disjoint width"),
    (r"R\^2=0\.59|0\.45--0\.64", "the cost map as a live result"),
    (r"2,636", "the all-phase recording count"),
    (r"36\.4/43\.5|15\.9/10\.5", "the all-phase Table 1 values"),
    (r"6\.4 (pp|points)", "the all-phase Gaussian worst miss"),
    (r"Five cells are resolution-limited|Five cells have", "the all-phase resolution-limited count"),
]
for fname in ("READING-MAP.md", "AUTHORS-DOUBTS.md"):
    doc = (DOCS / fname).read_text()
    for pat, why in READER_STALE:
        if re.search(pat, doc):
            failures.append(f"{fname}: STALE — {why} (/{pat}/ matches)")
    if fname == "READING-MAP.md" and "measurement study" not in doc:
        failures.append(f"{fname}: does not describe the paper as a measurement study")
    if "hidden" not in doc or "AMENDMENT-3" not in doc:
        failures.append(f"{fname}: does not describe the trial-phase protocol correction "
                        "(hidden phase, AMENDMENT-3)")
    if "must be corrected in the text" in doc and "0.975--0.997" in TEX and "17--93 pp" in TEX:
        failures.append(f"{fname}: STALE — reports the AUC/naive-transfer literals as still open after the text was fixed")
    for lit in (f"{n_miss} of {len(cells)}", f"{len(hidden)} of the {len(in_tol)}",
                f"{_F['vanilla_fpr_mean']*100:.2f}",
                f"{sum(_cors_n.values())} of {42 * len(_cors_n)}"):
        if lit not in doc:
            failures.append(f"{fname}: does not carry the current value {lit!r}")
    print(f"  ok  {fname} carries no retired framing")

# --- delivered review artifact: verify bytes, not a prose promise -----------
print("\nreview-artifact binding checks:")
ROOT = HERE.parent.parent


def verify_pdf_layout(pdf):
    """Bind the submission to the ICASSP 4+1 layout, including page-5 content."""
    if not pdf.is_file():
        failures.append(f"submission PDF missing: {pdf}")
        return
    try:
        info = subprocess.run(["pdfinfo", str(pdf)], capture_output=True,
                              text=True, check=True).stdout
        match = re.search(r"^Pages:\s+(\d+)$", info, re.MULTILINE)
        if match is None or int(match.group(1)) != 5:
            failures.append(f"submission PDF must have exactly 5 pages; pdfinfo says "
                            f"{match.group(1) if match else 'unknown'}: {pdf}")
            return
        page5 = subprocess.run(["pdftotext", "-f", "5", "-l", "5", str(pdf), "-"],
                               capture_output=True, text=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        failures.append(f"could not verify PDF page layout: {exc}")
        return
    flat = _flat(page5)
    # Page 5 must begin with the bibliography: its heading, or a numbered
    # reference when the heading fits at the end of page 4. Never continued
    # technical prose.
    if not re.match(r"^\s*(\d+\.\s*REFERENCES\s*)?\[\d+\]", page5):
        failures.append("page 5 does not begin with a numbered reference")
    for heading in ("DISCUSSION", "CONCLUSION"):
        if heading in flat:
            failures.append(f"technical section {heading} spills onto references-only page 5")
    print("  ok  PDF is exactly 5 pages and page 5 is references-only")


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_manifest_tree(package_root):
    manifest_path = package_root / "manifest.json"
    if not manifest_path.is_file():
        failures.append(f"review artifact missing manifest: {manifest_path}")
        return None
    manifest = json.loads(manifest_path.read_text())
    for relative, expected in manifest.items():
        path = package_root / relative
        if not path.is_file():
            failures.append(f"review artifact missing manifest entry: {relative}")
            continue
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            failures.append(f"review artifact hash/size mismatch: {relative}")
    return manifest


def verify_claim_map(claim_map):
    """The claim map must carry the CURRENT flagship and grid values, whatever its keys."""
    if not claim_map.is_file():
        failures.append("review artifact missing claim_map.json")
        return
    claims = json.loads(claim_map.read_text()).get("claims", {})
    flag = [v for k, v in claims.items() if k.startswith("flagship")]
    grid = [v for k, v in claims.items() if k.startswith("grid")]
    keys = ("vanilla_fnr_mean", "fnr_oracle", "vanilla_fpr_mean")
    if not flag or any(flag[0].get("values", {}).get(k) != _F[k] for k in keys):
        failures.append("review artifact claim map: flagship values differ from results_drift.json")
    if not grid or grid[0].get("miss") != n_miss or grid[0].get("hidden") != len(hidden):
        failures.append(f"review artifact claim map: grid claim does not state {n_miss}/{len(hidden)}")
    for key in ("a5_recording_speaker_disjoint_187_of_264", "quantile_full_grid",
                "score_weighting_heuristic"):
        if key not in claims:
            failures.append(f"review artifact claim map missing: {key}")


if (ROOT / "manifest.json").is_file():
    # Running from a clean extracted package: self-authenticate it.
    verify_pdf_layout(HERE / "main.pdf")
    manifest = verify_manifest_tree(ROOT)
    if manifest is not None:
        verify_claim_map(ROOT / "claim_map.json")
        print(f"  ok  extracted package authenticates {len(manifest)} manifest entries")
else:
    verify_pdf_layout(HERE / "main.pdf")
    package_root = E102 / "audit_package"
    manifest = verify_manifest_tree(package_root)
    required = ("main.pdf", "main.tex", "refs.bib", "spconf.sty", "IEEEbib.bst",
                "figs/drift.pdf", "check_numbers.py", "number_census.py")
    if manifest is not None:
        for name in required:
            relative = f"paper/A2/{name}"
            live = HERE / name
            packaged = package_root / relative
            if relative not in manifest:
                failures.append(f"review artifact does not bind {relative}")
            elif not live.is_file() or not packaged.is_file() or sha256(live) != sha256(packaged):
                failures.append(f"review artifact stale against live file: {relative}")
        verify_claim_map(package_root / "claim_map.json")
        if not any("review artifact" in failure for failure in failures):
            print("  ok  manifest passes and exact PDF/source/guards match the live submission")

print()
if failures:
    print(f"FAILED ({len(failures)}):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("all checks pass")
