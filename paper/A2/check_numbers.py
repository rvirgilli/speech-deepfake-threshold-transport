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

Run from paper/A2/. Exit 1 on any failure.
"""

import json
import hashlib
import math
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
EXP = HERE.parent.parent / "experiments"
TEX = (HERE / "main.tex").read_text()

drift = json.load(open(EXP / "EXP-102-a2-campaign/results_drift.json"))
countboot = json.load(open(EXP / "EXP-102-a2-campaign/results_count_bootstrap.json"))

ALPHA, SEV_BAR, OLD_RULE = 0.05, 1.0, 0.05
cells = {k: v for sec in ("within", "cross") for k, v in drift[sec].items()}
old_pass = [v for v in cells.values() if v["excursion"] <= OLD_RULE]
hidden = [v for v in old_pass if abs(v["log2_fpr_ratio"]) > SEV_BAR]
# Spoof-side quantities (price, actDCF) exclude overlap-dominated cells, which is
# the paper's own viability rule; two of the 34 are. The 34 is an FPR-axis count
# and stays. Keeping the two sets distinct here is what stops the axis boundary
# slipping in the guard the way it slipped in the prose.
hidden_viable = [v for v in hidden if v["fnr_oracle"] <= 0.50]
worst = max(cells.items(), key=lambda kv: kv[1]["fnr_price"])

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


# --- values that must match the regenerated drift map -----------------------
print("position-anchored value checks:")
n_miss = sum(1 for v in cells.values() if abs(v['log2_fpr_ratio']) > SEV_BAR)
WORDS = {58: "Fifty-eight"}
check("108-cell decomposition is stated", "The measurement covers 108 deployment cells",
      "two detectors $\\times$ (42 ordered pairs of seven 21LA channel "
      "conditions $+$ 12 ordered pairs of four corpora)",
      "84 within + 24 cross", "results_drift.json cell keys")
check("total miss count (abstract)", "Across 108 deployment cells",
      f"{n_miss} miss target",
      n_miss, "results_drift.json")
check("hidden-cell count (abstract)", "Across 108 deployment cells",
      f"{len(hidden)} are conservative failures", f"{len(hidden)}/{len(old_pass)}", "results_drift.json")
check("hidden-cell count (experiments)", "\\textbf{Drift} is detector-conditioned",
      f"{len(hidden)} of the {len(old_pass)} cells",
      f"{len(hidden)}/{len(old_pass)}", "results_drift.json")
# The cell the paper NAMES must be the costliest one it is allowed to name:
# resolution-limited cells (FPR resting on a handful of utterances) are excluded
# by the correctness audit, so citing the raw argmax would be wrong.
quotable = max((kv for kv in cells.items() if not kv[1]["resolution_limited"]),
               key=lambda kv: kv[1]["fnr_price"])
det, pair = quotable[0].split("/")
src, dst = pair.split("->")
CODEC_TEX = {"g722": "G.722", "none": "none", "pstn": "PSTN", "gsm": "GSM",
             "opus": "Opus", "alaw": "A-law", "ulaw": "$\\mu$-law"}
check("worst spoof-side price (value)", "Their spoof-side cost",
      f"$+{quotable[1]['fnr_price']:.2f}$", quotable[0], "max quotable fnr_price")
check("worst spoof-side price (cell named correctly)",
      "Their spoof-side cost",
      f"{det.upper().replace('AASIST', 'AASIST')}, {CODEC_TEX[src]}$\\to${CODEC_TEX[dst]}",
      quotable[0], "argmax fnr_price among non-resolution-limited cells")
if worst[0] != quotable[0]:
    print(f"  note: raw argmax is {worst[0]} ({worst[1]['fnr_price']:+.3f}) but it is "
          f"resolution-limited, so the paper must not name it")
if any(v["resolution_limited"] and f"{v['fnr_price']:.2f}" in TEX for v in cells.values()):
    pass  # value collisions are possible; the named-cell check above is the binding one
check("hidden-cell count (figure caption)", "\\caption{Drift map.",
      f"{len(hidden)} of the {len(old_pass)} band-passing cells", None, "results_drift.json")

# A count is only correct together with the set it was computed over. The audit
# of 2026-08-15 found the abstract asserting "all conservative" of the 58 cells
# missing by >2x, when only the 34 of them inside the band are -- 24 are liberal,
# the worst reading 8.6x target. The integer 58 was checked and passed; the
# quantifier attached to it was never looked at. So every headline count below is
# bound to its scope, and the scoping words are part of the check.
n_conservative = sum(1 for v in cells.values()
                     if abs(v["log2_fpr_ratio"]) > SEV_BAR and v["log2_fpr_ratio"] < 0)
n_liberal = sum(1 for v in cells.values()
                if abs(v["log2_fpr_ratio"]) > SEV_BAR and v["log2_fpr_ratio"] > 0)
assert n_conservative + n_liberal == n_miss, "conservative/liberal split must partition the misses"
check("the hidden >2x misses are scoped to the additive audit",
      "Across 108 deployment cells",
      f"{len(hidden)} are conservative failures that pass an additive $\\pm5$-point audit",
      f"{n_conservative} conservative / {n_liberal} liberal", "results_drift.json")
# The two sets of size 84 must each name itself; a bare "84" is ambiguous.
# Same class as the count-scope rule, found by enumerating siblings of the F1
# defect rather than waiting for another audit: a universal claim about what
# failed must say whose failures it is describing. Compression drops the
# qualifier, and "no monitor survives" reads as a claim about all monitors.
# By-construction claims must carry their forcedness WHERE THEY ARE MADE.
# Four were found in this paper: the conformal row hitting target (marginal rank),
# the +-alpha band admitting no liberal 2x miss, the price/severity correlation
# being two readouts of one scalar, and -- caught last, in the abstract -- the
# fact that a conservative cell cannot read further than alpha from target.
# A claim that could not have come out any other way is not evidence, and a
# reader who spots the forcedness discounts everything near it.
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

for phrase, need in ((r"no unlabeled drift monitor", "we tested"),
                     (r"every label-free correction", "we could pose")):
    for m in re.finditer(phrase + r"(.{0,40})", _flat(TEX)):
        if need not in m.group(1):
            failures.append(f"unscoped universal claim: {m.group(0)!r} -- needs {need!r}; "
                            "we tested some monitors, not all of them")

for m in re.finditer(r"the 84 (?:band-passing )?cells(.{0,30})", _flat(TEX)):
    if not re.match(r"\s*(passing|are off|are within-corpus)", m.group(1)):
        failures.append(f"an unqualified 'the 84 cells' at ...{m.group(0)!r}: 84 within-corpus "
                        "and 84 band-passing cells are different sets overlapping on 68 -- "
                        "name which")

# The paper deliberately attaches NO interval to the count: 68 of the 84 cells
# are within-corpus, where calibration and deployment are the same recordings, so
# the bootstrap (kept in results_count_bootstrap.json, labelled a lower bound)
# does not cover the dominant dependence. What must appear instead is the
# threshold-sensitivity range and the dependence disclosure -- asserted below.
for bar_label, key in (("60 of 84", 0.585), ("22 of 84", 2.0)):
    n = sum(1 for v in cells.values()
            if v["excursion"] <= OLD_RULE and abs(v["log2_fpr_ratio"]) > key)
    if f"{n} of 84" != bar_label:
        failures.append(f"sensitivity range stale: {2**key:.1f}x gives {n} of 84, "
                        f"paper says {bar_label}")

sev = drift["severity_vs_w1_transfer"]
check("corrected transfer R2", "\\caption{Drift map.",
      f"{sev['fit_ssl_test_aasist']['r2_transfer']:.2f}", None, "severity_vs_w1_transfer",
      chars=2200)  # the caption grew; the window must still end inside it

pre = drift["monitor_eval_PREREGISTERED"]["ssl/w1_mixture"]
check("pre-registered monitor TPR", "\\textbf{Monitoring} was a pre-registered",
      f"TPR {pre['achieves_tpr80_fpr20']['tpr']:.2f}", None, "monitor_eval_PREREGISTERED")
check("pre-registered monitor Spearman", "\\textbf{Monitoring} was a pre-registered",
      f"Spearman {pre['spearman_vs_target']:.2f}", None, "monitor_eval_PREREGISTERED")

# --- the ASVspoof 5 replicate, recomputed from its own artifacts ---------------
# The delta-check of 2026-08-15 found 28 of 28 mutations on this round's numbers
# passing silently: every new value could be deleted or corrupted and the suite
# still printed "all checks pass". Six checker edits that round were all
# reword-chases and none opened the new JSONs. These do.
A5 = json.load(open(EXP / "EXP-103-a5-replicate/artifacts/results_a5.json"))
PRE = json.load(open(EXP / "EXP-103-a5-replicate/artifacts/precheck.json"))
COST = json.load(open(EXP / "EXP-103-a5-replicate/artifacts/cost_map_sensitivity.json"))
BND = json.load(open(EXP / "EXP-403-a2-detector-families/artifacts/bound_transfer.json"))

def _miss_frac(section, bar=1.0):
    n = tot = 0
    for det in ("ssl", "aasist"):
        for v in A5[f"{det}/{section}"].values():
            tot += 1
            n += abs(v["log2_fpr_ratio"]) > bar
    return n, tot

_n_tf, _tot_tf = _miss_frac("twin_free")
_n_cr, _tot_cr = _miss_frac("crossed")
_la = sum(1 for k, v in drift["within"].items() if abs(v["log2_fpr_ratio"]) > SEV_BAR)
_la_tot = len(drift["within"])
print("\nASVspoof 5 replicate (recomputed):")
check("replicate miss count and rate", "Over both detectors the transported threshold",
      f"{_n_tf} of {_tot_tf} pairs ({round(100*_n_tf/_tot_tf)}\\%)",
      f"{_n_tf}/{_tot_tf}", "results_a5.json")
check("the grid it is contrasted against, at the SAME scope",
      "Over both detectors the transported threshold",
      f"against {round(100*_la/_la_tot)}\\% on\nthe grid above",
      f"{_la}/{_la_tot} pooled over both detectors", "results_drift.json")
check("abstract carries the pooled A5 count and denominator",
      "The failure persists on ASVspoof~5", f"{_n_tf} of {_tot_tf} ordered pairs",
      f"{_n_tf}/{_tot_tf}", "results_a5.json")
check("abstract carries the pooled A5 rate",
      "The failure persists on ASVspoof~5", f"({round(100*_n_tf/_tot_tf)}\\%)",
      f"{_n_tf}/{_tot_tf}", "results_a5.json")
check("twin arm, pooled like the primary", "recur under every",
      f"give {_n_cr} of {_tot_cr} ({round(100*_n_cr/_tot_cr)}\\%)", f"{_n_cr}/{_tot_cr}", "results_a5.json")
for tag, name in (("ssl", "SSL-AASIST"), ("aasist", "AASIST")):
    check(f"gate oracle FNR ({tag})", "AASIST is\noverlap-dominated on this corpus",
          f"{PRE[tag]['oracle_fnr_at_5pct_fpr']*100:.1f}\\%",
          tag, "precheck.json")
_cov = sorted(BND["coverage"].values())
check("detector-transfer envelope coverage", "Detector transfer also",
      f"covers only {_cov[0]*100:.0f}--{_cov[-1]*100:.0f}\\%",
      f"{[round(c,3) for c in _cov]}", "bound_transfer.json")
# The held-out point estimate is descriptive. The former Fisher interval and
# p-value were invalid for held-out R^2 over crossed directed conditions; this
# guard now requires the scope and forbids that inference from returning.
check("descriptive held-out R2", "Exploratory cost-map sensitivity",
      f"R^2\\,{COST['primary']['heldout_r2']:.2f}",
      COST['primary']['heldout_r2'], "cost_map_sensitivity.json")
_cost_loo = COST["delete_one_a5_condition_fixed_primary_line"]["range"]
check("delete-one-A5 cost-map sensitivity range", "Exploratory cost-map sensitivity",
      f"R^2={_cost_loo[0]:.2f}$--${_cost_loo[1]:.2f}",
      _cost_loo, "cost_map_sensitivity.json")
check("R2 is explicitly non-inferential", "Exploratory cost-map sensitivity",
      "We report these points descriptively, not as established inference",
      None, "adversarial estimator audit")
for bad, why in ((r"p\{=\}0\.002", "invalid p-value"),
                 (r"\[0\.12,\\,0\.87\]", "invalid R2 interval"),
                 (r"cost becomes estimable", "unsupported estimability claim")):
    if re.search(bad, TEX):
        failures.append(f"RETIRED INFERENCE back in paper: {why} (/{bad}/)")

# --- THE FLAGSHIP TRIPLE, ANCHORED AT BOTH SITES ------------------------------
# The census cannot catch these. It is value-based, so mutating 74 -> 84 or
# 0.47 -> 0.87 turns the flagship into a numeral that is legitimately accounted
# for elsewhere (84 within-corpus cells; the R^2 interval upper bound), and a
# census that asks "is this number accounted for" answers yes. Position is the
# only discriminator, which is what these anchors supply. Verified: without
# them, both mutations pass the combined suite silently.
_F = drift["within"]["aasist/pstn->g722"]
print("\nflagship triple, both sites:")
for site, anc in (("abstract", "We calibrate a speech-deepfake detector"),
                  ("Fig. 1 caption", "The costliest on the spoof side are among them")):
    check(f"flagship missed-spoof rate ({site})", anc,
          f"{_F['vanilla_fnr_mean']*100:.0f}\\% of spoofs",
          f"{_F['vanilla_fnr_mean']:.4f}", "results_drift.json")
    check(f"flagship oracle rate ({site})", anc,
          f"{_F['fnr_oracle']*100:.0f}\\%", f"{_F['fnr_oracle']:.4f}", "results_drift.json")
    check(f"flagship realized FPR ({site})", anc,
          f"{_F['vanilla_fpr_mean']*100:.2f}\\%", f"{_F['vanilla_fpr_mean']:.4f}",
          "results_drift.json")

# --- EVERY SITE, NOT THE SITE THAT BROKE ---------------------------------------
# Four rounds running, a fix landed in one site and not its twin: the contrast
# numbers, the interval sites, the beta figure, the actDCF denominator. Guarding
# "the pair that desynced" produced a different desynced pair each time. So each
# artifact-derived quantity is asserted at EVERY site it appears.
NSW = json.load(open(EXP / "EXP-102-a2-campaign/results_nsweep.json"))
PAR = json.load(open(EXP / "EXP-102-a2-campaign/results_parametric.json"))
SPK = json.load(open(EXP / "EXP-102-a2-campaign/artifacts/speaker_clustering.json"))

print("\nevery-site checks:")
# contamination -- never opened by the checker before, and the sentence it guards
# vanished once already and was self-refuting when restored
_cont = [e["N500_c0.05"]["fpr_mean"] for e in NSW["contamination"].values()
         if "N500_c0.05" in e]
_below = sorted(v for v in _cont if v < ALPHA)
check("contamination: how many cells fall below target",
      "adding 5\\% spoofs to the calibration cohort",
      f"on seven of eight cells ({_below[0]*100:.2f}--{_below[-1]*100:.2f}\\%)",
      f"{len(_below)}/{len(_cont)} below {ALPHA}", "results_nsweep.json")
assert len(_below) == 7, f"contamination: {len(_below)} cells below target, prose says seven"
check("contamination: the cell above target is named as near-inert",
      "adding 5\\% spoofs to the calibration cohort",
      f"nearly unchanged at {max(_cont)*100:.2f}\\%", f"max {max(_cont)}", "results_nsweep.json")

# the label sweep's coverage, and the two policies it is contrasted against
check("the sweep's cell count", "\\textbf{FPR control and FNR cost.}",
      f"covers all {len(NSW['n_sweep'])} cells".replace("8", "eight"),
      f"{len(NSW['n_sweep'])} cells in results_nsweep.json", "results_nsweep.json")
_z = [e["znorm"]["fpr_mean"] for N, e in NSW["n_sweep"]["ssl/itw"].items() if isinstance(e, dict)]
_g = [e["parametric"]["fpr_mean"] for N, e in PAR["ssl"]["itw"].items() if isinstance(e, dict)]
check("z-norm's ITW offset, recomputed", "\\textbf{FPR control and FNR cost.}",
      f"z-norm remains {round((min(_z)-ALPHA)*100)}--{round((max(_z)-ALPHA)*100)}\\,pp high",
      f"{min(_z):.3f}-{max(_z):.3f}", "results_nsweep.json")
check("the Gaussian's ITW offset, recomputed", "\\textbf{FPR control and FNR cost.}",
      f"Gaussian quantile remains {(min(_g)-ALPHA)*100:.1f}--{(max(_g)-ALPHA)*100:.1f}\\,pp high",
      f"{min(_g):.3f}-{max(_g):.3f}", "results_parametric.json")

# Exact Beta(k,N+1-k) dispersion for one deployed threshold. Integer-parameter
# Beta CDF equals a binomial upper tail, so this needs no scipy dependency.
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

_p100, _, _ = _beta_summary(100)
_p500, _q500lo, _q500hi = _beta_summary(500)
check("N=100 probability within +/-2 FPR points", "\\textbf{Calibration and detection are distinct.}",
      f"{_p100*100:.1f}\\% probability", _p100, "exact Beta law")
check("N=500 probability within +/-2 FPR points", "\\textbf{Calibration and detection are distinct.}",
      f"{_p500*100:.1f}\\%", _p500, "exact Beta law")
check("N=500 central 95% Beta interval", "\\textbf{Calibration and detection are distinct.}",
      f"{_q500lo*100:.2f}--{_q500hi*100:.2f}\\% central 95\\% interval",
      (_q500lo, _q500hi), "exact Beta law")
_spk = SPK["cells"]["aasist/pstn"]
check("speaker-disjoint flagship width mean", "A post-hoc speaker-disjoint diagnostic",
      f"{_spk['seed_sweep']['mean_width_pp']:.2f}",
      _spk['seed_sweep']['mean_width_pp'], "speaker_clustering.json")
check("speaker-disjoint flagship width SD", "A post-hoc speaker-disjoint diagnostic",
      f"MC SD {_spk['seed_sweep']['sd_width_pp']:.2f}",
      _spk['seed_sweep']['sd_width_pp'], "speaker_clustering.json")
check("speaker-permutation null median", "A post-hoc speaker-disjoint diagnostic",
      f"near {_spk['permutation']['null_median_pp']:.1f}",
      _spk['permutation']['null_median_pp'], "speaker_clustering.json")
if _spk["permutation"]["null_exceedances"] != 0:
    failures.append("speaker clustering: flagship no longer exceeds every permutation null")

# --- SCOPE WORDS AND DISQUALIFICATIONS -----------------------------------------
# The round that added the numeric guards left 9 of 32 mutations passing, and
# none of the survivors were numbers: the guards covered values thoroughly and
# scope words not at all, which is the inverse of where three rounds of damage
# actually occurred. Every item below is something a previous round RESTORED
# after it had silently vanished, so each can vanish again identically.
SCOPE_CRITICAL = [
    ("C5's disqualification, lost once with Table 2's footnote",
     "C5 AS-norm is excluded as degenerate"),
    ("the viability rule is named as ours, not borrowed",
     "our own viability rule excludes it"),
    ("the axis boundary itself -- the sentence the whole argument rests on",
     "the spoof-side axis here rests on one detector"),
    ("the cost-map detector is explicit",
     "A5 SSL-AASIST cells"),
    ("the cost map is descriptive, not inferential",
     "no valid crossed-condition interval is available"),
    ("the A5 rate contrast is not attributed to disjointness",
     "numerical rate difference is confounded by corpus, codecs and attacks"),
    ("the abstract distinguishes FPR replication from viable spoof cost",
     "Spoof-side cost there is measurable for SSL-AASIST only"),
    ("the finite-sample claim is marginal",
     "\\emph{marginal expected} FPR control"),
    ("the realized-threshold limitation is explicit",
     "This is not a per-deployment guarantee"),
    ("exchangeability is scoped to marginal rank validity",
     "Exchangeability gives marginal rank validity"),
    ("the conditional-FPR Beta law has its stronger assumptions",
     "conditional-FPR Beta law requires iid continuous sampling from a fixed distribution"),
]
print("\nscope words and disqualifications:")
for why, needle in SCOPE_CRITICAL:
    if _flat(needle) not in _flat(TEX):
        failures.append(f"SCOPE WORD GONE — {why}: {needle!r}")
    else:
        print(f"  ok  {why}")

# --- statements that must EXIST (the half that catches silent omissions) ----
PRESENCE = [
    # --- what the reframe must keep saying -------------------------------
    ("the paper disclaims both the rule and the audit practice as prior art",
     r"We claim neither the threshold rule nor\s+the practice of auditing both error rates"),
    ("the direct 2026 quantile prior art is cited", r"zhao26ca"),
    ("the direct 2026 fixed-threshold audit is cited", r"schaefer26reality"),
    ("the quantitative delta from the closest threshold-transfer audit is explicit",
     r"Relative to that audit's one detector and two target corpora,.{0,180}108-cell.{0,180}264-pair"),
    ("the 2x severity bar is disclosed as post-hoc in the main text",
     r"The \$2\\times\$ bar is post-hoc"),
    ("the 2x severity bar is disclosed as post-hoc in the figure caption",
     r"Under a\s+post-hoc \$2\\times\$ two-sided bar"),
    ("the +-5pp band is identified as OUR pre-registration, not the field's",
     r"passing our\s+pre-registered \$\\pm\$5\\,pp test"),
    ("the conservative direction is explained as forced, not reported as a finding",
     r"necessarily\s+conservatively, since that band excludes"),
    ("the measurable/unmeasurable asymmetry is stated",
     r"Spoof-side cost needs the unavailable\s+class; low FPR alone cannot distinguish control from collapse"),
    ("the sibling prior art on the negatives is cited and differentiated",
     r"falsesafety26\} finds distribution-free risk control overruns under group shift"),
    # --- disclosures carried over, all still load-bearing ----------------
    ("spoof-side cost is oracle-referenced, not calibration-referenced",
     r"against the oracle\s*\n?\s*threshold for that deployment"),
    ("both monitor readings are reported, not just the corrected one",
     r"We report both readings"),
    ("monitor's pre-registered pass is disclosed", r"met its pre-registered criteria"),
    ("no predictive law is claimed", r"no predictive law is claimed"),
    ("limitations section exists", r"\\textbf\{Limitations\.\}"),
    ("dependence unit is disclosed for the within-corpus cells",
     r"same 2,636 source recordings from the same 67 speakers under seven\s+21LA channel conditions"),
    ("non-independence is stated, not implied",
     r"not independent observations and no significance claim is made"),
    ("the count is given as a threshold-sensitivity range",
     r"60 at \$1\.5\\times\$, 22 at \$4\\times\$"),
    ("detector-conditioning is scoped, not claimed", r"\$n\{=\}2\$ cannot separate the explanations"),
    ("the cost is not called identifiable from bona-fide scores",
     r"does not make spoof-side cost identifiable from bona-fide scores alone"),
    ("the N=100 single-draw probability is stated",
     r"65\.5\\% probability of landing within \$\\pm2\$ FPR points"),
    ("the N=500 single-draw probability is stated", r"96\.2\\%"),
    ("the speaker-unit failure is disclosed", r"speaker-disjoint diagnostic on 21LA widened"),
    ("the speaker variability is labeled as Monte Carlo SD", r"\(MC SD 0\.17\)"),
    ("the dissociating cell is framed as an existence proof, not a rate",
     r"an existence proof,\s+not a rate"),
    ("the dissociating cell is disclosed as resolution-limited",
     r"resolution-limited SSL GSM\$\\to\$none cell.{0,100}rule-of-three-censored"),
]
print("\nrequired-presence checks:")
# Match against whitespace-normalised text for the same reason the value checks
# do: a line wrap must not read as a missing statement. Patterns may still use
# \s+ between words; a single space in the pattern matches any run.
TEX_FLAT = _flat(TEX)
for name, pat in PRESENCE:
    if re.search(pat, TEX_FLAT) or re.search(pat, TEX):
        print(f"  ok  {name}")
    else:
        failures.append(f"MISSING STATEMENT — {name} (/{pat}/ matches nothing)")

# --- retired numbers that must NOT reappear ---------------------------------
RETIRED = [
    (r"37 of 42", "the pre-correction graceful count"),
    (r"37/42", "the pre-correction graceful count"),
    (r"\\le\$?5 points on 37", "the pre-correction abstract claim"),
    (r"CI 33--35", "an interval on a count whose cells are not independent"),
    (r"all (of them )?conservatively", "an algebraic identity of the one-sided rule"),
    (r"costliest.{0,40}all pass the one-sided test", "a second identity (price>0 forces passing)"),
    (r"solved,? (and )?nearly.free", "a 'solved problem' claim against 58 of 108 cells off target"),
    (r"says noth-?\s*\n?ing about what the miscalibration costs",
     "the refuted 'observable carries no information' claim (W1 ranks |price| at 0.54/0.85)"),
    (r"carries no information about the cost",
     "the refuted 'no information' claim in any phrasing"),
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
for fname in ("READING-MAP.md", "AUTHORS-DOUBTS.md"):
    doc = (HERE / fname).read_text()
    for pat, why in [
        (r"one-sided criterion", "retired mis-description of the additive band"),
        (r"identified the criterion defect.{0,40}as the transferable result",
         "the criterion framing presented as still live"),
        (r"criterion(?: defect)? (?:is|as) (?:the )?(?:transferable|main) (?:result|contribution)",
         "the criterion claimed as this paper's contribution"),
    ]:
        if re.search(pat, doc):
            failures.append(f"{fname}: STALE — {why} (/{pat}/ matches)")
    if fname == "READING-MAP.md" and "measurement study" not in doc:
        failures.append(f"{fname}: does not describe the paper as a measurement study")
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
    # The heading may legally be at the end of page 4 when the bibliography
    # starts there. Page 5 itself must begin with a numbered reference, not
    # continued technical prose.
    if not re.match(r"^\s*\[\d+\]", page5):
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


if (ROOT / "manifest.json").is_file():
    # Running from a clean extracted package: self-authenticate it.
    verify_pdf_layout(HERE / "main.pdf")
    manifest = verify_manifest_tree(ROOT)
    if manifest is not None:
        print(f"  ok  extracted package authenticates {len(manifest)} manifest entries")
else:
    verify_pdf_layout(HERE / "main.pdf")
    package_root = EXP / "EXP-102-a2-campaign/audit_package"
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
        claim_map = package_root / "claim_map.json"
        if not claim_map.is_file():
            failures.append("review artifact missing claim_map.json")
        else:
            claims = json.loads(claim_map.read_text()).get("claims", {})
            for key in ("flagship_74_19_0.47", "grid_58_and_hidden_34",
                        "a5_recording_speaker_disjoint_187_of_264",
                        "quantile_full_grid", "score_weighting_heuristic"):
                if key not in claims:
                    failures.append(f"review artifact claim map missing: {key}")
        if not any("review artifact" in failure for failure in failures):
            print("  ok  manifest passes and exact PDF/source/guards match the live submission")

print()
if failures:
    print(f"FAILED ({len(failures)}):")
    for f in failures:
        print(f"  - {f}")
    sys.exit(1)
print("all checks pass")
