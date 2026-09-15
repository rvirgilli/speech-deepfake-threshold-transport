"""Recompute every EXP-110 value the manuscript prints, from the released results.

The aggregation order is the part a reader cannot infer from the numbers alone, so it
is executed here rather than described:

  K            counted per checkpoint over the 42 ordered 21LA condition pairs
  K_median     median over that run's five selected checkpoints
  arm range    SSL-AASIST: min/max of the three run medians, one per seed
               AASIST:     min/max of the five checkpoint K values of its single run
  excess FNR   maximum over every cell of every selected checkpoint in the arm,
               and over all three seeds for SSL-AASIST

The two ranges are therefore different objects: SSL-AASIST's spans seeds, AASIST's
spans checkpoints. The registered material-reduction rule is evaluated on AASIST's
selected-checkpoint ranges, never on a range of run medians.

Run from this directory. Exits non-zero if any printed value no longer reproduces.
"""

import json
import statistics as st
import sys

SSL = {"arm1": ["results_arm1.json", "results_arm1_s1235.json", "results_arm1_s1236.json"],
       "arm2": ["results_arm2.json", "results_arm2_s1235.json", "results_arm2_s1236.json"]}
AASIST = {arm: f"results_aasist_{arm}_s1234.json" for arm in ("arm1", "arm2")}

# What section 4 and section 6 of the manuscript print.
EXPECTED = {"ssl_arm1_range": (30, 33), "ssl_arm2_range": (16, 23),
            "ssl_arm1_maxfnr": 17.50, "ssl_arm2_maxfnr": 0.55,
            "aasist_arm1_median": 30, "aasist_arm2_median": 22,
            "aasist_arm1_ckpt_range": (28, 33), "aasist_arm2_ckpt_range": (18, 27),
            "aasist_arm1_maxfnr": 56.1, "aasist_arm2_maxfnr": 69.4}


def checkpoint_ks(doc):
    return sorted(ep["K"] for ep in doc["epochs"].values())


def max_excess_fnr(docs):
    return max(c.get("fnr_price", 0) * 100
               for d in docs for ep in d["epochs"].values() for c in ep["cells"].values())


def main():
    got = {}
    for arm, files in SSL.items():
        docs = [json.load(open(f)) for f in files]
        medians = [st.median(checkpoint_ks(d)) for d in docs]
        got[f"ssl_{arm}_range"] = (int(min(medians)), int(max(medians)))
        got[f"ssl_{arm}_maxfnr"] = round(max_excess_fnr(docs), 2)
    for arm, path in AASIST.items():
        doc = json.load(open(path))
        ks = checkpoint_ks(doc)
        got[f"aasist_{arm}_median"] = int(st.median(ks))
        got[f"aasist_{arm}_ckpt_range"] = (min(ks), max(ks))
        got[f"aasist_{arm}_maxfnr"] = round(max_excess_fnr([doc]), 1)

    bad = {k: (EXPECTED[k], got[k]) for k in EXPECTED if EXPECTED[k] != got[k]}
    for k in sorted(EXPECTED):
        mark = "FAIL" if k in bad else "ok  "
        print(f"  {mark} {k:26} expected {EXPECTED[k]!s:12} got {got[k]}")

    # The registered rule, evaluated on the selected-checkpoint ranges it names.
    lo1, hi1 = got["aasist_arm1_ckpt_range"]
    lo2, hi2 = got["aasist_arm2_ckpt_range"]
    separated = hi2 < lo1
    decrease = got["aasist_arm1_median"] - got["aasist_arm2_median"]
    wider = max(hi1 - lo1, hi2 - lo2)
    print(f"\n  registered rule: separation {separated}; median decrease {decrease}; "
          f"wider within-arm range width {wider}; rule met {separated and decrease > wider}")
    assert separated and decrease == 8 and wider == 9 and not (decrease > wider), \
        "the registered rule's inputs changed; the manuscript's verdict must be rechecked"

    if bad:
        print(f"\n{len(bad)} manuscript value(s) no longer reproduce", file=sys.stderr)
        return 1
    print("\nevery manuscript value reproduces from the released results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
