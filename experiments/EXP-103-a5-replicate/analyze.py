"""EXP-103 readings 2a and 2b, against the bars frozen in PREREG.md.

2a  Realized FPR at a transported threshold across the 132 ordered
    calibrate->deploy condition pairs, on the TWIN-FREE subset (source
    recordings appearing under exactly one condition), speaker-disjoint by
    seeded hash on top of that.

2b  The severity->price map fitted on 21LA and applied entirely held out to A5.

Two arms are reported separately and never pooled:

  twin-free   80.5% of sources, one condition each -> the replicate proper
  crossed     19.5% of sources, 11 conditions each -> 21LA's twin structure on
              the same corpus, which measures what the twinning is worth

Per the gate (precheck.json): SSL-AASIST is viable on A5 (oracle FNR 7.9%);
AASIST is overlap-dominated (63.9%) and is therefore excluded from viable
claims by A2's own definition, exactly as BRSpeech and ITW cells are. Its
FPR-severity numbers are still reported -- that axis needs no spoof labels.
"""

import glob
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

A5 = Path.home() / "data/corpora/anti-spoofing/asvspoof5"
PROTO = A5 / "ASVspoof5.eval.track_1.tsv"
RAW = Path.home() / "icassp-runs/EXP-103-a5-replicate/raw"
OUT = Path(__file__).parent / "artifacts"
ALPHA = 0.05
N_CAL = 500
B = 1000
SEED = 103
OVERLAP_BAR = 0.50


def load_scores(model):
    out = {}
    for f in sorted(glob.glob(str(RAW / model / "*.npz"))):
        with np.load(f, allow_pickle=True) as z:
            for u, s in zip(z["utt_id"], z["score"]):
                out[str(u)] = float(s)
    return out


def half(speaker):
    """Stable 50/50 speaker split, so calibration and deployment never share a
    speaker. Hash rather than index so the split does not depend on ordering."""
    return int(hashlib.md5(f"{SEED}:{speaker}".encode()).hexdigest(), 16) & 1


def build():
    rows = [l.split() for l in open(PROTO)]
    src_conds = defaultdict(set)
    for r in rows:
        if r[8] == "bonafide":
            src_conds[r[5]].add(r[3])
    twin_free = {s for s, c in src_conds.items() if len(c) == 1}
    return rows, twin_free


def cells(rows, scores, twin_free, arm):
    """(condition, half) -> (bona scores, spoof scores) for the chosen arm."""
    keep = (lambda src: src in twin_free) if arm == "twin_free" else (lambda src: src not in twin_free)
    d = defaultdict(lambda: ([], []))
    for r in rows:
        spk, utt, cond, src, lab = r[0], r[1], r[3], r[5], r[8]
        if lab == "bonafide" and not keep(src):
            continue
        s = scores.get(utt)
        if s is None:
            continue
        d[(cond, half(spk))][0 if lab == "bonafide" else 1].append(s)
    return {k: (np.array(v[0]), np.array(v[1])) for k, v in d.items()}


def log2_ratio(fpr, n_dep):
    """Rule-of-three floor, as EXP-102."""
    return float(np.log2(max(fpr, 3.0 / max(n_dep, 1)) / ALPHA))


def run_pairs(cl, rng):
    conds = sorted({c for c, _ in cl})
    res = {}
    for c_cal in conds:
        for c_dep in conds:
            if c_cal == c_dep:
                continue
            # calibrate on half 0, deploy on half 1: speaker-disjoint
            cal_b = cl.get((c_cal, 0), (np.array([]), np.array([])))[0]
            dep_b, dep_s = cl.get((c_dep, 1), (np.array([]), np.array([])))
            if len(cal_b) < N_CAL + 50 or len(dep_b) < 50 or len(dep_s) < 50:
                continue
            k = int(np.floor((N_CAL + 1) * ALPHA))
            fprs, fnrs = [], []
            for _ in range(B):
                coh = cal_b[rng.choice(len(cal_b), N_CAL, replace=False)]
                t = np.sort(coh)[k - 1]
                fprs.append(float(np.mean(dep_b < t)))
                fnrs.append(float(np.mean(dep_s >= t)))
            fpr = float(np.mean(fprs))
            t_or = float(np.quantile(dep_b, ALPHA))
            fnr_or = float(np.mean(dep_s >= t_or))
            res[f"{c_cal}->{c_dep}"] = {
                "fpr": fpr, "log2_fpr_ratio": log2_ratio(fpr, len(dep_b)),
                "fnr": float(np.mean(fnrs)), "fnr_oracle": fnr_or,
                "fnr_price": float(np.mean(fnrs)) - fnr_or,
                "oracle_overlap_dominated": bool(fnr_or > OVERLAP_BAR),
                "n_dep_bona": len(dep_b), "n_dep_spoof": len(dep_s),
            }
    return res


def main():
    OUT.mkdir(exist_ok=True)
    rows, twin_free = build()
    print(f"twin-free bona-fide sources: {len(twin_free):,}")
    allres = {}
    for model in ("ssl", "aasist"):
        sc = load_scores(model)
        print(f"\n=== {model}: {len(sc):,} scores ===")
        for arm in ("twin_free", "crossed"):
            rng = np.random.default_rng(SEED)
            cl = cells(rows, sc, twin_free, arm)
            r = run_pairs(cl, rng)
            allres[f"{model}/{arm}"] = r
            if not r:
                print(f"  {arm}: no usable pairs")
                continue
            lr = np.array([v["log2_fpr_ratio"] for v in r.values()])
            od = sum(1 for v in r.values() if v["oracle_overlap_dominated"])
            print(f"  {arm}: {len(r)} pairs | "
                  f"|log2 ratio| >1 (2x miss): {int((np.abs(lr) > 1).sum())} | "
                  f"conservative: {int((lr < 0).sum())} | "
                  f"overlap-dominated cells: {od}")
            print(f"    median signed log2 ratio {np.median(lr):+.3f}, "
                  f"range [{lr.min():+.2f}, {lr.max():+.2f}]")
    (OUT / "results_a5.json").write_text(json.dumps(allres, indent=1))
    print(f"\nwrote {OUT/'results_a5.json'}")


if __name__ == "__main__":
    main()
