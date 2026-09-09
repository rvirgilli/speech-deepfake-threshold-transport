"""Pre-check: is ASVspoof 5 eval overlap-dominated for our 19LA-trained detectors?

The branch this decides is frozen in PREREG.md before the run:

  either detector's oracle FNR at 5% FPR <= 50%  -> proceed to readings 2a and 2b
  both exceed 50%                                -> A5 is overlap-dominated, the
                                                    PRICE axis dies, the
                                                    FPR-SEVERITY axis still runs

The detectors are trained on 2019 LA; A5's eval carries unseen attacks over
different source speech, so this is a real possibility and not a formality.
1,500 bona fide + 1,500 spoof from the no-codec condition, ~2 min GPU.
"""

import json
import random
import sys
from pathlib import Path

import numpy as np

A5 = Path.home() / "data/corpora/anti-spoofing/asvspoof5"
PROTO = A5 / "ASVspoof5.eval.track_1.tsv"
AUDIO = A5 / "flac_E_eval"
OUT = Path(__file__).parent / "artifacts"
N_PER_CLASS = 1500
SEED = 103
ALPHA = 0.05
OVERLAP_BAR = 0.50   # PREREG: both detectors above this => overlap-dominated


BATCH = 32


def score_batched(scorer, paths, batch=BATCH):
    """Score in fixed batches.

    `score_files` stacks EVERY path into one tensor and runs a single forward
    pass -- it is built for short lists. Handing it 3,000 paths asked XLS-R's
    conv front-end for a 73.92 GiB allocation and killed job 129 after 92 s.
    The audio length was never the problem: the adapter already crops to
    NUM_SAMPLES=64,600 via load_fixed_waveform.
    """
    assert batch <= 64, f"batch {batch} is large for a 3090 with XLS-R"
    out = []
    for i in range(0, len(paths), batch):
        out.extend(scorer.score_files(paths[i:i + batch]))
        if (i // batch) % 20 == 0:
            print(f"    {i + min(batch, len(paths) - i)}/{len(paths)}", flush=True)
    return out


def oracle_fnr_at_alpha(bona, spoof):
    """FNR at the threshold realizing alpha on THIS deployment's bona fide.

    Scorer contract: higher = bona fide, so a deployment flags spoof below the
    threshold and the alpha-quantile of bona fide is the oracle operating point.
    """
    t = float(np.quantile(bona, ALPHA))
    return float(np.mean(spoof >= t)), t


def main():
    rows = [l.split() for l in open(PROTO)]
    # column 3 is the codec condition, 8 the label (verified against the file)
    nocodec = [r for r in rows if r[3] == "-"]
    bona = [r[1] for r in nocodec if r[8] == "bonafide"]
    spoof = [r[1] for r in nocodec if r[8] == "spoof"]
    rng = random.Random(SEED)
    bona = rng.sample(bona, min(N_PER_CLASS, len(bona)))
    spoof = rng.sample(spoof, min(N_PER_CLASS, len(spoof)))
    print(f"no-codec condition: sampling {len(bona)} bona fide, {len(spoof)} spoof",
          flush=True)

    paths = [str(AUDIO / f"{u}.flac") for u in bona + spoof]
    missing = [p for p in paths[:50] if not Path(p).exists()]
    assert not missing, f"audio not found, e.g. {missing[0]}"

    sys.path.insert(0, str(Path.home() / "projects/voxtech/deepfake-model-assessment"))
    from dfeval.references.aasist import AASISTScorer
    from dfeval.references.ssl_antispoof import SSLAntiSpoofScorer

    results = {}
    for tag, cls in (("ssl", SSLAntiSpoofScorer), ("aasist", AASISTScorer)):
        print(f"\nscoring with {tag} ...", flush=True)
        scores = np.asarray(score_batched(cls(), paths), dtype=float)
        b, s = scores[:len(bona)], scores[len(bona):]
        fnr, thr = oracle_fnr_at_alpha(b, s)
        results[tag] = {"oracle_fnr_at_5pct_fpr": fnr, "threshold": thr,
                        "bona_median": float(np.median(b)),
                        "spoof_median": float(np.median(s)),
                        "n_bona": len(b), "n_spoof": len(s)}
        print(f"  bona median {np.median(b):+.3f}  spoof median {np.median(s):+.3f}")
        print(f"  oracle FNR at {ALPHA:.0%} FPR: {fnr*100:.1f}%"
              f"  ({'overlap-dominated' if fnr > OVERLAP_BAR else 'viable'})")

    both_dominated = all(v["oracle_fnr_at_5pct_fpr"] > OVERLAP_BAR for v in results.values())
    verdict = "OVERLAP-DOMINATED" if both_dominated else "VIABLE"
    results["verdict"] = verdict
    print(f"\nPRE-REGISTERED BRANCH: {verdict}")
    if both_dominated:
        print("  Both detectors exceed 50%. The PRICE axis dies on A5 with these")
        print("  checkpoints; the FPR-severity axis still runs (PREREG.md).")
        print("  Next per PREREG: try XLS-R+SLS or XLSR-Mamba, else the ITW grid.")
    else:
        print("  Proceed to readings 2a (132-pair severity map) and 2b (held-out")
        print("  severity->price transfer from 21LA).")

    OUT.mkdir(exist_ok=True)
    (OUT / "precheck.json").write_text(json.dumps(results, indent=1))
    print(f"\nwrote {OUT / 'precheck.json'}")


if __name__ == "__main__":
    main()
