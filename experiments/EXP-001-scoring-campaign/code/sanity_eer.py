"""EXP-001 sanity: pooled EERs of the reproduced score files vs published values.

21LA/21DF filtered to eval-phase trials (official protocol); other corpora pooled.
"""

import csv
import gzip
import json
from pathlib import Path

import numpy as np

EXP = Path(__file__).resolve().parents[1]
DATA = Path.home() / "data/corpora/anti-spoofing"


def eer(bona, spoof):
    scores = np.concatenate([bona, spoof])
    labels = np.concatenate([np.ones(len(bona)), np.zeros(len(spoof))])
    o = np.argsort(scores)
    l = labels[o]
    frr = np.cumsum(l) / l.sum()
    far = 1 - np.cumsum(1 - l) / (1 - l).sum()
    i = np.argmin(np.abs(frr - far))
    return 100 * float((frr[i] + far[i]) / 2)


def phase_map(key_path):
    return {p[1]: p[7] for p in
            (l.split() for l in Path(key_path).read_text().splitlines())}


def main():
    phases = {"asv21la": phase_map(DATA / "keys/LA/CM/trial_metadata.txt"),
              "asv21df_100k": phase_map(DATA / "DF-keys-full/keys/DF/CM/trial_metadata.txt")}
    out = {}
    for f in sorted((EXP / "scores").glob("*.csv.gz")):
        name = f.stem.replace(".csv", "")
        corpus = name.split("_", 1)[1]
        with gzip.open(f, "rt") as fh:
            rows = list(csv.DictReader(fh))
        if corpus in phases:
            rows = [r for r in rows if phases[corpus].get(r["utt_id"]) == "eval"]
        s = np.array([float(r["score"]) for r in rows])
        b = np.array([r["label"] == "bonafide" for r in rows])
        out[name] = {"n": len(rows), "eer": round(eer(s[b], s[~b]), 3)}
        print(name, out[name], flush=True)
    (EXP / "sanity_eer.json").write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
