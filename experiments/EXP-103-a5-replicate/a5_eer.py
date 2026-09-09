"""Pooled EER on the ASVspoof 5 deployment half used by the twin-free replication.

Same roster as analyze.py: bona-fide sources appearing under one condition
(twin-free arm), deployment speaker half `half(speaker) == 1`, all twelve
conditions pooled. Spoof trials are those of the same deployment speakers. Reads
the exported score tables in artifacts/. This is a discrimination reference for
the replication set, not a full-challenge protocol EER.
"""

import csv
import gzip
import json
from pathlib import Path

import numpy as np

from analyze import build, half

HERE = Path(__file__).parent


def eer(bona, spoof):
    ts = np.unique(np.concatenate([bona, spoof]))
    fpr = np.searchsorted(np.sort(bona), ts, side="left") / len(bona)
    fnr = 1 - np.searchsorted(np.sort(spoof), ts, side="left") / len(spoof)
    i = int(np.argmin(np.abs(fpr - fnr)))
    return float((fpr[i] + fnr[i]) / 2)


def main():
    rows, twin_free = build()
    out = {}
    for model in ("ssl", "aasist"):
        with gzip.open(HERE / "artifacts" / f"scores_{model}_asv5_eval.csv.gz", "rt") as f:
            sc = {r["utt_id"]: float(r["score"]) for r in csv.DictReader(f)}
        bona, spoof = [], []
        for r in rows:
            spk, utt, src, lab = r[0], r[1], r[5], r[8]
            if half(spk) != 1 or utt not in sc:
                continue
            if lab == "bonafide":
                if src in twin_free:
                    bona.append(sc[utt])
            else:
                spoof.append(sc[utt])
        out[model] = {"n_bona": len(bona), "n_spoof": len(spoof), "eer": round(eer(np.array(bona), np.array(spoof)), 4)}
        print(model, out[model], flush=True)
    (HERE / "artifacts" / "a5_eer.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
