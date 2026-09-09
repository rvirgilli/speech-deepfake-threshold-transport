"""Export the ASVspoof 5 trial scores used by analyze.py as public score tables.

Reads the same raw chunk dumps analyze.py reads and writes
artifacts/scores_<model>_asv5_eval.csv.gz (utt_id, score, label) with labels from
the EXP-001 asv5_eval manifest. No audio or embeddings are written.
"""

import csv
import glob
import gzip
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
RAW = Path.home() / "icassp-runs/EXP-103-a5-replicate/raw"
MANIFEST = Path("/home/rv/exp-artifacts/icassp2027/EXP-001/manifests/asv5_eval.csv")


def main():
    with open(MANIFEST) as f:
        labels = {r["utt_id"]: r["label"] for r in csv.DictReader(f)}
    for model in ("ssl", "aasist"):
        rows = []
        for path in sorted(glob.glob(str(RAW / model / "*.npz"))):
            with np.load(path, allow_pickle=True) as z:
                rows.extend(zip(z["utt_id"], z["score"]))
        out = HERE / "artifacts" / f"scores_{model}_asv5_eval.csv.gz"
        with gzip.open(out, "wt", newline="") as f:
            w = csv.writer(f)
            w.writerow(["utt_id", "score", "label"])
            for u, s in rows:
                w.writerow([str(u), f"{float(s):.6f}", labels[str(u)]])
        print(model, len(rows), "trials ->", out.name)


if __name__ == "__main__":
    main()
