"""EXP-102 cell 1 (D6): score XLS-R+SLS on 19LA dev + BRSpeech-DF test.

Uses the vendored official SLS adapter (deepfake-model-assessment
dfeval.references.SLSScorer; checkpoint asvdf_sls_best.pth). Resumable chunking
as in EXP-001's score_dump.py; no embeddings needed. Sanity mode scores a
500-utt 21LA subset and compares per-trial against the official released
scores_LA.txt (agreement, not EER reproduction — stronger and cheaper).
Output: <out>/sls_<corpus>.csv.gz in EXP-001 format (utt_id,score,label).
"""

import argparse
import csv
import gzip
import sys
from pathlib import Path

import numpy as np
import torch

VOX = Path.home() / "projects/voxtech/deepfake-model-assessment"
sys.path.insert(0, str(VOX))

from dfeval.references.sls import SLSScorer  # noqa: E402
from dfeval.references.aasist import NUM_SAMPLES, load_fixed_waveform  # noqa: E402

MANIFESTS = Path.home() / "exp-artifacts/icassp2027/EXP-001/manifests"
OFFICIAL_LA = Path.home() / "data/corpora/anti-spoofing/official-scores/xlsr-sls/scores_LA.txt"
CHUNK = 1000


def read_manifest(name):
    with open(MANIFESTS / f"{name}.csv") as f:
        return [(r["utt_id"], r["path"], r["label"]) for r in csv.DictReader(f)]


def score_rows(scorer, rows, batch):
    scores, utts, labels = [], [], []
    for i in range(0, len(rows), batch):
        block = rows[i:i + batch]
        wavs, ids, labs = [], [], []
        for utt, path, lab in block:
            try:
                wavs.append(load_fixed_waveform(path, NUM_SAMPLES))
                ids.append(utt)
                labs.append(lab)
            except Exception as e:
                print(f"SKIP {utt}: {e}", flush=True)
        if not wavs:
            continue
        x = torch.stack(wavs).to(scorer.device)
        with torch.inference_mode():
            lp = scorer.model(x)
        scores.append(lp[:, 1].float().cpu().numpy())
        utts.extend(ids)
        labels.extend(labs)
    return utts, np.concatenate(scores), labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", action="append", required=True)
    ap.add_argument("--out", default=str(Path(__file__).parent / "scores"))
    ap.add_argument("--batch-size", type=int, default=24)
    ap.add_argument("--sanity", action="store_true")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    scorer = SLSScorer()
    print(f"SLS loaded on {scorer.device}", flush=True)

    if args.sanity:
        rows = read_manifest("asv21la")
        rng = np.random.default_rng(0)
        sub = [rows[i] for i in rng.choice(len(rows), 500, replace=False)]
        utts, s, _ = score_rows(scorer, sub, args.batch_size)
        off = dict(l.split()[:2] for l in OFFICIAL_LA.read_text().splitlines())
        both = [(x, float(off[u])) for u, x in zip(utts, s) if u in off]
        a, b = map(np.array, zip(*both))
        corr = float(np.corrcoef(a, b)[0, 1])
        mae = float(np.mean(np.abs(a - b)))
        print(f"SANITY n={len(both)} corr={corr:.5f} mae={mae:.4f}", flush=True)
        assert corr > 0.99, "per-trial agreement with official scores failed"

    for corpus in args.corpus:
        dst = out / f"sls_{corpus}.csv.gz"
        rows = read_manifest(corpus)
        done = set()
        part = out / f"sls_{corpus}.partial.csv"
        if part.exists():
            with open(part) as f:
                done = {l.split(",")[0] for l in f}
        todo = [r for r in rows if r[0] not in done]
        print(f"{corpus}: {len(rows)} rows, {len(todo)} to score", flush=True)
        with open(part, "a") as f:
            for i in range(0, len(todo), CHUNK):
                utts, s, labs = score_rows(scorer, todo[i:i + CHUNK], args.batch_size)
                for u, x, lab in zip(utts, s, labs):
                    f.write(f"{u},{x},{lab}\n")
                f.flush()
                if i % 10000 < CHUNK:
                    print(f"{corpus}: {i + CHUNK}/{len(todo)}", flush=True)
        with open(part) as f, gzip.open(dst, "wt") as g:
            g.write("utt_id,score,label\n")
            g.writelines(f)
        print(f"wrote {dst}", flush=True)


if __name__ == "__main__":
    main()
