"""Build per-corpus scoring manifests (utt_id,path,label) for EXP-001.

The 21DF corpus is subsampled to 100k trials uniformly at random with the
pre-registered seed 20260813. All other corpora are complete.
"""

import csv
import os
import random
from pathlib import Path

DATA = Path(os.environ.get("A2_DATA", Path.home() / "data/corpora/anti-spoofing"))
CML = Path(os.environ.get("A2_CML", Path.home() / "data/corpora/speech-resources/BRSpeech_CML_TTS_v04012024"))
OUT = Path(os.environ.get("A2_MANIFESTS", Path.home() / "exp-artifacts/icassp2027/EXP-001/manifests"))
SEED = 20260813


def write(name, rows):
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{name}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["utt_id", "path", "label"])
        w.writerows(rows)
    print(name, len(rows))


def asv19(split):
    proto = (DATA / "ASVspoof2019/LA/ASVspoof2019_LA_cm_protocols"
             / f"ASVspoof2019.LA.cm.{split}.trl.txt")
    base = DATA / f"ASVspoof2019/LA/ASVspoof2019_LA_{split}/flac"
    rows = []
    for line in proto.read_text().splitlines():
        p = line.split()
        rows.append([p[1], str(base / f"{p[1]}.flac"), p[4]])
    return rows


def asv21(track, subsample=None):
    if track == "LA":
        key = DATA / "keys/LA/CM/trial_metadata.txt"
        base = DATA / "ASVspoof2021_LA_eval/flac"
    else:
        key = DATA / "DF-keys-full/keys/DF/CM/trial_metadata.txt"
        base = DATA / "ASVspoof2021_DF_eval/flac"
    rows = []
    for line in key.read_text().splitlines():
        p = line.split()
        rows.append([p[1], str(base / f"{p[1]}.flac"), p[5]])
    if subsample:
        rows = random.Random(SEED).sample(rows, subsample)
        rows.sort(key=lambda r: r[0])
    return rows


def itw():
    base = DATA / "release_in_the_wild"
    rows = []
    with open(base / "meta.csv") as f:
        for r in csv.DictReader(f):
            label = "bonafide" if r["label"] == "bona-fide" else "spoof"
            rows.append([Path(r["file"]).stem, str(base / r["file"]), label])
    return rows


def brspeech():
    rows = []
    for sysdir in sorted((DATA / "brspeech_df").iterdir()):
        for p in sorted((sysdir / "test").rglob("*.flac")):
            rows.append([f"{sysdir.name}/{p.stem}", str(p), "spoof"])
    with open(CML / "test.csv") as f:
        for r in csv.DictReader(f, delimiter="|"):
            p = CML / r["wav_filename"]
            rows.append([f"cml/{p.stem}", str(p), "bonafide"])
    return rows


def asv5_eval(conditions=None, per_condition=None):
    """ASVspoof 5 eval track 1, emitted for A2's codec replicate and M1's
    clustered-inference arm off one manifest.

    Columns of ASVspoof5.eval.track_1.tsv, verified against the file:
      0 speaker  1 utt_id  2 gender  3 codec condition  4 -  5 source utt
      6 attack-cond  7 attack  8 label  9 -

    Speaker and condition are carried in the utt_id column of the manifest as a
    suffix-free id, so downstream code joins back to the TSV rather than
    re-parsing paths. `conditions` restricts to a subset (e.g. {"-"} for the
    no-codec viability pre-check); `per_condition` caps per condition for cheap
    probes, seeded.
    """
    proto = DATA / "asvspoof5/ASVspoof5.eval.track_1.tsv"
    base = DATA / "asvspoof5/flac_E_eval"
    by_cond = {}
    for line in proto.read_text().splitlines():
        f = line.split()
        if len(f) < 9:
            continue
        cond, utt, label = f[3], f[1], f[8]
        if conditions and cond not in conditions:
            continue
        by_cond.setdefault(cond, []).append([utt, str(base / f"{utt}.flac"), label])
    rows = []
    for cond in sorted(by_cond):
        r = by_cond[cond]
        if per_condition and len(r) > per_condition:
            r = random.Random(SEED).sample(r, per_condition)
        rows.extend(sorted(r, key=lambda x: x[0]))
    return rows


if __name__ == "__main__":
    write("asv19_dev", asv19("dev"))
    write("asv19_eval", asv19("eval"))
    write("asv21la", asv21("LA"))
    write("asv21df_100k", asv21("DF", subsample=100_000))
    write("itw", itw())
    write("brspeech_test", brspeech())
    # A5 eval track 1: 12 conditions x 737 speakers, recording-disjoint.
    # Shared by A2 (EXP-103 codec replicate) and M1 (clustered inference).
    write("asv5_eval", asv5_eval())
