"""ASVspoof 2021 trial-phase protocol shared by every A2 analysis.

The released 21LA and 21DF evaluation keys carry three phases: `eval`,
`progress` and `hidden`. The hidden subset is the same material with all
non-speech intervals removed by a voice activity detector (Liu et al., TASLP
2023, Sec. IV-A); the organisers did not use it in challenge results. Its
score distribution differs sharply from untrimmed speech, so the A2 analyses
exclude it. `progress` is untrimmed and is kept: with it, every 21LA condition
holds the same 2,356 bona-fide recordings of the same 67 speakers, the twin
structure the drift map relies on.
"""

import os
from pathlib import Path

DATA = Path(os.environ.get("A2_DATA", Path.home() / "data/corpora/anti-spoofing"))
KEYS = (DATA / "keys/LA/CM/trial_metadata.txt", DATA / "keys/DF/CM/trial_metadata.txt")
EXCLUDED_PHASES = ("hidden",)
PHASE_COL = 7


def hidden_utterances():
    out = set()
    for key in KEYS:
        for line in key.read_text().splitlines():
            p = line.split()
            if p[PHASE_COL] in EXCLUDED_PHASES:
                out.add(p[1])
    return out


HIDDEN = hidden_utterances()


def drop_hidden(rows):
    """Remove hidden-phase trials from EXP-001 score rows (utt_id, score, label)."""
    return [r for r in rows if r["utt_id"] not in HIDDEN]
