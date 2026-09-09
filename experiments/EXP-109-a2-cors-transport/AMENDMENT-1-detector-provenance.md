# AMENDMENT 1 — the three added detectors are released scores, not checkpoints

Date: **2026-08-21**, written before any new score was computed. Append-only; it does not
edit the frozen [PREREG.md](PREREG.md) (`a7a8dbec…`).

## Finding

The PREREG's cells A and B assumed XLS-R+SLS, XLSR-Mamba and XLSR-Conformer were runnable
checkpoints. They are not. `~/data/corpora/anti-spoofing/official-scores/` holds **released
per-trial score files** under the DATA.md heading *"Official SOTA per-trial scores"*; the only
detectors this project can execute are SSL-AASIST and AASIST, through the EXP-001 harness.

## Verified coverage

| Detector | 21LA file | Trials |
|---|---|---:|
| XLS-R+SLS | `xlsr-sls/scores_LA.txt` | **181,566** |
| XLSR-Mamba | `xlsr-mamba/Bmamba5_LA_WCE_1e-06_ES144_NE12.txt` | **181,566** |
| XLSR-Conformer | `xlsr-conformer-rosello/` — three LA variants | see selection below |

The 21LA key `keys/LA/CM/trial_metadata.txt` also has 181,566 rows, with the condition in
column 3: `none, alaw, ulaw, gsm, g722, opus, pstn`. Utterance IDs match (`LA_E_*`).

## Consequences

**Cell A becomes CPU-only and larger than planned.** A within-21LA transport map over 7
conditions is 42 ordered condition pairs per detector; across 5 detectors that is **210
cells**, against A2's current 108 — computed from files already on disk, at **0 GPU-h**
(the registered estimate was 5.4). Six of the seven conditions are real transmission
(Asterisk PBX VoIP; C3 a real Spanish PSTN carrier), so this is a real-telephony transport
map, not a codec one.

**Cell B shrinks.** CoRS can only be scored with the two runnable detectors:
266,730 × 2 ÷ 64 utts/s = **2.3 GPU-h** (registered 5.8). Standing up the other three stacks
would need weight downloads and three unfamiliar environments; per the EXP-201 anchor
(~35 min lost to one env fight) that is deferred, not scheduled.

**A provenance stratum is now mandatory.** Released scores and our reproductions are not the
same object — M1 measures Arena RawNet2 at 40.67% EER against 22.38% in the organizer release,
score correlation 0.36. The three added detectors are therefore reported as a **separate
provenance stratum**, exactly as M1 does, and no claim pools them with the executed detectors.

## Prospective variant selection, frozen before computation

Where a detector ships several LA score files, the variant is chosen now, on name alone,
without computing any transport cell:

- XLSR-Mamba: `Bmamba5_LA_WCE_1e-06_ES144_NE12.txt` — the 181,566-row LA file.
  `Bmamba3_…` has 611,829 rows (DF-sized) and is not an LA scoring.
- XLSR-Conformer: `Scores_Best_LA_Fixed_size_train.txt` — the fixed-size-train "best LA"
  file, chosen over the `variable_size_train` and `eval_lv` variants because fixed-size
  training matches the other systems' protocol.

Any later change of variant is a deviation and is recorded as one.

## Registered readings are unchanged

The outcome table in the PREREG applies verbatim to the 210-cell map, including the branch
where the transport failure does not reproduce on the added detectors, which is reported as a
limitation of A2's existing claim rather than omitted.
