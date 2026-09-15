"""Score every retained epoch checkpoint on ASVspoof 2019 LA dev, and read the
pre-registered saturation bar off the result.

The bar (frozen in PREREG.md before the run): rank checkpoints by dev EER, take
the best, bootstrap the PAIRED difference (contender - best) over the same
resampled dev trials. If every 2nd-5th ranked checkpoint's difference CI
contains zero, the selection criterion is tied among its top candidates and the choice among them is made by the training script's
tie-break rather than by evidence. If any falls outside, the criterion
discriminates.

Dependence unit is the trial for the CI and the checkpoint for the spread. There
is one training run here, so nothing across runs is claimed -- the seed question
needs stage 1.

Resumable: per-checkpoint scores are cached as .npy and skipped if present.
"""

import argparse
import json
import sys
import os
from pathlib import Path

import numpy as np
import torch

# The training module lives in the recipe directory the job runs from (EXP110_CODE_DIR
# or the current directory), not beside this script.
sys.path.insert(0, os.environ.get("EXP110_CODE_DIR", os.getcwd()))
sys.path.insert(0, str(Path(__file__).parent))
from train import install_xlsr_compat

# EXP-110: the run directory is a parameter. The upstream constant pointed at
# EXP-401, whose dev_scores/ cache is keyed by epoch number only, so a second
# training arm would have silently reused EXP-401's cached scores.
RUN = Path(os.environ["EXP110_RUN_DIR"])
DB = Path.home() / "data/corpora/anti-spoofing/ASVspoof2019/LA"
DEV_PROTO = DB / "ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt"
N_BOOT = 1000
SEED = 401
TOP_K = 5


class Args:
    """The authors' defaults, as in smoke.py. algo=0 disables RawBoost: dev
    scoring must not augment, or the 'dev EER' is not the selection criterion
    the training script computed."""
    algo = 0
    nBands, minF, maxF, minBW, maxBW = 5, 20, 8000, 100, 1000
    minCoeff, maxCoeff, minG, maxG = 10, 100, 0, 0
    minBiasLinNonLin, maxBiasLinNonLin, N_f = 5, 20, 5
    P, g_sd, SNRmin, SNRmax = 10, 2, 10, 40


def eer(bona, spoof):
    """EER from the pooled score distribution. `bona` scores high by contract."""
    y = np.r_[np.ones(len(bona)), np.zeros(len(spoof))]
    s = np.r_[bona, spoof]
    o = np.argsort(-s)
    y = y[o]
    fnr = 1.0 - np.cumsum(y) / max(len(bona), 1)
    fpr = np.cumsum(1 - y) / max(len(spoof), 1)
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fnr[i] + fpr[i]) / 2)


def score_checkpoint(ckpt, loader, model, device):
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    out = []
    with torch.no_grad():
        for x, _ in loader:
            out.append(model(x.to(device))[:, 1].cpu().numpy())
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models_dir", required=True,
                    help="directory holding epoch_*.pth")
    args_cli = ap.parse_args()

    install_xlsr_compat()
    from data_utils_SSL import Dataset_ASVspoof2019_train, genSpoof_list
    from model import Model
    from torch.utils.data import DataLoader

    device = "cuda"
    labels, files = genSpoof_list(dir_meta=str(DEV_PROTO), is_train=False, is_eval=False)
    ds = Dataset_ASVspoof2019_train(Args(), list_IDs=files, labels=labels,
                                    base_dir=str(DB / "ASVspoof2019_LA_dev") + "/",
                                    algo=Args.algo)
    loader = DataLoader(ds, batch_size=32, num_workers=8, shuffle=False)
    is_bona = np.array([labels[f] for f in files], dtype=bool)
    print(f"dev: {len(files)} trials, {is_bona.sum()} bona fide", flush=True)

    model = Model(Args(), device).to(device)
    cache = RUN / "dev_scores"
    cache.mkdir(parents=True, exist_ok=True)

    rows = {}
    for ckpt in sorted(Path(args_cli.models_dir).glob("epoch_*.pth"),
                       key=lambda p: int(p.stem.split("_")[1])):
        ep = int(ckpt.stem.split("_")[1])
        npy = cache / f"epoch_{ep}.npy"
        if npy.exists():
            s = np.load(npy)
        else:
            s = score_checkpoint(ckpt, loader, model, device)
            np.save(npy, s)
            print(f"  scored epoch {ep}", flush=True)
        rows[ep] = eer(s[is_bona], s[~is_bona]) * 100

    ranked = sorted(rows.items(), key=lambda kv: kv[1])
    best_ep, best_eer = ranked[0]
    print(f"\nbest checkpoint: epoch {best_ep} at dev EER {best_eer:.4f}%")

    # PAIRED bootstrap of the difference, per the blind amendment in PREREG.md.
    # Checkpoints are scored on identical trials, so their EERs are strongly
    # correlated; comparing a contender's point estimate against the best
    # checkpoint's MARGINAL CI is ~1.7x too permissive and biased toward
    # declaring "tied", which is the reading that favours us. Resample trial
    # indices once per draw and apply them to both checkpoints.
    best = np.load(cache / f"epoch_{best_ep}.npy")
    bi = np.flatnonzero(is_bona)
    si = np.flatnonzero(~is_bona)
    rng = np.random.default_rng(SEED)
    draws = [(rng.integers(0, len(bi), len(bi)), rng.integers(0, len(si), len(si)))
             for _ in range(N_BOOT)]

    contenders = ranked[1:TOP_K]
    inside, outside, cis = [], [], {}
    for ep, val in contenders:
        other = np.load(cache / f"epoch_{ep}.npy")
        d = [ (eer(other[bi[jb]], other[si[js]]) - eer(best[bi[jb]], best[si[js]])) * 100
              for jb, js in draws ]
        dlo, dhi = np.percentile(d, [2.5, 97.5])
        cis[ep] = [float(dlo), float(dhi)]
        (inside if dlo <= 0 <= dhi else outside).append((ep, val))
        print(f"  epoch {ep}: dev EER {val:.4f}%, paired diff vs best "
              f"95% CI [{dlo:+.4f}, {dhi:+.4f}] -> {'tied' if dlo <= 0 <= dhi else 'DIFFERENT'}")

    print(f"\nranks 2-{TOP_K}: {[(e, round(v, 4)) for e, v in contenders]}")
    verdict = "SATURATED" if not outside else "KILL"
    print(f"\nPRE-REGISTERED VERDICT: {verdict}")
    if verdict == "KILL":
        print("  the dev criterion discriminates among its top candidates;")
        print("  II-A's mechanism is absent (PREREG.md, primary reading).")
    else:
        print("  top candidates are statistically tied: the choice among them is")
        print("  made by the training script's tie-break, not by evidence.")

    out = RUN / "results_stage0.json"
    out.write_text(json.dumps({
        "dev_eer_by_epoch": rows, "best_epoch": best_ep, "best_dev_eer": best_eer,
        "paired_diff_ci95_by_epoch": cis, "n_dev_trials": int(len(best)),
        "n_boot": N_BOOT, "top_k": TOP_K,
        "contenders": {str(e): v for e, v in contenders},
        "verdict": verdict,
    }, indent=1))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
