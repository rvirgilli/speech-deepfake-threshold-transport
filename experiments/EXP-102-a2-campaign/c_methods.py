"""EXP-102 cell 2: unlabeled corrections from arXiv 2606.21584 at our operating point.

Resource class: unlabeled target data only. Each correction is fitted on
unlabeled TARGET statistics, the operating threshold comes from labeled SOURCE
(19LA dev) transformed by the same correction fitted on dev's own unlabeled
statistics — i.e. what a deployment without target labels can actually do.
  C1 z-norm:      s' = (s - mu_mix)/sd_mix (target mixture stats).
  C2 temp/shift:  quartile pseudo-labels (top 25% bona, bottom 25% spoof),
                  1-D logistic fit -> s' = (s + b)/T.
  C5 AS-norm:     per-utterance top-k cosine cohort (k=100) in embedding space
                  over an unlabeled target cohort subsample; s'_i = (s_i - mu_i)/sd_i.
C4 (CORAL) requires re-scoring through the frozen classifier head and is
deferred (REPORT deviation); anchor reports it non-monotone but still failing.
"""

import csv
import os
import gzip
import json
from pathlib import Path

import numpy as np

SCORES = Path(os.environ.get("A2_SCORES", Path.home() / "projects/academic/icassp2027/experiments/EXP-001-scoring-campaign/scores"))
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "EXP-001-scoring-campaign/code"))
from protocol import drop_hidden  # noqa: E402

EMB = Path(os.environ.get("A2_EMB", Path.home() / "exp-artifacts/icassp2027/EXP-001/emb"))
ALPHA = 0.05
K_COHORT = 100
COHORT_SUB = 5000
SEED = 20260813
TARGETS = ["asv21la", "asv21df_full", "itw", "brspeech_test"]


def load(model, corpus):
    with gzip.open(SCORES / f"{model}_{corpus}.csv.gz", "rt") as f:
        rows = drop_hidden(list(csv.DictReader(f)))
    utts = [r["utt_id"] for r in rows]
    s = np.array([float(r["score"]) for r in rows])
    bona = np.array([r["label"] == "bonafide" for r in rows])
    return utts, s, bona


def load_emb(model, corpus, utts):
    e = np.load(EMB / f"{model}_{corpus}_emb.npy")
    eu = np.load(EMB / f"{model}_{corpus}_utts.npy", allow_pickle=True)
    idx = {u: i for i, u in enumerate(eu)}
    rows = [idx[u] for u in utts]
    e = e[rows].astype(np.float32)
    return e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-8)


def fit_temp_shift(s):
    """Quartile pseudo-labels, 1-D logistic fit by damped Newton; returns (T, b).

    Each Newton step is backtracked (halved, up to 30 times) until the mean
    logistic loss decreases, so the iteration cannot diverge on separable
    pseudo-labels; the endpoint after 50 accepted steps defines the transform.
    """
    q1, q3 = np.quantile(s, 0.25), np.quantile(s, 0.75)
    x = np.concatenate([s[s >= q3], s[s <= q1]])
    y = np.concatenate([np.ones((s >= q3).sum()), np.zeros((s <= q1).sum())])

    def loss(w, b):
        z = w * x + b
        return float(np.mean(np.logaddexp(0.0, z) - y * z))

    w, b = 1.0, 0.0
    current = loss(w, b)
    for _ in range(50):
        z = np.clip(w * x + b, -30, 30)
        p = 1 / (1 + np.exp(-z))
        g = np.array([np.sum((p - y) * x), np.sum(p - y)])
        r = p * (1 - p)
        H = np.array([[np.sum(r * x * x), np.sum(r * x)],
                      [np.sum(r * x), np.sum(r)]]) + 1e-6 * np.eye(2)
        step = np.linalg.solve(H, g)
        scale = 1.0
        for _ in range(30):
            candidate = loss(w - scale * step[0], b - scale * step[1])
            if candidate < current:
                w, b, current = w - scale * step[0], b - scale * step[1], candidate
                break
            scale *= 0.5
    T = 1.0 / max(w, 1e-6)
    return T, b * T  # s' = (s + b*T)/T == w*s + b


def _self_check():
    """The damped fit must lower the loss on a separable pseudo-label set."""
    rng = np.random.default_rng(0)
    s = np.concatenate([rng.normal(-6.2, 0.05, 2000), rng.normal(-5.9, 0.05, 2000)])
    T, b = fit_temp_shift(s)
    q1, q3 = np.quantile(s, 0.25), np.quantile(s, 0.75)
    x = np.concatenate([s[s >= q3], s[s <= q1]])
    y = np.concatenate([np.ones((s >= q3).sum()), np.zeros((s <= q1).sum())])
    z = (x + b) / T
    assert np.mean(np.logaddexp(0.0, z) - y * z) < 0.1


def asnorm(rng, s, emb, cohort_emb, cohort_s):
    """AS-norm with unlabeled cohort: per-utt top-K cosine stats of cohort scores."""
    out = np.empty(len(s))
    for i in range(0, len(s), 2000):
        sim = emb[i:i + 2000] @ cohort_emb.T
        top = np.argpartition(-sim, K_COHORT, axis=1)[:, :K_COHORT]
        cs = cohort_s[top]
        out[i:i + 2000] = (s[i:i + 2000] - cs.mean(1)) / np.maximum(cs.std(1), 1e-6)
    return out


def rates(t, bona_s, spoof_s):
    return round(float(np.mean(bona_s < t)), 4), round(float(np.mean(spoof_s >= t)), 4)


def main():
    rng = np.random.default_rng(SEED)
    results = {}
    for model in ["ssl", "aasist"]:
        d_utts, d_s, d_bona = load("ssl" if model == "ssl" else "aasist", "asv19_dev")
        # C5 is the only correction that needs embeddings; C1 and C2 run from the
        # released score tables alone when the embedding arrays are absent.
        with_c5 = (EMB / f"{model}_asv19_dev_emb.npy").exists()

        # Per-correction dev-side threshold (5% FPR on dev bona, corrected space).
        t_c1 = float(np.quantile((d_s[d_bona] - d_s.mean()) / d_s.std(), ALPHA))
        T_d, b_d = fit_temp_shift(d_s)
        t_c2 = float(np.quantile((d_s[d_bona] + b_d) / T_d, ALPHA))
        if with_c5:
            d_emb = load_emb(model, "asv19_dev", d_utts)
            sub = rng.choice(len(d_s), min(COHORT_SUB, len(d_s)), replace=False)
            d_asn = asnorm(rng, d_s, d_emb, d_emb[sub], d_s[sub])
            t_c5 = float(np.quantile(d_asn[d_bona], ALPHA))
        else:
            print(f"{model}: embeddings not found under {EMB}; C5 skipped", flush=True)

        results[model] = {}
        for corpus in TARGETS:
            utts, s, bona = load(model, corpus)
            cell = {}
            s1 = (s - s.mean()) / s.std()
            cell["C1_znorm"] = dict(zip(("fpr", "fnr"), rates(t_c1, s1[bona], s1[~bona])))
            T_t, b_t = fit_temp_shift(s)
            s2 = (s + b_t) / T_t
            cell["C2_tempshift"] = dict(zip(("fpr", "fnr"), rates(t_c2, s2[bona], s2[~bona])))
            if with_c5:
                emb = load_emb(model, corpus, utts)
                sub = rng.choice(len(s), min(COHORT_SUB, len(s)), replace=False)
                s5 = asnorm(rng, s, emb, emb[sub], s[sub])
                cell["C5_asnorm"] = dict(zip(("fpr", "fnr"), rates(t_c5, s5[bona], s5[~bona])))
            results[model][corpus] = cell
            print(f"{model}/{corpus}: " + " ".join(
                f"{k}: fpr={v['fpr']} fnr={v['fnr']}" for k, v in cell.items()), flush=True)

    out = Path(__file__).parent / "results_cmethods.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote {out}")


if __name__ == "__main__":
    _self_check()
    main()
