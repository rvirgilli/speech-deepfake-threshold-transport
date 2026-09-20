"""Generate paper figures for M1 and A2 from campaign result JSONs.

IEEE two-column: column width ~3.45 in. Okabe-Ito subset palette (validated
CVD-safe); direct labels on every series; single-hue/size encodings only.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).parent
EXP = HERE.parent / "experiments"
BLUE, VERM, GREEN, ORANGE, GRAY = "#0072B2", "#D55E00", "#009E73", "#E69F00", "#6e6e6e"

plt.rcParams.update({
    "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
    "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e6e6e6", "grid.linewidth": 0.5,
    "pdf.fonttype": 42,
})
COL = 86 / 25.4  # ICASSP column width, 86 mm, so the included figure is never scaled


def m1_power():
    r = json.load(open(EXP / "EXP-101-m1-campaign/results_power.json"))
    A = [5, 10, 20, 40, 80, 110]
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.62), sharey=True)
    for ax, tag, title in ((axes[0], "high_icc_organizer", "organizer-era ICC (EER 18–25%)"),
                           (axes[1], "low_eer_sota", "SSL-era ICC (EER <2%)")):
        pw = r[tag]["power"]
        stagger = ({"delta_1.0": 5, "delta_0.5": -1, "delta_0.2": -7}
                   if tag == "high_icc_organizer" else
                   {"delta_1.0": 4, "delta_0.5": 0, "delta_0.2": -4})
        for dk, color, lab in (("delta_1.0", BLUE, "Δ=1.0"), ("delta_0.5", VERM, "Δ=0.5"),
                               ("delta_0.2", GREEN, "Δ=0.2")):
            y = [pw[dk][f"S93_A{a}"] for a in A]
            ax.plot(A, y, color=color, lw=1.6, marker="o", ms=2.5)
            ax.annotate(lab, (A[-1], y[-1]), textcoords="offset points",
                        xytext=(3, stagger[dk]), color=color, fontsize=6.5, va="center")
        y0 = [pw["delta_0.0"][f"S93_A{a}"] for a in A]
        ax.plot(A, y0, color=GRAY, lw=1.2, ls="--")
        ax.annotate("size (Δ=0)", (A[1], y0[1]), textcoords="offset points",
                    xytext=(0, 7), color=GRAY, fontsize=6, ha="center")
        ax.axhline(0.8, color=GRAY, lw=0.7, ls=":", zorder=0)
        ax.axhline(0.05, color=GRAY, lw=0.7, ls=":", zorder=0)
        ax.set_title(title)
        ax.set_xlabel("attack clusters $A$ (S=93)")
        ax.set_xscale("log")
        ax.set_xticks([5, 10, 20, 40, 110])
        ax.set_xticklabels([5, 10, 20, 40, 110])
        ax.set_xlim(4.5, 260)
        ax.minorticks_off()
    axes[0].set_ylabel("P(95% CI excludes 0)")
    axes[0].set_ylim(0, 1.02)
    fig.tight_layout(pad=0.4)
    fig.savefig(HERE / "M1/figs/power.pdf")
    plt.close(fig)


def m1_graph():
    r = json.load(open(EXP / "EXP-101-m1-campaign/results_pairs.json"))
    fig, axes = plt.subplots(2, 1, figsize=(COL, 1.9))
    for ax, ds, label in ((axes[0], "21df", "ASVspoof 2021 DF"), (axes[1], "itw", "In-the-Wild")):
        eers = r[ds]["pooled_eer"]
        names = sorted(eers, key=eers.get)
        x = np.arange(len(names), dtype=float)  # rank spacing: EER packs the organizer cluster unreadably
        y = np.zeros(len(names))
        for (a, b) in zip(names, names[1:]):
            v = r[ds]["pairs"][f"{a} vs {b}"]
            i, j = names.index(a), names.index(b)
            resolved = v["generalization_resolved_consensus"]
            ax.plot([x[i], x[j]], [0, 0],
                    color=BLUE if resolved else VERM,
                    lw=1.8 if resolved else 1.2,
                    ls="-" if resolved else (0, (3, 2)), zorder=1)
        ax.scatter(x, y, s=26, color="#222", zorder=2)
        for i, m in enumerate(names):
            short = m.replace("XLSR-", "").replace("XLS-R+", "").replace("SSL-AASIST", "SSL-AAS.")
            ax.annotate(f"{short}\n{eers[m]:.2f}", (x[i], 0), textcoords="offset points",
                        xytext=(0, 7 if i % 2 == 0 else -20), ha="center", fontsize=5.8)
        ax.set_xlim(-0.5, len(names) - 0.5)
        ax.set_ylim(-1.1, 1.1)
        ax.set_yticks([])
        ax.set_xticks([])
        ax.grid(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_visible(False)
        ax.set_title(label, loc="left", fontsize=7)
    fig.tight_layout(pad=0.4, h_pad=1.4)
    fig.savefig(HERE / "M1/figs/graph.pdf")
    plt.close(fig)


def a2_drift():
    """Joint view: transported-FPR displacement against spoof-side cost.

    The x axis is the signed floored log2(FPR/alpha) of the transported threshold; the y
    axis is the mean FNR at that threshold minus the deployment-oracle FNR, in percentage
    points. 21LA within-corpus and cross-corpus cells for both reproduction-scored
    detectors, plus the ASVspoof 5 twin-free pairs of SSL-AASIST, filled where the
    destination is usable and hollow where it is overlap-dominated.

    ICASSP requires at least 9 pt everywhere, figure text included, so the figure is drawn
    at 9 pt and at the exact column width, which keeps LaTeX's inclusion scale at 1. The
    marker classes are named in the caption rather than in a legend, which at 9 pt would
    cover the data.
    """
    r = json.load(open(EXP / "EXP-102-a2-campaign/results_drift.json"))
    a5 = json.load(open(EXP / "EXP-103-a5-replicate/artifacts/results_a5.json"))
    rc = {"font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 9, "ytick.labelsize": 9}
    with plt.rc_context(rc):
        fig, ax = plt.subplots(1, 1, figsize=(COL, 1.62))
        for model, color, marker in (("ssl", BLUE, "o"), ("aasist", VERM, "s")):
            pts = [(v["log2_fpr_ratio"], 100 * v["fnr_price"])
                   for fam in ("within", "cross") for k, v in r[fam].items()
                   if k.startswith(model + "/")]
            x, y = map(np.array, zip(*pts))
            ax.scatter(x, y, s=9, color=color, marker=marker, alpha=0.8,
                       edgecolors="white", linewidths=0.3)
        cells = a5["ssl/twin_free"]
        dest = {}
        for k, v in cells.items():
            dest.setdefault(k.split("->")[1], []).append(v["fnr_oracle"])
        usable = {d for d, vals in dest.items() if max(vals) <= 0.5}
        for filled in (True, False):
            pts = [(v["log2_fpr_ratio"], 100 * v["fnr_price"]) for k, v in cells.items()
                   if (k.split("->")[1] in usable) == filled]
            x, y = map(np.array, zip(*pts))
            ax.scatter(x, y, s=11, marker="^", facecolors=GREEN if filled else "none",
                       edgecolors=GREEN, linewidths=0.5, alpha=0.9)
        ax.axvline(0, color=GRAY, lw=0.8)
        for xv in (-1, 1):
            ax.axvline(xv, color=GRAY, lw=0.8, ls=":")
        ax.axhline(0, color=GRAY, lw=0.6)
        ax.set_xlabel("$\\log_2$(transported FPR / 5%), floored at $3/n$")
        ax.set_ylabel("spoof-side cost (pp)")
        fig.tight_layout(pad=0.4)
        fig.savefig(HERE / "A2/figs/drift.pdf")
    plt.close(fig)

def a2_nsweep():
    r = json.load(open(EXP / "EXP-102-a2-campaign/results_nsweep.json"))
    rp = json.load(open(EXP / "EXP-102-a2-campaign/results_parametric.json"))
    key = "ssl/itw"
    fig, axes = plt.subplots(1, 2, figsize=(COL, 1.7),
                             gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    Ns = [30, 100, 300, 1000, 3000, 10000]
    q_m, q_lo, q_hi, z_m, b_lo, b_hi = [], [], [], [], [], []
    for N in Ns:
        c = r["n_sweep"][key][str(N)]
        q_m.append(100 * c["quantile"]["fpr_mean"])
        q_lo.append(100 * c["quantile"]["fpr_ci"][0])
        q_hi.append(100 * c["quantile"]["fpr_ci"][1])
        z_m.append(100 * c["znorm"]["fpr_mean"])
        b_lo.append(100 * c["beta_law"]["band95"][0])
        b_hi.append(100 * c["beta_law"]["band95"][1])
    p_m = [100 * rp["ssl"]["itw"][str(N)]["parametric"]["fpr_mean"] for N in Ns]
    ax.fill_between(Ns, b_lo, b_hi, color=BLUE, alpha=0.14, lw=0)
    ax.plot(Ns, q_m, color=BLUE, lw=1.6, marker="o", ms=2.5)
    ax.plot(Ns, z_m, color=VERM, lw=1.6, marker="s", ms=2.5)
    ax.plot(Ns, p_m, color=GREEN, lw=1.6, marker="^", ms=2.5)
    ax.axhline(5, color=GRAY, lw=0.7, ls=":")
    ax.annotate("conformal (Beta band)", (Ns[-1], q_m[-1]), xytext=(0, -9),
                textcoords="offset points", color=BLUE, fontsize=6, ha="right")
    ax.annotate("cohort z-norm", (Ns[-1], z_m[-1]), xytext=(0, 4),
                textcoords="offset points", color=VERM, fontsize=6, ha="right")
    ax.annotate("Gaussian quantile", (Ns[0], p_m[0]), xytext=(2, 5),
                textcoords="offset points", color=GREEN, fontsize=6)
    ax.set_xscale("log")
    ax.set_xlabel("labeled bona-fide samples $N$")
    ax.set_ylabel("realized FPR (%)")
    ax.set_title("Cost of labels (ITW)")
    ax = axes[1]
    rates = [0, 1, 2, 5]
    for corpus, color, lab in (("ssl/itw", BLUE, "ITW"), ("ssl/brspeech_test", VERM, "BRSp.")):
        base = 100 * r["n_sweep"][corpus.split("/")[0] + "/" + corpus.split("/")[1]].get("500", r["n_sweep"][corpus]["1000"])["quantile"]["fpr_mean"] if False else 5.0
        y = [5.0] + [100 * r["contamination"][corpus][f"N500_c0.0{k}" if k < 5 else "N500_c0.05"]["fpr_mean"]
                     for k in (1, 2, 5)]
        ax.plot(rates, y, color=color, lw=1.6, marker="o", ms=2.5)
        ax.annotate(lab, (rates[-1], y[-1]), xytext=(2, 0), textcoords="offset points",
                    color=color, fontsize=6.5, va="center")
    ax.axhline(5, color=GRAY, lw=0.7, ls=":")
    ax.set_ylim(0, 6.5)
    ax.set_xticks(rates)
    ax.set_xlabel("contamination (%)")
    ax.set_title("Contamination", fontsize=7)
    fig.tight_layout(pad=0.4)
    fig.savefig(HERE / "A2/figs/nsweep.pdf")
    plt.close(fig)


if __name__ == "__main__":
    (HERE / "M1/figs").mkdir(exist_ok=True)
    (HERE / "A2/figs").mkdir(exist_ok=True)
    m1_power()
    m1_graph()
    a2_drift()
    a2_nsweep()  # not used by the paper since 2026-08-15; output is gitignored
    print("figures written")
