"""Spoof-side cost distribution on the usable ASVspoof 5 pairs (SSL-AASIST, twin-free arm).

Usable destinations are those whose oracle FNR at 5% FPR is at most 50% for every
source. Writes artifacts/a5_usable_cost.json.
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent


def main():
    a5 = json.load(open(HERE / "artifacts" / "results_a5.json"))
    out = {}
    for model in ("ssl", "aasist"):
        c = a5[f"{model}/twin_free"]
        dest = {}
        for k, v in c.items():
            dest.setdefault(k.split("->")[1], []).append(v["fnr_oracle"])
        usable = sorted(d for d, vals in dest.items() if max(vals) <= 0.5)
        pairs = [k for k in c if k.split("->")[1] in usable]
        price = np.array([c[k]["fnr_price"] for k in pairs]) * 100
        out[model] = {"usable_destinations": usable, "n_pairs": len(pairs),
                      "positive_cost": int((price > 0).sum()) if len(pairs) else 0,
                      "conservative_2x": int(sum(c[k]["log2_fpr_ratio"] < -1 for k in pairs)),
                      "cost_median_pp": float(np.median(price)) if len(pairs) else None,
                      "cost_over_10pp": int((price > 10).sum()) if len(pairs) else 0,
                      "cost_iqr_pp": [float(x) for x in np.percentile(price, [25, 75])] if len(pairs) else None}
        print(model, out[model], flush=True)
    (HERE / "artifacts" / "a5_usable_cost.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
