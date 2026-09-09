"""Summarise the drift map under the corrected metrics, per detector.

Separates the two failure modes the old scalar could not see:
  MISS  — the FPR guarantee is off target by more than a factor of two in
          either direction (|log2(FPR/alpha)| > 1). On the conservative side
          |FPR-alpha| cannot express this: it saturates at alpha.
  PRICE — the spoof-side cost of the miscalibration: deployment FNR at the
          conformal threshold minus deployment FNR at the oracle threshold that
          realizes alpha on deployment bona. Referencing the ORACLE, not the
          calibration condition, is what isolates the cost of the threshold
          being wrong from the difficulty gap between calibration-condition and
          deployment attacks. MISS and PRICE are reported independently (a cell
          can be off target cheaply, or on target while the attacks got harder),
          with the cross-tab printed; "graceful" means neither.

Both bars are POST-HOC and unregistered. The counts below move by roughly a
factor of two across defensible bars, so the direction is the finding and the
counts are not quotable as stable numbers -- see the sensitivity block.
"""

import json
from pathlib import Path

HERE = Path(__file__).parent
SEVERITY_BAR = 1.0     # doublings off target
FNR_PRICE_PP = 0.10    # spoof-side cost that makes a conservative miss operational
OLD_RULE_PP = 0.05     # the released rule: |FPR - alpha| <= 5 points


def main():
    res = json.load(open(HERE / "results_drift.json"))
    print(f"{'section':7s} {'model':7s} {'n':>4s} {'old-pass':>9s} {'graceful':>9s} "
          f"{'miss':>6s} {'  of which cons.':>16s} {'priced':>7s}")
    for section in ("within", "cross"):
        for model in ("ssl", "aasist"):
            cells = {k: v for k, v in res[section].items() if k.startswith(model + "/")}
            n = len(cells)
            old_pass = sum(v["excursion"] <= OLD_RULE_PP for v in cells.values())
            miss = [v for v in cells.values() if abs(v["log2_fpr_ratio"]) > SEVERITY_BAR]
            cons = [v for v in miss if v["log2_fpr_ratio"] < 0]
            priced = [v for v in cells.values() if v["fnr_price"] > FNR_PRICE_PP]
            graceful = [v for v in cells.values()
                        if abs(v["log2_fpr_ratio"]) <= SEVERITY_BAR
                        and v["fnr_price"] <= FNR_PRICE_PP]
            print(f"{section:7s} {model:7s} {n:4d} {old_pass:9d} {len(graceful):9d} "
                  f"{len(miss):6d} {len(cons):16d} {len(priced):7d}")

    allc = [(k, v) for sec in ("within", "cross") for k, v in res[sec].items()]

    print("\nMISS x PRICE cross-tab (all cells):")
    for m in (False, True):
        for p in (False, True):
            n = sum(1 for _, v in allc
                    if (abs(v["log2_fpr_ratio"]) > SEVERITY_BAR) == m
                    and (v["fnr_price"] > FNR_PRICE_PP) == p)
            print(f"  miss={str(m):5s} priced={str(p):5s}: {n:3d}")

    print("\nWorst cells by spoof-side price (oracle-referenced), all sections:")
    for k, v in sorted(allc, key=lambda x: -x[1]["fnr_price"])[:10]:
        old = "graceful" if v["excursion"] <= OLD_RULE_PP else "flagged"
        rl = " [resolution-limited]" if v.get("resolution_limited") else ""
        print(f"  {k:42s} FPR {v['vanilla_fpr_mean']:.4f}  log2 {v['log2_fpr_ratio']:+6.2f}  "
              f"FNR {v['fnr_oracle']:.3f}->{v['vanilla_fnr_mean']:.3f} "
              f"(price {v['fnr_price']:+.3f}, diag {v['fnr_shift_diagnostic']:+.3f})  "
              f"old rule: {old}{rl}")

    print("\nCells the old rule called graceful that miss target by >2x:")
    hidden = [(k, v) for k, v in allc
              if v["excursion"] <= OLD_RULE_PP and abs(v["log2_fpr_ratio"]) > SEVERITY_BAR]
    print(f"  {len(hidden)} of {sum(v['excursion'] <= OLD_RULE_PP for _, v in allc)} "
          f"old-rule passes ({len(allc)} cells total)")

    print("\nSensitivity of the counts to the two post-hoc bars:")
    for sb in (0.585, 1.0, 1.585, 2.0):
        h = sum(1 for _, v in allc
                if v["excursion"] <= OLD_RULE_PP and abs(v["log2_fpr_ratio"]) > sb)
        print(f"  severity>{sb:5.3f} ({2**sb:.1f}x): hidden {h:3d} of "
              f"{sum(v['excursion'] <= OLD_RULE_PP for _, v in allc)} old-rule passes")
    for pb in (0.05, 0.10, 0.20):
        print(f"  price>{pb:.2f}: {sum(1 for _, v in allc if v['fnr_price'] > pb):3d} cells")

    n_rl = sum(1 for _, v in allc if v.get("resolution_limited"))
    print(f"\nResolution-limited cells (FPR below 5/n_dep_bona): {n_rl} — their "
          f"severity rests on a handful of utterances; do not print as point values.")


if __name__ == "__main__":
    main()
