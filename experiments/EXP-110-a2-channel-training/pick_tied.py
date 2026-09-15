"""Emit the five lowest-dev-EER epochs, the same top-k rule EXP-401 applied."""

import argparse
import json

ap = argparse.ArgumentParser()
ap.add_argument("--results", required=True)
ap.add_argument("--k", type=int, default=5)
a = ap.parse_args()

eer = json.load(open(a.results))["dev_eer_by_epoch"]
ranked = sorted(eer.items(), key=lambda kv: (kv[1], int(kv[0])))[:a.k]
print(" ".join(e for e, _ in ranked))
