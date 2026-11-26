"""
Runtime scaling of TAPRA (answers Reviewer 6 comment 6 and Reviewer 5 comment 6).

Run ALONE on an otherwise idle machine: timings are wall-clock, one process.
Varies the number of nodes with the city fixed, then the building density with the
number of nodes fixed, and records the time per decision interval.

  python runtime_scaling.py --out results/runtime.csv
"""
import argparse
import csv
import os
import time

import numpy as np

from link_model import LinkModel
from optimizer import Config, run_trajectory
from scenario_gen import generate

METHODS = {"TAPRA": ("adaptive", {}),
           "TAPRA uncertainty-aware (2 samples)": ("adaptive", {"pos_sigma": 5.0, "pos_samples": 2}),
           "TAPRA uncertainty-aware (3 samples)": ("adaptive", {"pos_sigma": 5.0, "pos_samples": 3}),
           "Full search every slot": ("full", {})}

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/runtime.csv")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--slots", type=int, default=20)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cases = [("nodes", "urban", 500, n) for n in (10, 20, 30, 50, 80)] + \
            [("density", env, 500, 20) for env in ("suburban", "urban", "dense_urban")] + \
            [("area", "urban", 1000, 50)]
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sweep", "env", "area", "nodes", "buildings", "seed", "method",
                    "first_slot_s", "per_slot_s", "evals_per_slot"])
        for sweep, env, area, n in cases:
            lm = LinkModel(env=env)
            for seed in range(1, a.seeds + 1):
                sc = generate(env, area, n, "homogeneous", a.slots, 5.0, seed=seed)
                for name, (m, over) in METHODS.items():
                    res = run_trajectory(sc, lm, Config(**over), m)
                    row = [sweep, env, area, n, len(sc["Buildings"]), seed, name,
                           round(res[0].seconds, 4),
                           round(float(np.mean([r.seconds for r in res[1:]])), 4),
                           round(float(np.mean([r.evals for r in res[1:]])), 1)]
                    w.writerow(row)
                    f.flush()
                    print(*row)
