"""
Batch evaluation (Python model) for the TAPRA paper.

For every scenario (environment x size x traffic x seed) and every method it records
the planned served share, the realised share with the geometry updated every second
(nodes on their tracks, UAV on its flight path), node outage, fairness and runtime.

  python batch.py --seeds 1-10 --workers 6 --out results/batch.csv
Re-running appends only the missing rows, so it can be stopped and restarted.
RLTOPA rows need a trained agent rltopa_<env>_<area>m_<n>n_<traffic>.pt (rltopa.py);
without it those rows are skipped.
"""
import argparse
import csv
import os
import time
from multiprocessing import Pool

import numpy as np

from link_model import LinkModel
from optimizer import Config, run_trajectory, realised_inslot
from scenario_gen import generate
from tapra_export import rooftop_baseline, uav_track

SCENARIOS = [  # env, area, nodes, traffic
    ("urban", 500, 20, "homogeneous"), ("urban", 500, 20, "heterogeneous"),
    ("urban", 1000, 50, "homogeneous"), ("urban", 1000, 50, "heterogeneous"),
    ("suburban", 500, 20, "homogeneous"), ("suburban", 500, 20, "heterogeneous"),
]
METHODS = {  # name: (method, Config overrides)
    "TAPRA": ("adaptive", {}),
    "TAPRA uncertainty-aware (5 m)": ("adaptive", {"pos_sigma": 5.0, "pos_samples": 3}),
    "Full search every slot": ("full", {}),
    "VTOPA rule": ("full", {"objective": "vtopa"}),
    "RLTOPA": ("policy", {}),
    "Traffic centroid": ("centroid", {}),
    "Fixed rooftop AP": ("fixed", {}),
    "Ablation: slot-start objective": ("adaptive", {"slot_average": False}),
    "Ablation: no remembered spots": ("adaptive", {"use_cache": False}),
    "Ablation: no trigger": ("adaptive", {"use_trigger": False}),
}
FIELDS = ["env", "area", "nodes", "traffic", "seed", "method", "planned_share",
          "realised_share", "node_outage", "jain", "sec_per_slot", "evals_per_slot",
          "full_searches"]
SLOTS = 50


def run_one(job):
    env, area, n, traffic, seed, name = job
    method, over = METHODS[name]
    cfg = Config(**over)
    lm = LinkModel(env=env)
    sc = generate(env, area, n, traffic, SLOTS, cfg.tau, seed=seed)
    policy = None
    if method == "policy":
        path = f"rltopa_{env}_{area}m_{n}n_{traffic}.pt"
        if not os.path.exists(path):
            return None
        from rltopa import load_policy
        policy = load_policy(path)
    res = run_trajectory(sc, lm, cfg, method, fixed_q=rooftop_baseline(sc["Buildings"]),
                         policy=policy)
    m = realised_inslot(sc, res, lm, cfg, uav_track, details=True)
    return {"env": env, "area": area, "nodes": n, "traffic": traffic, "seed": seed,
            "method": name,
            "planned_share": np.mean([r.served_mbps / r.demand_mbps for r in res]),
            "realised_share": m["realised_share"], "node_outage": m["node_outage"],
            "jain": m["jain"],
            "sec_per_slot": np.mean([r.seconds for r in res[1:]]),
            "evals_per_slot": np.mean([r.evals for r in res[1:]]),
            "full_searches": sum(r.mode in ("full", "adaptive+full") for r in res[1:])}


def parse_seeds(s):
    a, b = (s.split("-") + [s])[:2]
    return list(range(int(a), int(b) + 1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1-10")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="results/batch.csv")
    ap.add_argument("--only", default="", help="comma-separated method names to run")
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            for r in csv.DictReader(f):
                done.add((r["env"], int(r["area"]), int(r["nodes"]), r["traffic"],
                          int(r["seed"]), r["method"]))
    names = [m for m in METHODS if not a.only or m in a.only.split(",")]
    jobs = [(e, ar, n, t, s, m) for (e, ar, n, t) in SCENARIOS for s in parse_seeds(a.seeds)
            for m in names if (e, ar, n, t, s, m) not in done]
    print(f"{len(jobs)} runs to do ({len(done)} already in {a.out})")
    new = not os.path.exists(a.out)
    t0 = time.time()
    with open(a.out, "a", newline="") as f, Pool(a.workers) as pool:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for i, row in enumerate(pool.imap_unordered(run_one, jobs), 1):
            if row:
                w.writerow(row)
                f.flush()
            if i % 10 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} done, {(time.time() - t0) / 60:.1f} min")
