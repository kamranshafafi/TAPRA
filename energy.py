"""
UAV propulsion energy of the planned trajectories (answers Reviewer 3 comment 4 and
Reviewer 5 comment 6). Energy is reported, not optimised.

Rotary-wing power model of Zeng, Xu and Zhang, IEEE TWC 2019 (Table I parameters):
  P(V) = P0 (1 + 3V^2/U^2) + Pi (sqrt(1 + V^4/(4 v0^4)) - V^2/(2 v0^2))^(1/2)
         + 0.5 d0 rho s A V^3
plus the work against gravity when climbing (W * climb rate).
In each interval the UAV flies straight to its new position at v_max and hovers for
the rest of the interval.

  python energy.py --seeds 1-10 --workers 6 --out results/energy.csv
"""
import argparse
import csv
import os
from multiprocessing import Pool

import numpy as np

from link_model import LinkModel
from optimizer import Config, run_trajectory
from scenario_gen import generate
from tapra_export import rooftop_baseline

P0, PI, U_TIP, V0, D0, RHO, S, A, W = 79.86, 88.63, 120.0, 4.03, 0.6, 1.225, 0.05, 0.503, 20.0
METHODS = {"TAPRA": ("adaptive", {}), "Full search every slot": ("full", {}),
           "VTOPA rule": ("full", {"objective": "vtopa"}),
           "Traffic centroid": ("centroid", {}), "Fixed rooftop AP": ("fixed", {})}
SCENARIOS = [("urban", 500, 20, "homogeneous"), ("urban", 500, 20, "heterogeneous"),
             ("suburban", 500, 20, "homogeneous")]


def power(v):
    return (P0 * (1 + 3 * v ** 2 / U_TIP ** 2)
            + PI * np.sqrt(np.sqrt(1 + v ** 4 / (4 * V0 ** 4)) - v ** 2 / (2 * V0 ** 2))
            + 0.5 * D0 * RHO * S * A * v ** 3)


def trajectory_energy(qs, tau, v_max):
    e = power(0.0) * tau                      # first interval: hover
    for q0, q1 in zip(qs[:-1], qs[1:]):
        d = np.linalg.norm(q1 - q0)
        if d < 1e-9:
            e += power(0.0) * tau
            continue
        t_fly = min(d / v_max, tau)
        vh = v_max * np.linalg.norm((q1 - q0)[:2]) / d
        vz = v_max * (q1 - q0)[2] / d
        e += (power(vh) + W * max(vz, 0.0)) * t_fly + power(0.0) * (tau - t_fly)
    return e


def run_one(job):
    env, area, n, traffic, seed, name = job
    m, over = METHODS[name]
    cfg = Config(**over)
    sc = generate(env, area, n, traffic, 50, cfg.tau, seed=seed)
    res = run_trajectory(sc, LinkModel(env=env), cfg, m, fixed_q=rooftop_baseline(sc["Buildings"]))
    qs = np.array([r.q for r in res])
    e = trajectory_energy(qs, cfg.tau, cfg.v_max)
    served_mbit = sum(r.served_mbps for r in res) * cfg.tau
    dist = float(np.linalg.norm(np.diff(qs, axis=0), axis=1).sum())
    return [env, area, n, traffic, seed, name, round(e / 1000, 3), round(dist, 1),
            round(e / (len(res) * cfg.tau), 2), round(served_mbit / (e / 1000), 2)]


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1-10")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="results/energy.csv")
    a = ap.parse_args()
    s0, s1 = (a.seeds.split("-") + [a.seeds])[:2]
    jobs = [(e, ar, n, t, s, m) for (e, ar, n, t) in SCENARIOS
            for s in range(int(s0), int(s1) + 1) for m in METHODS]
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    with open(a.out, "w", newline="") as f, Pool(a.workers) as pool:
        w = csv.writer(f)
        w.writerow(["env", "area", "nodes", "traffic", "seed", "method", "energy_kj",
                    "distance_m", "mean_power_w", "planned_mbit_per_kj"])
        for i, row in enumerate(pool.imap_unordered(run_one, jobs), 1):
            w.writerow(row)
            f.flush()
            if i % 10 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} done")
