"""Rev. 2: run each method once per city and store the trajectory (waypoints, admitted
sets, timings) so later analyses reuse the same waypoints.
  python rev2_cache.py main     -> 6 scenarios x 10 seeds x 5 methods
  python rev2_cache.py small    -> small cases (N = 8, 10, 12) for the best-subset check"""
import os, pickle, sys, time
from multiprocessing import Pool
import numpy as np
from link_model import LinkModel
from optimizer import Config, run_trajectory
from scenario_gen import generate
from tapra_export import rooftop_baseline
import batch

METHODS = {k: batch.METHODS[k] for k in ["TAPRA", "Full search every slot", "VTOPA rule",
                                          "Traffic centroid", "Fixed rooftop AP"]}

def path(env, area, n, traffic, seed, name):
    return f"results/traj/{env}_{area}_{n}_{traffic}_s{seed}_{name.replace(' ', '_')}.pkl"

def run(job):
    env, area, n, traffic, seed, name = job
    p = path(*job)
    if os.path.exists(p):
        return p
    method, over = METHODS[name]
    cfg = Config(**over)
    sc = generate(env, area, n, traffic, 50, cfg.tau, seed=seed)
    res = run_trajectory(sc, LinkModel(env=env), cfg, method,
                         fixed_q=rooftop_baseline(sc["Buildings"]))
    out = [dict(k=r.k, q=r.q, admitted=r.admitted, served=r.served_mbps, demand=r.demand_mbps,
                seconds=r.seconds, evals=r.evals, mode=r.mode) for r in res]
    pickle.dump(out, open(p + ".tmp", "wb")); os.replace(p + ".tmp", p)
    return p

if __name__ == "__main__":
    which = sys.argv[1]
    if which in ("main", "tapra", "others"):
        names = {"main": list(METHODS), "tapra": ["TAPRA"],
                 "others": [m for m in METHODS if m != "TAPRA"]}[which]
        jobs = [(e, a, n, t, s, m) for (e, a, n, t) in batch.SCENARIOS for s in range(1, 11)
                for m in names]
    else:
        jobs = [("urban", 500, n, t, s, "TAPRA") for n in (8, 10, 12)
                for t in ("homogeneous", "heterogeneous") for s in range(1, 11)]
    jobs.sort(key=lambda j: -j[1] * j[2])
    t0 = time.time()
    with Pool(int(sys.argv[2]) if len(sys.argv) > 2 else 2) as pool:
        for i, p in enumerate(pool.imap_unordered(run, jobs), 1):
            print(f"{i}/{len(jobs)} {(time.time()-t0)/60:.1f} min {os.path.basename(p)}", flush=True)
