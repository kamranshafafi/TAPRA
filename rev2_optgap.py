"""Rev. 2, Reviewer 3 (optimality): per-slot gap between the position TAPRA selects and a
dense search of the same reachable region (5 m horizontal, 10 m vertical grid, then pattern
search from the 5 best points), same objective, same previous UAV position.
Slots 10, 20, ..., 50 of seeds 1-2 in the six scenarios. Output: results/rev2_optgap.csv"""
import pickle, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
import batch
from link_model import LinkModel
from optimizer import Config, Evaluator, pattern_search
from scenario_gen import generate
from rev2_cache import path

def job(a):
    env, area, n, tr, seed = a
    import os
    f = f"results/optgap/{env}_{area}_{tr}_s{seed}.pkl"
    if os.path.exists(f): return pickle.load(open(f, "rb"))
    os.makedirs("results/optgap", exist_ok=True)
    rows = _job(a); pickle.dump(rows, open(f, "wb")); return rows

def _job(a):
    env, area, n, tr, seed = a
    cfg = Config(); lm = LinkModel(env=env)
    sc = generate(env, area, n, tr, 50, cfg.tau, seed=seed)
    traj = pickle.load(open(path(env, area, n, tr, seed, "TAPRA"), "rb"))
    P = [np.array(x["position"], float) for x in sc["UENodes"]]
    T = np.array([x["traffic_demand_mbps"] for x in sc["UENodes"]], float)
    ev = Evaluator(lm, sc["Buildings"], area, cfg); R = cfg.v_max * cfg.tau
    rows = []
    for k in range(9, 50, 10):                      # slots 10, 20, ..., 50 (0-based index)
        qp = traj[k - 1]["q"]
        ev.set_slot(np.array([p[k] for p in P]), np.array([p[k - 1] for p in P]), qp, T)
        v_tapra = ev.evaluate(traj[k]["q"])[0]
        xs = np.arange(-R, R + 1e-6, 5.0); zs = np.arange(cfg.z_min, cfg.z_max + 1e-6, 10.0)
        pts = []
        for dx in xs:
            for dy in xs:
                for z in zs:
                    q = np.array([qp[0] + dx, qp[1] + dy, z])
                    if np.linalg.norm(q - qp) <= R and ev.valid(q): pts.append(q)
        vals = np.array([ev.evaluate(q)[0] for q in pts])
        best = vals.max()
        for i in np.argsort(-vals)[:5]:
            q, v = pattern_search(ev, pts[i], 5.0, 200, qp, R)
            best = max(best, v[0])
        rows.append(dict(env=env, area=area, traffic=tr, seed=seed, slot=k + 1, grid_points=len(pts),
                         tapra=v_tapra / T.sum(), dense=best / T.sum(), mode=traj[k]["mode"]))
    return rows

if __name__ == "__main__":
    jobs = [(e, a, n, t, s) for (e, a, n, t) in batch.SCENARIOS for s in (1, 2)]
    R = []
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 1) as p:
        for r in p.imap_unordered(job, jobs): R += r; print(len(R), flush=True)
    pd.DataFrame(R).to_csv("results/rev2_optgap.csv", index=False); print("done")
