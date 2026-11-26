"""Rev. 2, Reviewer 1 comment 2: does sampling density change the served share or the ranking?
Cached plans evaluated at the true positions with three sampling grids per 5 s slot:
3 instants (0.25, 0.5, 0.75 tau), every 1 s, every 0.25 s. Positions interpolated on the 1 s tracks."""
import pickle, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
import batch
from link_model import LinkModel
from optimizer import Config
from scenario_gen import generate
from tapra_export import uav_track
from los_geometry import boxes_from_buildings
from rev2_cache import path, METHODS

GRIDS = {"3 instants": np.array([0.25, 0.5, 0.75]) * 5.0, "1 s": np.arange(0.5, 5.0, 1.0),
         "0.25 s": np.arange(0.125, 5.0, 0.25)}

def job(a):
    env, area, n, tr, seed, name = a
    cfg = Config(); lm = LinkModel(env=env)
    sc = generate(env, area, n, tr, 50, cfg.tau, seed=seed)
    traj = pickle.load(open(path(env, area, n, tr, seed, name), "rb"))
    B = boxes_from_buildings(sc["Buildings"])
    T = np.array([x["traffic_demand_mbps"] for x in sc["UENodes"]], float)
    tracks = [np.array(x["track_1s"]) for x in sc["UENodes"]]
    ut = np.array(uav_track([r["q"].tolist() for r in traj], cfg.tau, cfg.v_max))
    out = []
    for gname, grid in GRIDS.items():
        sh, S, los_changes = [], [], 0
        for r in traj:
            adm = np.asarray(r["admitted"], bool); prev = None
            for s in grid:
                t = (r["k"] - 1) * cfg.tau + s
                q = np.array([np.interp(t, ut[:, 0], ut[:, c]) for c in (1, 2, 3)])
                P = np.array([[np.interp(t, tr_[:, 0], tr_[:, c]) for c in (1, 2, 3)] for tr_ in tracks])
                snr, los = lm.snr_db(q, P, B); g = lm.goodput(snr)
                ok = adm & (g > 0); sat = np.zeros(len(T))
                if ok.any(): sat[ok] = lm.fraction(int(ok.sum()), (T[ok] / g[ok]).sum())
                sh.append((sat * T).sum() / T.sum()); S.append(sat)
                if prev is not None: los_changes += int((prev != los).sum())
                prev = los
        S = np.array(S)
        out.append(dict(env=env, area=area, traffic=tr, seed=seed, method=name, grid=gname,
                        served_share=float(np.mean(sh)), outage=float((S < 0.95).mean()),
                        los_changes_per_link_min=los_changes / len(T) / (len(traj) * cfg.tau / 60)))
    return out

if __name__ == "__main__":
    jobs = [(e, a, n, t, s, m) for (e, a, n, t) in batch.SCENARIOS for s in range(1, 11) for m in METHODS]
    R = []
    with Pool(int(sys.argv[1]) if len(sys.argv) > 1 else 1) as p:
        for i, r in enumerate(p.imap_unordered(job, jobs), 1):
            R += r
            if i % 30 == 0: print(i, flush=True)
    pd.DataFrame(R).to_csv("results/rev2_sampling.csv", index=False); print("done")
