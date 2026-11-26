"""Rev. 2, Reviewer 1 comment 4: rankings under independently parameterized channels and
results by environment and elevation angle. Re-evaluates the cached plans (same waypoints
and admitted sets) with the true channel = planner model (A2G), A2G + shadowing, or
ITU-R P.1411 (ns-3 models). Geometry every 1 s.
Outputs: results/rev2_channel.csv (per run) and results/rev2_elevation.csv (per run x bin)."""
import os, pickle, sys
from multiprocessing import Pool
import numpy as np, pandas as pd
import batch
from link_model import LinkModel
from optimizer import Config
from robustness import ShadowedLinkModel, P1411LinkModel
from scenario_gen import generate
from tapra_export import uav_track
from los_geometry import boxes_from_buildings
from rev2_cache import path, METHODS

BINS = [0, 15, 30, 45, 60, 90]

def evaluate(sc, traj, lm, cfg):
    B = boxes_from_buildings(sc["Buildings"])
    T = np.array([n["traffic_demand_mbps"] for n in sc["UENodes"]], float)
    tracks = [np.array(n["track_1s"]) for n in sc["UENodes"]]
    ut = np.array(uav_track([r["q"].tolist() for r in traj], cfg.tau, cfg.v_max))
    shares, S, E, LOS = [], [], [], []
    for r in traj:
        adm = np.asarray(r["admitted"], bool)
        for s in np.arange(0.5, cfg.tau, 1.0):
            t = (r["k"] - 1) * cfg.tau + s
            q = np.array([np.interp(t, ut[:, 0], ut[:, c]) for c in (1, 2, 3)])
            P = np.array([[np.interp(t, tr[:, 0], tr[:, c]) for c in (1, 2, 3)] for tr in tracks])
            snr, los = lm.snr_db(q, P, B)
            g = lm.goodput(snr)
            ok = adm & (g > 0)
            sat = np.zeros(len(T))
            if ok.any():
                sat[ok] = lm.fraction(int(ok.sum()), (T[ok] / g[ok]).sum())
            shares.append((sat * T).sum() / T.sum())
            d = np.linalg.norm(P - q, axis=1)
            E.append(np.degrees(np.arcsin(np.clip((q[2] - P[:, 2]) / d, -1, 1))))
            S.append(sat); LOS.append(los)
    return np.array(shares), np.array(S), np.array(E), np.array(LOS)

def job(a):
    env, area, n, tr, seed, name = a
    cfg = Config()
    sc = generate(env, area, n, tr, 50, cfg.tau, seed=seed)
    traj = pickle.load(open(path(env, area, n, tr, seed, name), "rb"))
    rows, erows = [], []
    for ch, lm in [("A2G (planner model)", LinkModel(env=env)),
                   ("A2G + shadowing", ShadowedLinkModel(seed=20_000 + seed, env=env)),
                   ("ITU-R P.1411", P1411LinkModel(env=env))]:
        sh, S, E, L = evaluate(sc, traj, lm, cfg)
        base = dict(env=env, area=area, nodes=n, traffic=tr, seed=seed, method=name, channel=ch)
        rows.append(base | dict(served_share=sh.mean(), outage=float((S < 0.95).mean())))
        b = np.digitize(E, BINS[1:-1])
        for i in range(len(BINS) - 1):
            m = b == i
            if m.any():
                erows.append(base | dict(elev_bin=f"{BINS[i]}-{BINS[i+1]}", link_seconds=int(m.sum()),
                                         los_prob=float(L[m].mean()), satisfaction=float(S[m].mean())))
    return rows, erows

if __name__ == "__main__":
    jobs = [(e, a, n, t, s, m) for (e, a, n, t) in batch.SCENARIOS for s in range(1, 11) for m in METHODS]
    W = int(sys.argv[1]) if len(sys.argv) > 1 else 1
    R, ER = [], []
    with Pool(W) as p:
        for i, (r, er) in enumerate(p.imap_unordered(job, jobs), 1):
            R += r; ER += er
            if i % 20 == 0: print(i, len(jobs), flush=True)
    pd.DataFrame(R).to_csv("results/rev2_channel.csv", index=False)
    pd.DataFrame(ER).to_csv("results/rev2_elevation.csv", index=False)
    print("done")
