"""Rev. 2, Reviewer 1 comment 1 and comment 3 (best subset), Reviewer 3 (optimality,
fairness). At the fixed TAPRA waypoints, compare admission rules:
  TAPRA     : cheapest airtime first, best prefix (the paper's rule)
  All       : admit every UE with a usable predicted link at one or more instants
  SNR>=g    : admit UEs whose predicted SNR is >= g dB at every instant (g swept)
  Best      : exhaustive best subset of the planner objective (small N only)
Service is evaluated at the true positions every second (admitted set fixed per slot).
Metrics are over ALL UEs requesting service."""
import itertools, os, pickle, sys
import numpy as np, pandas as pd
from link_model import LinkModel
from optimizer import Config, Evaluator
from scenario_gen import generate
from tapra_export import uav_track
from los_geometry import boxes_from_buildings
from rev2_cache import path

THR = [8, 11, 15, 18, 23]   # SNR thresholds of MCS 1, 2, 3, 4, 5 in the calibration

def phi(lm, n, L):
    n = np.asarray(n, float); L = np.asarray(L, float)
    f = np.minimum(1 - lm.KAPPA * np.maximum(0, n - lm.N0), (lm.ETA0 - lm.BETA * n) / np.maximum(L, 1e-12))
    return np.where(n > 0, f, 0.0)

def planner_value(lm, G, T, masks):
    """Slot-average served demand (planner model) for many admission masks (M x N)."""
    val = np.zeros(len(masks))
    for g in G:
        ok = masks & (g > 0)
        n = ok.sum(1)
        L = (ok * np.where(g > 0, T / np.maximum(g, 1e-12), 0)).sum(1)
        val += (ok * T).sum(1) * phi(lm, n, L)
    return val / len(G)

def realised(sc, qs, adms, lm, cfg, B):
    T = np.array([n["traffic_demand_mbps"] for n in sc["UENodes"]], float)
    tracks = [np.array(n["track_1s"]) for n in sc["UENodes"]]
    ut = np.array(uav_track([q.tolist() for q in qs], cfg.tau, cfg.v_max))
    S = []
    for k, adm in enumerate(adms):
        for s in np.arange(0.5, cfg.tau, 1.0):
            t = k * cfg.tau + s
            q = np.array([np.interp(t, ut[:, 0], ut[:, c]) for c in (1, 2, 3)])
            P = np.array([[np.interp(t, tr[:, 0], tr[:, c]) for c in (1, 2, 3)] for tr in tracks])
            g = lm.goodput(lm.snr_db(q, P, B)[0])
            ok = adm & (g > 0)
            sat = np.zeros(len(T))
            if ok.any():
                sat[ok] = lm.fraction(int(ok.sum()), (T[ok] / g[ok]).sum())
            S.append(sat)
    S = np.array(S)                                   # instants x UEs
    share = float(((S * T).sum(1) / T.sum()).mean())
    per_ue = S.mean(0)
    adm_arr = np.repeat(np.array(adms), 5, axis=0)
    return dict(served_share=share, outage=float((S < 0.95).mean()),
                jain_time=float(np.mean([(s.sum() ** 2) / (len(s) * (s ** 2).sum()) if s.sum() > 0 else 0 for s in S])),
                jain_ue=float(per_ue.sum() ** 2 / (len(per_ue) * (per_ue ** 2).sum())) if per_ue.sum() > 0 else 0.0,
                ue_never=float((per_ue < 0.05).mean()),
                admitted_unserved=float(((adm_arr) & (S < 0.95)).sum() / max(adm_arr.sum(), 1)),
                admitted_frac=float(adm_arr.mean()))

def analyse(env, area, n, traffic, seed, best=False):
    cfg = Config(); lm = LinkModel(env=env)
    sc = generate(env, area, n, traffic, 50, cfg.tau, seed=seed)
    traj = pickle.load(open(path(env, area, n, traffic, seed, "TAPRA"), "rb"))
    nodes_all = [np.array(x["position"], float) for x in sc["UENodes"]]
    T = np.array([x["traffic_demand_mbps"] for x in sc["UENodes"]], float)
    ev = Evaluator(lm, sc["Buildings"], area, cfg); B = ev.boxes
    pol = {"TAPRA": [], "All eligible": []} | {f"SNR>={g} dB": [] for g in THR}
    if best: pol["Best subset"] = []
    gaps = []
    q_prev = None
    for k, r in enumerate(traj):
        nodes = np.array([p[k] for p in nodes_all])
        prev = np.array([p[k - 1] for p in nodes_all]) if k else None
        ev.set_slot(nodes, prev, q_prev, T)
        q = r["q"]
        G = ev._goodputs(q)
        snr = [lm.snr_db(ev.uav_at(q, s), P, B)[0] for s, P in zip(ev.times, ev.node_sets)]
        pol["TAPRA"].append(np.asarray(r["admitted"], bool))
        pol["All eligible"].append(np.any([g > 0 for g in G], axis=0))
        for g_ in THR:
            pol[f"SNR>={g_} dB"].append(np.all([s >= g_ for s in snr], axis=0) & np.all([g > 0 for g in G], axis=0))
        if best:
            masks = np.array(list(itertools.product([0, 1], repeat=n)), bool)
            v = planner_value(lm, G, T, masks)
            vt = planner_value(lm, G, T, pol["TAPRA"][-1][None, :])[0]
            pol["Best subset"].append(masks[int(np.argmax(v))])
            gaps.append((v.max(), vt))
        q_prev = q
    qs = [r["q"] for r in traj]
    rows = []
    for name, adms in pol.items():
        m = realised(sc, qs, adms, lm, cfg, B)
        m.update(env=env, area=area, nodes=n, traffic=traffic, seed=seed, policy=name)
        rows.append(m)
    if best:
        g = np.array(gaps)
        rows.append(dict(env=env, area=area, nodes=n, traffic=traffic, seed=seed, policy="_gap",
                         planner_best=float(g[:, 0].sum()), planner_tapra=float(g[:, 1].sum()),
                         slots_equal=float(np.mean(np.isclose(g[:, 0], g[:, 1], atol=1e-6)))))
    return rows

if __name__ == "__main__":
    print(pd.DataFrame(analyse("urban", 500, 8, "homogeneous", 1, best=True)).round(3).to_string())
