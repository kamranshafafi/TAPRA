"""Rev. 2, Reviewer 1 comment 1: admission rules at the FIXED TAPRA waypoints, in ns-3.
For every run folder runs/<name>_tapra it writes runs/<name>_tapra_adm-<policy>/ with the
same waypoints and tracks, and a planner_slot_node.csv in which only the UEs admitted by
<policy> have pred_served_mbps > 0 (ns-3 --senders=servable lets exactly those send).
  python rev2_policy_export.py --policies all,snr11
Policies: all    = every UE with a usable predicted link at one or more instants
          snrX   = UEs whose predicted SNR is >= X dB at every instant
(TAPRA's own admission is the original folder.)"""
import argparse, glob, json, os, shutil
import numpy as np, pandas as pd
from link_model import LinkModel
from optimizer import Config, Evaluator

ap = argparse.ArgumentParser()
ap.add_argument("--policies", default="all,snr11")
ap.add_argument("--runs", default="runs")
a = ap.parse_args()
for base in sorted(glob.glob(os.path.join(a.runs, "*_tapra"))):
    sc = json.load(open(os.path.join(base, "scenario.json")))
    env = sc["generator"]["environment"]
    lm, cfg = LinkModel(env=env), Config()
    area = sc["simulationParams"]["area_size_m"]
    ev = Evaluator(lm, sc["Buildings"], area, cfg)
    T = np.array([n["traffic_demand_mbps"] for n in sc["UENodes"]], float)
    P = [np.array(n["position"], float) for n in sc["UENodes"]]
    uav = pd.read_csv(os.path.join(base, "uav_slot.csv")).sort_values("slot")
    qs = uav[["x", "y", "z"]].values
    pl = pd.read_csv(os.path.join(base, "planner_slot_node.csv"))
    adm = {p: {} for p in a.policies.split(",")}
    for k in range(len(qs)):
        nodes = np.array([p[k] for p in P])
        prev = np.array([p[k - 1] for p in P]) if k else None
        ev.set_slot(nodes, prev, qs[k - 1] if k else None, T)
        G = ev._goodputs(qs[k])
        snr = [lm.snr_db(ev.uav_at(qs[k], s), X, ev.boxes)[0] for s, X in zip(ev.times, ev.node_sets)]
        for p in adm:
            if p == "all":
                m = np.any([g > 0 for g in G], axis=0)
            else:
                thr = float(p[3:])
                m = np.all([x >= thr for x in snr], axis=0) & np.all([g > 0 for g in G], axis=0)
            adm[p][k + 1] = m
    for p, d in adm.items():
        out = f"{base}_adm-{p}"
        os.makedirs(out, exist_ok=True)
        for f in os.listdir(base):
            if f.endswith(".csv") and not f.startswith("ns3_") or f in ("scenario.json",):
                if f != "planner_slot_node.csv":
                    shutil.copy(os.path.join(base, f), out)
        q = pl.copy()
        ids = sorted(q.node.unique())
        q["admitted"] = [int(d[s][ids.index(n)]) for s, n in zip(q.slot, q.node)]
        q["pred_served_mbps"] = np.where(q.admitted == 1, q.demand_mbps, 0.0)
        q.to_csv(os.path.join(out, "planner_slot_node.csv"), index=False)
        par = open(os.path.join(base, "params.txt")).read().replace(
            os.path.abspath(base), os.path.abspath(out))
        open(os.path.join(out, "params.txt"), "w").write(par)
        n_adm = np.mean([m.sum() for m in d.values()])
        print(f"{out}: {n_adm:.1f} UEs admitted per slot on average")
