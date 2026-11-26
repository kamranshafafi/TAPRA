"""Rev. 2, Reviewer 3: impact of UE speed, slot duration and key TAPRA parameters.
Urban 500 m, 20 UEs, both traffic types, seeds 1-10. Resumable: appends to results/rev2_sens.csv.
Horizon fixed at 250 s, so slots = 250 / tau."""
import csv, os, sys, time
from multiprocessing import Pool
import numpy as np
import scenario_gen
from link_model import LinkModel
from optimizer import Config, run_trajectory, realised_inslot
from tapra_export import rooftop_baseline, uav_track

OUT = "results/rev2_sens.csv"
HORIZON = 250.0
BASE_TYPES = dict(scenario_gen.NODE_TYPES)
METH = {"TAPRA": ("adaptive", {}), "Slot-start objective": ("adaptive", {"slot_average": False}),
        "VTOPA rule": ("full", {"objective": "vtopa"}), "Traffic centroid": ("centroid", {})}
FIELDS = ["sweep", "value", "traffic", "seed", "method", "planned_share", "realised_share",
          "node_outage", "jain", "evals_per_slot", "sec_per_slot", "full_searches"]

def jobs():
    J = []
    for tr in ("homogeneous", "heterogeneous"):
        for s in range(1, 11):
            for m in METH:
                for v in (0.5, 2.0, 4.0):                      # speed multiplier (1.0 = paper)
                    J.append(("speed", v, tr, s, m, {}))
                for v in (2.0, 10.0):                          # tau (5 = paper)
                    J.append(("tau", v, tr, s, m, {"tau": v}))
                J.append(("base", 1.0, tr, s, m, {}))
            P = [("J", 1, {"instants": (0.5,)}), ("J", 5, {"instants": (0.1, 0.3, 0.5, 0.7, 0.9)}),
                 ("mu", 0.0, {"momentum": 0.0}), ("mu", 0.4, {"momentum": 0.4}), ("mu", 1.0, {"momentum": 1.0}),
                 ("omega", 0.0, {"traffic_weight": 0.0}), ("omega", 0.6, {"traffic_weight": 0.6}),
                 ("Delta", 0.02, {"trigger_drop": 0.02}), ("Delta", 0.10, {"trigger_drop": 0.10}),
                 ("K_f", 5, {"period": 5}), ("K_f", 20, {"period": 20}),
                 ("Q_xy", 11, {"grid_xy": 11}), ("Q_xy", 31, {"grid_xy": 31}),
                 ("E_a", 30, {"max_evals_adapt": 30}), ("E_a", 120, {"max_evals_adapt": 120})]
            for sw, v, over in P:
                J.append((sw, v, tr, s, "TAPRA", over))
    return J

def key(j): return f"{j[0]}|{float(j[1])}|{j[2]}|{j[3]}|{j[4]}"

def run(j):
    sweep, v, tr, seed, name, over = j
    method, mo = METH[name]
    cfg = Config(**{**mo, **over})
    scenario_gen.NODE_TYPES = {k: (a * v, b * v) for k, (a, b) in BASE_TYPES.items()} if sweep == "speed" else dict(BASE_TYPES)
    slots = int(round(HORIZON / cfg.tau))
    sc = scenario_gen.generate("urban", 500, 20, tr, slots, cfg.tau, seed=seed)
    lm = LinkModel(env="urban")
    res = run_trajectory(sc, lm, cfg, method, fixed_q=rooftop_baseline(sc["Buildings"]))
    m = realised_inslot(sc, res, lm, cfg, uav_track, details=True)
    return dict(sweep=sweep, value=v, traffic=tr, seed=seed, method=name,
                planned_share=np.mean([r.served_mbps / r.demand_mbps for r in res]),
                realised_share=m["realised_share"], node_outage=m["node_outage"], jain=m["jain"],
                evals_per_slot=np.mean([r.evals for r in res[1:]]),
                sec_per_slot=np.mean([r.seconds for r in res[1:]]),
                full_searches=sum(r.mode in ("full", "adaptive+full") for r in res[1:]))

if __name__ == "__main__":
    done = set()
    if os.path.exists(OUT):
        import pandas as pd
        d = pd.read_csv(OUT)
        done = {f"{r.sweep}|{float(r.value)}|{r.traffic}|{r.seed}|{r.method}" for r in d.itertuples()}
    todo = [j for j in jobs() if key(j) not in done]
    if os.environ.get("REDUCED"): todo = [j for j in todo if j[2] == "homogeneous" or (j[0] in ("speed", "tau", "base") and j[3] <= 5)]
    if len(sys.argv) > 1: todo = todo[:int(sys.argv[1])]
    print(len(todo), "jobs to run", flush=True)
    new = not os.path.exists(OUT)
    with open(OUT, "a", newline="") as f, Pool(2) as p:
        w = csv.DictWriter(f, FIELDS)
        if new: w.writeheader()
        for i, r in enumerate(p.imap_unordered(run, todo), 1):
            w.writerow(r); f.flush()
            print(i, len(todo), r["sweep"], r["value"], r["method"], round(r["realised_share"], 3), flush=True)
