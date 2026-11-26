"""
Plan a scenario with TAPRA (or a baseline) and export everything ns-3 needs.

Example:
  python tapra_export.py --env urban --area 500 --nodes 20 --traffic homogeneous \
         --seed 1 --method adaptive --out runs/urban500_s1_adaptive

Files written to --out:
  buildings.csv         xmin,xmax,ymin,ymax,height
  demands.csv           node,demand_mbps
  nodes_slot.csv        slot,node,x,y,z          (positions at slot start)
  nodes_track.csv       node,t,x,y,z             (every 1 s, for continuous mode)
  uav_slot.csv          slot,x,y,z               (planned position per slot)
  uav_track.csv         t,x,y,z                  (flies to each new position at v_max)
  planner_slot_node.csv slot,node,demand_mbps,pred_served_mbps,pred_los,pred_static_mbps,admitted
                        (pred_served = slot average, compare with --mode=continuous;
                         pred_static = slot start, UAV at its new position, compare with
                         --mode=snapshot; ns-3 lets a node send when pred_served > 0)
  planner_slot.csv      slot,mode,served_mbps,demand_mbps,margin,evals,seconds
  scenario.json         the full scenario (for reproduction)
  params.txt            values to pass to ns-3 on the command line
"""
import argparse
import json
import os

import numpy as np

from link_model import LinkModel
from optimizer import Config, run_trajectory, realised_inslot
from scenario_gen import generate
from los_geometry import boxes_from_buildings


METHODS = {  # name: (optimizer method, Config overrides)
    "tapra": ("adaptive", {}),
    "tapra_ua": ("adaptive", {"pos_sigma": 5.0, "pos_samples": 3}),
    "full": ("full", {}),
    "vtopa": ("full", {"objective": "vtopa"}),
    "centroid": ("centroid", {}),
    "fixed": ("fixed", {}),
    # backward compatibility with earlier commands
    "adaptive": ("adaptive", {}),
}


def rooftop_baseline(buildings):
    B = boxes_from_buildings(buildings)
    c = np.array([((b[0] + b[1]) / 2, (b[2] + b[3]) / 2) for b in B])
    i = int(np.argmin(np.linalg.norm(c, axis=1)))
    return [c[i, 0], c[i, 1], max(30.0, B[i, 4] + 5.0)]


def uav_track(qs, tau, v_max):
    """Waypoints: at each slot start the UAV leaves its old position and flies
    straight at v_max to the new one, then hovers until the next slot."""
    wp = [[0.0, *qs[0]]]
    for k in range(1, len(qs)):
        t0 = k * tau
        d = float(np.linalg.norm(np.asarray(qs[k]) - np.asarray(qs[k - 1])))
        wp.append([t0, *qs[k - 1]])
        wp.append([t0 + min(max(d / v_max, 1e-3), tau * 0.999), *qs[k]])  # strictly increasing times
    wp.append([len(qs) * tau, *qs[-1]])
    return wp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="urban")
    ap.add_argument("--area", type=float, default=500)
    ap.add_argument("--nodes", type=int, default=20)
    ap.add_argument("--traffic", default="homogeneous")
    ap.add_argument("--slots", type=int, default=50)
    ap.add_argument("--tau", type=float, default=5.0)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--method", default="tapra", choices=list(METHODS))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    method, over = METHODS[a.method]
    cfg = Config(tau=a.tau, **over)
    sc = generate(a.env, a.area, a.nodes, a.traffic, a.slots, a.tau, seed=a.seed)
    lm = LinkModel(env=a.env if a.env in ("suburban", "urban", "dense_urban", "highrise_urban") else "urban")
    res = run_trajectory(sc, lm, cfg, method, fixed_q=rooftop_baseline(sc["Buildings"]))

    os.makedirs(a.out, exist_ok=True)
    w = lambda name: open(os.path.join(a.out, name), "w")
    with w("scenario.json") as f:
        json.dump(sc, f)
    with w("buildings.csv") as f:
        f.write("xmin,xmax,ymin,ymax,height\n")
        for b in boxes_from_buildings(sc["Buildings"]):
            f.write(",".join(f"{v:.3f}" for v in b) + "\n")
    with w("demands.csv") as f:
        f.write("node,demand_mbps\n")
        for n in sc["UENodes"]:
            f.write(f"{n['id']},{n['traffic_demand_mbps']}\n")
    with w("nodes_slot.csv") as f:
        f.write("slot,node,x,y,z\n")
        for n in sc["UENodes"]:
            for k, p in enumerate(n["position"]):
                f.write(f"{k + 1},{n['id']},{p[0]},{p[1]},{p[2]}\n")
    with w("nodes_track.csv") as f:
        f.write("node,t,x,y,z\n")
        for n in sc["UENodes"]:
            for t, x, y, z in n["track_1s"]:
                f.write(f"{n['id']},{t},{x},{y},{z}\n")
    qs = [r.q.tolist() for r in res]
    with w("uav_slot.csv") as f:
        f.write("slot,x,y,z\n")
        for r in res:
            f.write(f"{r.k},{r.q[0]:.3f},{r.q[1]:.3f},{r.q[2]:.3f}\n")
    with w("uav_track.csv") as f:
        f.write("t,x,y,z\n")
        for t, x, y, z in uav_track(qs, a.tau, cfg.v_max):
            f.write(f"{t:.3f},{x:.3f},{y:.3f},{z:.3f}\n")
    with w("planner_slot_node.csv") as f:
        f.write("slot,node,demand_mbps,pred_served_mbps,pred_los,pred_static_mbps,admitted\n")
        for r in res:
            for j, n in enumerate(sc["UENodes"]):
                f.write(f"{r.k},{n['id']},{n['traffic_demand_mbps']},"
                        f"{r.per_node_served[j]:.4f},{int(r.los[j])},"
                        f"{r.per_node_static[j]:.4f},{int(r.admitted[j])}\n")
    with w("planner_slot.csv") as f:
        f.write("slot,mode,served_mbps,demand_mbps,margin,evals,seconds\n")
        for r in res:
            f.write(f"{r.k},{r.mode},{r.served_mbps:.4f},{r.demand_mbps},{r.margin:.4f},"
                    f"{r.evals},{r.seconds:.4f}\n")
    with w("params.txt") as f:
        f.write(f"--dir={os.path.abspath(a.out)} --slots={a.slots} --tau={a.tau} "
                f"--etaLos={lm.eta_los} --etaNlos={lm.eta_nlos}\n")
    share = np.mean([r.served_mbps / r.demand_mbps for r in res])
    real = realised_inslot(sc, res, lm, cfg, uav_track)
    with w("python_summary.txt") as f:
        f.write(f"planned_share={share:.6f}\nrealised_share={real:.6f}\n")
    print(f"{sc['scenario_name']} [{a.method}] planned served share {share:.1%} (slot average), "
          f"realised with geometry every second {real:.1%}; written to {a.out}")
    print("ns-3 arguments:", open(os.path.join(a.out, "params.txt")).read().strip())


if __name__ == "__main__":
    main()
