import numpy as np, time, sys
sys.path.insert(0,'.')
from link_model import LinkModel
from scenario_gen import generate
from optimizer import run_trajectory, Config, realised_inslot
from tapra_export import uav_track
from los_geometry import boxes_from_buildings
lm = LinkModel()
for env, area, n, tr, seed in [("urban",500,20,"homogeneous",1), ("urban",500,20,"heterogeneous",2)]:
    sc = generate(env, area, n, tr, seed=seed)
    B = boxes_from_buildings(sc["Buildings"])
    c = np.array([((b[0]+b[1])/2, (b[2]+b[3])/2) for b in B]); i = np.argmin(np.linalg.norm(c, axis=1))
    rooftop = [c[i,0], c[i,1], max(30.0, B[i,4] + 5)]
    res = {}
    for m in ("full", "adaptive", "fixed"):
        t0 = time.time()
        r = run_trajectory(sc, lm, Config(), m, fixed_q=rooftop)
        share = np.array([s.served_mbps / s.demand_mbps for s in r])
        out = np.array([[s.per_node_served[j] < 0.95 * sc["UENodes"][j]["traffic_demand_mbps"] for j in range(n)] for s in r])
        z = np.array([s.q[2] for s in r]); step = np.linalg.norm(np.diff([s.q for s in r], axis=0), axis=1)
        secs = np.array([s.seconds for s in r[1:]])
        res[m] = share
        if m != "fixed":
            print(f"   {m}: realised with geometry every second {realised_inslot(sc, r, lm, Config(), uav_track):.1%}")
        print(f"{sc['scenario_name']:32s} {m:8s} served {share.mean():.1%} (min {share.min():.1%}), node outage {out.mean():.1%}, "
              f"z {z.min():.0f}-{z.max():.0f} m, step mean {step.mean():.0f} m, per-slot time {secs.mean():.2f}s, evals {np.mean([s.evals for s in r[1:]]):.0f}, "
              f"full searches {sum(s.mode!='adaptive' for s in r[1:])}, total {time.time()-t0:.0f}s")
    print(f"   adaptive vs full-every-slot: mean gap {np.mean(res['full']-res['adaptive'])*100:.1f} points of demand")
