"""Rev. 2, Reviewer 1 comment 5: time from UE reports to motion and admission commands,
per slot, including the slots where a full search is triggered; deadline misses; reachable
travel distance; service when the UAV departs only after the computation ends.
Single process (one core). Seeds 1-3 of the six scenarios. Output: results/rev2_latency.csv
(per slot) and results/rev2_latency_runs.csv (per run)."""
import time
import numpy as np, pandas as pd
import batch
from link_model import LinkModel
from optimizer import Config, run_trajectory, realised_inslot
from scenario_gen import generate
from tapra_export import uav_track

def delayed_track(delays):
    """UAV track when the move of slot k starts only after the computation time c_k."""
    def f(qs, tau, v_max):
        wp = [[0.0, *qs[0]]]
        for k in range(1, len(qs)):
            t0 = k * tau + min(delays[k], tau * 0.999)
            d = float(np.linalg.norm(np.asarray(qs[k]) - np.asarray(qs[k - 1])))
            wp.append([t0, *qs[k - 1]])
            wp.append([min(t0 + max(d / v_max, 1e-3), (k + 1) * tau - 1e-3), *qs[k]])
        wp.append([len(qs) * tau, *qs[-1]])
        return wp
    return f

if __name__ == "__main__":
    S, R = [], []
    for (env, area, n, tr) in batch.SCENARIOS:
        for seed in (1, 2, 3):
            cfg = Config(); lm = LinkModel(env=env)
            sc = generate(env, area, n, tr, 50, cfg.tau, seed=seed)
            res = run_trajectory(sc, lm, cfg, "adaptive")
            c = [r.seconds for r in res]
            for k, r in enumerate(res):
                d = float(np.linalg.norm(r.q - res[k - 1].q)) if k else 0.0
                reach = cfg.v_max * max(cfg.tau - r.seconds, 0.0)
                S.append(dict(env=env, area=area, traffic=tr, seed=seed, slot=r.k, mode=r.mode,
                              seconds=r.seconds, evals=r.evals, travel_m=d, reachable_m=reach,
                              deadline_miss=int(r.seconds > cfg.tau), late_arrival=int(k > 0 and d > reach)))
            base = realised_inslot(sc, res, lm, cfg, uav_track)
            late = realised_inslot(sc, res, lm, cfg, delayed_track(c))
            R.append(dict(env=env, area=area, traffic=tr, seed=seed, served_share=base,
                          served_share_delayed_departure=late))
            print(env, area, tr, seed, round(np.median(c[1:]), 3), round(max(c[1:]), 3), round(base, 4), round(late, 4), flush=True)
    pd.DataFrame(S).to_csv("results/rev2_latency.csv", index=False)
    pd.DataFrame(R).to_csv("results/rev2_latency_runs.csv", index=False)
