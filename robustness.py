"""
Robustness study for TAPRA (answers Reviewer 3 comment 4 and Reviewer 6 comment 7).

The planner works with imperfect information; the realised service is computed with
the truth (real node tracks, real buildings, real channel), geometry every second.

Conditions:
  perfect          planner sees everything correctly (reference)
  gps_2m/5m/10m    planner sees node positions with Gaussian error (std 2, 5, 10 m)
  stale            planner sees node positions one slot old
  map_heights      planner's building heights wrong by a factor N(1, 0.2) each
  map_missing      10% of the buildings are missing from the planner's map
                   (stands in for unknown or temporary obstacles)
  shadowing        true channel adds log-normal shadowing, std from Al-Hourani et al.
                   (Globecom 2014, Table II, urban 5.8 GHz: sigma = a*exp(-b*theta))
  itu_p1411        true channel is ITU-R P.1411 (the ns-3 models, propagation.py)
                   instead of the planner's air-to-ground model

  python robustness.py --seeds 1-10 --workers 6 --out results/robustness.csv
"""
import argparse
import copy
import csv
import os
import time
from multiprocessing import Pool

import numpy as np

from link_model import LinkModel
from optimizer import Config, run_trajectory, realised_inslot
from propagation import itu1411_los_loss, itu1411_nlos_rooftop_loss
from scenario_gen import generate
from tapra_export import uav_track
from los_geometry import los_vector

CONDITIONS = ["perfect", "gps_2m", "gps_5m", "gps_10m", "stale",
              "map_heights", "map_missing", "shadowing", "itu_p1411"]
METHODS = {"TAPRA": ("adaptive", {}),
           # objective averaged over sampled true positions around each report;
           # assumed std = the condition's GPS error, otherwise 5 m (typical GPS)
           "TAPRA uncertainty-aware": ("adaptive", {"pos_samples": 3}),
           "VTOPA rule": ("full", {"objective": "vtopa"})}
SCENARIOS = [("urban", 500, 20, "homogeneous"), ("urban", 500, 20, "heterogeneous"),
             ("suburban", 500, 20, "homogeneous")]
SLOTS = 50

# Al-Hourani et al. 2014, Table II, 5.8 GHz: (a1, b1) LoS, (a2, b2) NLoS
SHADOW = {"urban": ((10.67, 0.05), (35.85, 0.04)),
          "suburban": ((11.04, 0.06), (39.56, 0.04))}


class ShadowedLinkModel(LinkModel):
    """True channel = planner's model + log-normal shadowing (random per link and time)."""

    def __init__(self, seed, **kw):
        super().__init__(**kw)
        self.rng = np.random.default_rng(seed)
        self.env = kw.get("env", "urban")

    def snr_db(self, uav, nodes, boxes):
        snr, los = super().snr_db(uav, nodes, boxes)
        nodes = np.atleast_2d(nodes)
        d = np.linalg.norm(nodes - np.asarray(uav, float), axis=1)
        theta = np.degrees(np.arcsin(np.clip((uav[2] - nodes[:, 2]) / d, -1, 1)))
        (a1, b1), (a2, b2) = SHADOW.get(self.env, SHADOW["urban"])
        sigma = np.where(los, a1 * np.exp(-b1 * theta), a2 * np.exp(-b2 * theta))
        return snr - self.rng.normal(0.0, sigma), los


class P1411LinkModel(LinkModel):
    """True channel = ITU-R P.1411 as implemented in ns-3 (LoS / NLoS over rooftop)."""

    def snr_db(self, uav, nodes, boxes):
        nodes = np.atleast_2d(nodes)
        los = los_vector(uav, nodes, boxes)
        L = np.array([itu1411_los_loss(uav, p, self.f) if l else
                      itu1411_nlos_rooftop_loss(uav, p, self.f) for p, l in zip(nodes, los)])
        return self.tx + 2 * self.g - L - self.noise_dbm, los


def planner_view(sc, cond, rng):
    """Copy of the scenario as the planner sees it."""
    p = copy.deepcopy(sc)
    if cond.startswith("gps_"):
        std = float(cond.split("_")[1].rstrip("m"))  # "gps_10m_mm" -> 10
        for n in p["UENodes"]:
            pos = np.array(n["position"], float)
            pos[:, :2] += rng.normal(0, std, pos[:, :2].shape)
            n["position"] = pos.tolist()
    elif cond == "stale":
        for n in p["UENodes"]:
            pos = n["position"]
            n["position"] = [pos[0]] + pos[:-1]
    elif cond == "map_heights":
        for b in p["Buildings"]:
            b["height"] = max(3.0, b["height"] * rng.normal(1.0, 0.2))
    elif cond == "map_missing":
        keep = rng.random(len(p["Buildings"])) >= 0.10
        p["Buildings"] = [b for b, k in zip(p["Buildings"], keep) if k]
    return p


def run_one(job):
    env, area, n, traffic, seed, cond, name = job
    method, over = METHODS[name]
    over = dict(over)
    if "uncertainty" in name:
        over["pos_sigma"] = float(cond.split("_")[1].rstrip("m")) if cond.startswith("gps_") else 5.0
    cfg = Config(**over)
    rng = np.random.default_rng(10_000 + seed)
    truth = generate(env, area, n, traffic, SLOTS, cfg.tau, seed=seed)
    plan_sc = planner_view(truth, cond, rng)
    lm_plan = LinkModel(env=env)
    res = run_trajectory(plan_sc, lm_plan, cfg, method)
    if cond == "shadowing":
        lm_true = ShadowedLinkModel(seed=20_000 + seed, env=env)
    elif cond == "itu_p1411":
        lm_true = P1411LinkModel(env=env)
    else:
        lm_true = lm_plan
    m = realised_inslot(truth, res, lm_true, cfg, uav_track, details=True)
    return {"env": env, "area": area, "nodes": n, "traffic": traffic, "seed": seed,
            "condition": cond, "method": name,
            "planned_share": np.mean([r.served_mbps / r.demand_mbps for r in res]),
            "realised_share": m["realised_share"], "node_outage": m["node_outage"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="1-10")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="results/robustness.csv")
    ap.add_argument("--only", default="", help="comma-separated method names to run")
    a = ap.parse_args()
    s0, s1 = (a.seeds.split("-") + [a.seeds])[:2]
    seeds = range(int(s0), int(s1) + 1)
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fields = ["env", "area", "nodes", "traffic", "seed", "condition", "method",
              "planned_share", "realised_share", "node_outage"]
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            for r in csv.DictReader(f):
                done.add((r["env"], int(r["area"]), int(r["nodes"]), r["traffic"],
                          int(r["seed"]), r["condition"], r["method"]))
    jobs = [(e, ar, n, t, s, c, m) for (e, ar, n, t) in SCENARIOS for s in seeds
            for c in CONDITIONS for m in METHODS if (e, ar, n, t, s, c, m) not in done
            and (not a.only or m in a.only.split(","))]
    print(f"{len(jobs)} runs to do")
    new = not os.path.exists(a.out)
    t0 = time.time()
    with open(a.out, "a", newline="") as f, Pool(a.workers) as pool:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        for i, row in enumerate(pool.imap_unordered(run_one, jobs), 1):
            w.writerow(row)
            f.flush()
            if i % 10 == 0 or i == len(jobs):
                print(f"{i}/{len(jobs)} done, {(time.time() - t0) / 60:.1f} min")
