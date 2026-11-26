"""
Compare what the planner predicted with what ns-3 measured.

  python compare_ns3.py runs/urban500_s1_adaptive snapshot_all
  (the second argument is the file tag: <mode>_<senders>[_rts])
"""
import sys

import numpy as np
import pandas as pd

import argparse
import os
_ap = argparse.ArgumentParser()
_ap.add_argument("dir")
_ap.add_argument("tag", nargs="?", default="snapshot_all")
_ap.add_argument("--summary", default=None, help="append one result row to this CSV")
_a = _ap.parse_args()
d = _a.dir
tag = _a.tag
mode = tag.split("_")[0]
pl = pd.read_csv(f"{d}/planner_slot_node.csv")
ns = pd.read_csv(f"{d}/ns3_slot_node_{tag}.csv")
# inslot mode has several rows per slot and node (one per instant): average them
ns = ns.groupby(["slot", "node", "demand_mbps"], as_index=False).agg(
    delivered_mbps=("delivered_mbps", "mean"), los_ns3=("los_ns3", "mean"))
ns["los_ns3"] = (ns.los_ns3 >= 0.5).astype(int)
fl = pd.read_csv(f"{d}/ns3_flow_{tag}.csv")
m = pl.merge(ns, on=["slot", "node", "demand_mbps"])
# snapshot mode is compared with the static prediction (slot start, UAV at q);
# continuous mode with the slot-average prediction
if mode == "snapshot" and "pred_static_mbps" in m.columns:
    m["pred_served_mbps"] = m["pred_static_mbps"]

los_agree = (m.pred_los == m.los_ns3).mean()
m["sat_ns3"] = (m.delivered_mbps / m.demand_mbps).clip(upper=1.0)
m["sat_pred"] = (m.pred_served_mbps / m.demand_mbps).clip(upper=1.0)
per_slot = m.groupby("slot").agg(pred=("pred_served_mbps", "sum"),
                                 meas=("delivered_mbps", "sum"),
                                 dem=("demand_mbps", "sum"))
share_pred = (per_slot.pred / per_slot.dem).mean()
share_meas = (np.minimum(per_slot.meas, per_slot.dem) / per_slot.dem).mean()
srv = m[m.pred_served_mbps > 0]
sat_srv = (srv.delivered_mbps / srv.pred_served_mbps).clip(upper=1.0).mean()
outage = (m.sat_ns3 < 0.95).mean()
err = (m.sat_ns3 - m.sat_pred)

print(f"{d} [{tag}]  {m.slot.nunique()} slots x {m.node.nunique()} nodes")
if mode == "snapshot":
    print(f"  LoS agreement planner vs ns-3 buildings: {los_agree:.1%}")
else:
    print(f"  LoS at slot start (planner) vs during the slot (ns-3), agreement: {los_agree:.1%}")
print(f"  served share of demand: planner {share_pred:.1%}, ns-3 {share_meas:.1%}")
print(f"  nodes the planner serves: ns-3 delivers {sat_srv:.1%} of their planned rate on average")
print(f"  per-node satisfaction error (ns-3 minus planner): mean {err.mean():+.3f}, "
      f"95% of links within [{err.quantile(0.025):+.2f}, {err.quantile(0.975):+.2f}]")
print(f"  node outage (below 95% of demand) in ns-3: {outage:.1%}")
ok = fl[fl.rx_pkts > 0]
print(f"  packet delivery ratio: mean {fl.pdr.mean():.1%}; "
      f"mean delay of delivered packets: {ok.mean_delay_ms.mean():.1f} ms")
mism = m[m.pred_los != m.los_ns3]
if len(mism) and mode == "snapshot":
    print(f"  {len(mism)} LoS mismatches, first ones:")
    print(mism.head(5)[["slot", "node", "pred_los", "los_ns3"]].to_string(index=False))

if _a.summary:
    py = {}
    if os.path.exists(f"{d}/python_summary.txt"):
        for line in open(f"{d}/python_summary.txt"):
            k, v = line.strip().split("=")
            py[k] = float(v)
    row = {"run": os.path.basename(os.path.normpath(d)), "tag": tag,
           "planned_share": share_pred, "python_realised_share": py.get("realised_share", np.nan),
           "ns3_share": share_meas, "admitted_delivery": sat_srv, "pdr": fl.pdr.mean(),
           "mean_delay_ms": ok.mean_delay_ms.mean(), "node_outage": outage,
           "los_agreement": los_agree}
    new = not os.path.exists(_a.summary)
    pd.DataFrame([row]).to_csv(_a.summary, mode="a", header=new, index=False)
    print("appended to", _a.summary)
