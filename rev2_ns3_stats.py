"""Rev. 2: packet-level statistics quoted in the ns-3 Validation section.
  python rev2_ns3_stats.py runs   (run folders with planner_slot_node.csv and the ns-3 outputs)"""
import glob, os, sys
import numpy as np, pandas as pd

root = sys.argv[1] if len(sys.argv) > 1 else "runs"
BASE = ("_tapra", "_vtopa", "_full")


def load(d, tag):
    pl = pd.read_csv(f"{d}/planner_slot_node.csv")
    ns = pd.read_csv(f"{d}/ns3_slot_node_{tag}.csv")
    ns = ns.groupby(["slot", "node", "demand_mbps"], as_index=False).agg(delivered_mbps=("delivered_mbps", "mean"))
    m = pl.merge(ns, on=["slot", "node", "demand_mbps"])
    fl = pd.read_csv(f"{d}/ns3_flow_{tag}.csv")
    return m, fl


for tag, label in [("inslot_servable_rts", "SNAPSHOT"), ("continuous_servable_rts", "CONTINUOUS")]:
    cls, flows, per_run = [], [], []
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        name = os.path.basename(d)
        if not name.endswith(BASE) and "_rep" not in name:
            continue
        if not os.path.exists(f"{d}/ns3_flow_{tag}.csv"):
            continue
        m, fl = load(d, tag)
        a = m[m.pred_served_mbps > 0].copy()
        a["frac"] = (a.delivered_mbps / a.pred_served_mbps).clip(upper=1.0)
        a["run"] = name
        cls.append(a)
        fl["run"] = name
        flows.append(fl)
    A = pd.concat(cls)
    F = pd.concat(flows)
    het = A[A.run.str.contains("hete")]
    print(f"== {label}: {A.run.nunique()} runs")
    print("  admitted delivery by demand class (heterogeneous runs):",
          (het.groupby("demand_mbps").frac.mean() * 100).round(1).to_dict())
    print("  admitted delivery all runs: %.1f" % (A.groupby("run").frac.mean().mean() * 100))
    print("  flows with no delivered packet: %.1f%% of %d" % ((F.rx_pkts == 0).mean() * 100, len(F)))
    print("  PDR mean over flows %.1f%%, packet-weighted %.1f%%" % (F.pdr.mean() * 100, F.rx_pkts.sum() / F.tx_pkts.sum() * 100))
    ok = F[F.rx_pkts > 0]
    print("  mean of per-flow mean delay: %.0f ms; packet-weighted mean delay %.0f ms" %
          (ok.mean_delay_ms.mean(), (ok.mean_delay_ms * ok.rx_pkts).sum() / ok.rx_pkts.sum()))
    good = F[F.pdr >= 0.99]
    print("  flows with PDR >= 0.99: %.1f%%; median across them of per-flow mean delay %.1f ms (TAPRA only %.1f)" %
          (len(good) / len(F) * 100, good.mean_delay_ms.median(), good[good.run.str.contains("tapra")].mean_delay_ms.median()))
    print("  median across all delivering flows of per-flow mean delay %.1f ms" % ok.mean_delay_ms.median())
