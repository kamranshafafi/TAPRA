"""Summary of results/batch.csv: mean and 95% confidence interval per scenario and
method, and paired differences TAPRA minus each other method (same cities)."""
import sys

import numpy as np
import pandas as pd

d = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "results/batch.csv")
key = ["env", "area", "nodes", "traffic"]


def ci(x):
    x = np.asarray(x, float)
    return 1.96 * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


pd.set_option("display.width", 200)
for k, g in d.groupby(key):
    print(f"\n=== {k[0]}, {k[1]} m, {k[2]} nodes, {k[3]} traffic ({g.seed.nunique()} cities)")
    t = g.groupby("method").agg(realised=("realised_share", "mean"),
                                ci=("realised_share", ci), planned=("planned_share", "mean"),
                                outage=("node_outage", "mean"), jain=("jain", "mean"),
                                sec=("sec_per_slot", "mean"), evals=("evals_per_slot", "mean"))
    print((t.sort_values("realised", ascending=False) * [100, 100, 100, 100, 1, 1, 1]).round(2)
          .to_string())
    tap = g[g.method == "TAPRA"].set_index("seed").realised_share
    for m, h in g.groupby("method"):
        if m == "TAPRA":
            continue
        diff = (tap - h.set_index("seed").realised_share).dropna() * 100
        if len(diff):
            print(f"  TAPRA minus {m}: {diff.mean():+.1f} points (95% CI +-{ci(diff):.1f}, "
                  f"{len(diff)} cities)")
