"""Rev. 2: merge the robustness reruns of TAPRA and the uncertainty-aware variant, and print the numbers of the text."""
import numpy as np, pandas as pd
from scipy import stats

m = pd.read_csv("results/robustness_merged.csv")
f = pd.read_csv("results/robustness_fix.csv")
FIX = f.method.unique().tolist()
r = pd.concat([m[~m.method.isin(FIX)], f], ignore_index=True)
r.to_csv("results/robustness_rev2.csv", index=False)
print("methods replaced:", FIX, "rows", len(f))
key = ["env", "area", "traffic", "seed", "method"]
p = r[r.condition == "perfect"].set_index(key).realised_share
x = r[r.condition != "perfect"].copy()
x["d"] = (x.set_index(key).realised_share.values - p.reindex(x.set_index(key).index).values) * 100


def tci(v):
    v = np.asarray(v, float)
    return stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))


g = x.groupby(["condition", "method"]).d.agg(["mean", tci, "count"]).round(2)
print(g.to_string())
w = r.pivot_table(index=["env", "area", "traffic", "seed", "condition"], columns="method", values="realised_share") * 100
ua = [c for c in w.columns if "uncertainty" in c][0]
for cond in ["gps_10m", "perfect", "stale"]:
    d = (w[ua] - w["TAPRA"]).xs(cond, level="condition")
    print(cond, "UA - TAPRA %.2f +- %.2f" % (d.mean(), tci(d)))
d = (w["TAPRA"] - w["VTOPA rule"]).xs("gps_10m", level="condition")
print("gps10 TAPRA - VTOPA by scenario", d.groupby(level=[0, 1, 2]).mean().round(2).to_dict())
for cond in ["itu_p1411", "shadowing", "map_heights", "map_missing"]:
    print(cond, x[x.condition == cond].groupby("method").d.mean().round(2).to_dict())
# layout-clustered CI check: average urban homo/hetero per layout (seed) first
x["layout"] = x.env + x.area.astype(str) + "_" + x.seed.astype(str)
mx = 0
for (c, me), gg in x.groupby(["condition", "method"]):
    a = tci(gg.d); b = tci(gg.groupby("layout").d.mean()); mx = max(mx, abs(a - b))
print("max change of CI half-width when clustered by layout: %.2f" % mx)
