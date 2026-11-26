"""Paired differences TAPRA - method by scenario (same city), t-based 95% CI over 10 cities."""
import numpy as np, pandas as pd
from scipy import stats
d = pd.read_csv("results/batch_final.csv")
rows = []
for (e, a, t), g in d.groupby(["env", "area", "traffic"]):
    p = g.pivot(index="seed", columns="method", values="realised_share") * 100
    for m in p.columns:
        if m == "TAPRA": continue
        x = (p["TAPRA"] - p[m]).dropna()
        h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
        rows.append(dict(env=e, area=a, traffic=t, method=m, n=len(x), mean_pp=x.mean(), ci95_pp=h,
                         min_pp=x.min(), max_pp=x.max(), tapra_wins=int((x > 0).sum()),
                         p_wilcoxon=stats.wilcoxon(x).pvalue if (x != 0).any() else 1.0))
r = pd.DataFrame(rows); r.to_csv("results/rev2_paired.csv", index=False)
print(r.round(2).to_string())
