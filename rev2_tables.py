"""Rev. 2: merge the rerun results and write Table 4 and the paired table (Student-t CIs, 9 df).
  python rev2_tables.py   -> results/batch_rev2.csv, results/tables/results.tex, results/tables/paired.tex,
                             results/rev2_paired.csv, and the numbers quoted in the text on stdout."""
import os
import numpy as np, pandas as pd
from scipy import stats

R = "results"
FIX = ["TAPRA", "TAPRA uncertainty-aware (5 m)", "Ablation: slot-start objective",
       "Ablation: no remembered spots", "Ablation: no trigger"]
if os.path.exists(f"{R}/main_results.csv"):   # released per-seed results (all methods, ablations, DQN)
    d = pd.read_csv(f"{R}/main_results.csv")
else:                                          # merge the raw runs
    old = pd.read_csv(f"{R}/batch_final.csv")
    new = pd.read_csv(f"{R}/batch_fix.csv")
    parts = [old[~old.method.isin(FIX)], new]
    if os.path.exists(f"{R}/rev2_ablation.csv") and os.path.getsize(f"{R}/rev2_ablation.csv") > 200:
        parts.append(pd.read_csv(f"{R}/rev2_ablation.csv"))
    if os.path.exists(f"{R}/rev2_dqn.csv") and os.path.getsize(f"{R}/rev2_dqn.csv") > 200:
        q = pd.read_csv(f"{R}/rev2_dqn.csv")
        parts.append(q[[c for c in old.columns if c in q.columns]])
    d = pd.concat(parts, ignore_index=True)
d.to_csv(f"{R}/batch_rev2.csv", index=False)

SC = [("suburban", 500, "homogeneous"), ("suburban", 500, "heterogeneous"),
      ("urban", 500, "homogeneous"), ("urban", 500, "heterogeneous"),
      ("urban", 1000, "homogeneous"), ("urban", 1000, "heterogeneous")]
COMP = ["VTOPA rule", "RLTOPA", "Traffic centroid", "Fixed rooftop AP"]


def tci(x):
    x = np.asarray(x, float)
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))


def cell(m, s):
    e, a, t = s
    x = d[(d.env == e) & (d.area == a) & (d.traffic == t) & (d.method == m)].realised_share * 100
    return (x.mean(), tci(x), len(x)) if len(x) else (np.nan, np.nan, 0)


ROWS = [("TAPRA", r"\textbf{TAPRA}"), ("Full search every slot", "Full search every slot"),
        ("VTOPA rule", r"VTOPA rule~\cite{shafafi2025autonomousvisionaideduavpositioning}"),
        ("RLTOPA", r"RLTOPA~\cite{10792915rltopa}"),
        ("DQN (TAPRA objective)", r"DQN, TAPRA objective"),
        ("Traffic centroid", "Traffic centroid"), ("Fixed rooftop AP", "Fixed rooftop AP"), None,
        ("Ablation: slot-start objective", "Slot-start objective"),
        ("Ablation: no motion prediction", "No motion prediction"),
        ("Ablation: no UAV travel", "No UAV travel"),
        ("Ablation: no remembered spots", "No remembered spots"),
        ("Ablation: no trigger", "No trigger")]
out = []
for r in ROWS:
    if r is None:
        out.append(r"\hline" + "\n" + r"\multicolumn{7}{l}{\textit{Ablations of TAPRA}} \\")
        continue
    m, lab = r
    cs = []
    for s in SC:
        mu, h, n = cell(m, s)
        cs.append("--" if n == 0 else (rf"\textbf{{{mu:.1f} $\pm$ {h:.1f}}}" if m == "TAPRA" else f"{mu:.1f} $\\pm$ {h:.1f}"))
    out.append(lab + " & " + " & ".join(cs) + r" \\")
ev = []
for m in ["TAPRA", "Full search every slot"]:
    ev.append(m + " & " + " & ".join(
        f"{d[(d.env == e) & (d.area == a) & (d.traffic == t) & (d.method == m)].evals_per_slot.mean():.0f}"
        for e, a, t in SC) + r" \\")
os.makedirs(f"{R}/tables", exist_ok=True)
open(f"{R}/tables/results.tex", "w").write("\n".join(out) + "\n% evaluations\n" + "\n".join(ev) + "\n")

# paired differences TAPRA - method, by scenario
rows = []
for s in SC:
    e, a, t = s
    g = d[(d.env == e) & (d.area == a) & (d.traffic == t)]
    p = g.pivot(index="seed", columns="method", values="realised_share") * 100
    for m in p.columns:
        if m == "TAPRA":
            continue
        x = (p["TAPRA"] - p[m]).dropna()
        if len(x) < 2:
            continue
        rows.append(dict(env=e, area=a, traffic=t, method=m, n=len(x), mean_pp=x.mean(), ci95_pp=tci(x),
                         min_pp=x.min(), max_pp=x.max(), tapra_wins=int((x > 0).sum()),
                         p_wilcoxon=stats.wilcoxon(x).pvalue if (x != 0).any() else 1.0))
P = pd.DataFrame(rows)
P.to_csv(f"{R}/rev2_paired.csv", index=False)

# strongest competing positioning method per scenario (full search and ablations excluded)
lines = []
print("\n== strongest competitor and full search, by scenario")
for s in SC:
    e, a, t = s
    q = P[(P.env == e) & (P.area == a) & (P.traffic == t)].set_index("method")
    comp = [m for m in COMP + ["DQN (TAPRA objective)"] if m in q.index]
    best = q.loc[comp].mean_pp.idxmin()
    b, f = q.loc[best], q.loc["Full search every slot"]
    print(f"{e} {a} {t}: best={best} {b.mean_pp:+.2f} +- {b.ci95_pp:.2f} wins {b.tapra_wins}/{b.n} | "
          f"full {f.mean_pp:+.2f} +- {f.ci95_pp:.2f} wins {f.tapra_wins}")
    lab = {"suburban": "Suburban", "urban": "Urban"}[e] + f", {a if a < 1000 else '1~km'}" \
        .replace("500", "500~m") + (", homo" if t == "homogeneous" else ", hetero")
    lines.append(f"{lab} & {best.replace(' rule', '')} & ${b.mean_pp:+.1f} \\pm {b.ci95_pp:.1f}$ & "
                 f"{b.min_pp:+.1f} to {b.max_pp:+.1f} & {b.tapra_wins}/{b.n} & "
                 f"${f.mean_pp:+.1f} \\pm {f.ci95_pp:.1f}$ & {f.tapra_wins}/{f.n} \\\\")
open(f"{R}/tables/paired.tex", "w").write("\n".join(lines) + "\n")
print("\n== all pairs")
print(P.round(2).to_string())
print("\n== table rows")
print(open(f"{R}/tables/results.tex").read())
