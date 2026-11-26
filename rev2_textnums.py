"""Rev. 2: admission table and the numbers quoted in the admission text (from results/rev2_admission.csv)."""
import pandas as pd, numpy as np
d = pd.read_csv("results/rev2_admission.csv")
M = d[d.nodes.isin([20, 50])]
SC = [("suburban", 500, "homogeneous", "Suburban, homo"), ("suburban", 500, "heterogeneous", "Suburban, hetero"),
      ("urban", 500, "homogeneous", "Urban 500~m, homo"), ("urban", 500, "heterogeneous", "Urban 500~m, hetero"),
      ("urban", 1000, "homogeneous", "Urban 1~km, homo"), ("urban", 1000, "heterogeneous", "Urban 1~km, hetero")]
POL = ["TAPRA", "All eligible", "SNR>=8 dB", "SNR>=11 dB"]
g = M.groupby(["env", "area", "traffic", "policy"])[["served_share", "outage", "jain_ue", "ue_never"]].mean()
lines = []
for e, a, t, lab in SC:
    cs = [f"{g.loc[(e,a,t,p)].served_share*100:.1f} / {g.loc[(e,a,t,p)].outage*100:.1f} / {g.loc[(e,a,t,p)].jain_ue:.2f}" for p in POL]
    lines.append(lab + " & " + " & ".join(cs) + r" \\")
open("results/tables/admission.tex", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
w = M.pivot_table(index=["env", "area", "traffic", "seed"], columns="policy", values="served_share") * 100
print("\nTAPRA - all eligible (pp) by scenario: mean/min/max, wins")
for (e, a, t), x in (w["TAPRA"] - w["All eligible"]).groupby(level=[0, 1, 2]):
    print(e, a, t, round(x.mean(), 2), round(x.min(), 2), round(x.max(), 2), (x > 0).sum())
best_thr = w[[c for c in w.columns if c.startswith("SNR")]].groupby(level=[0, 1, 2]).mean()
print("\nmean served by SNR threshold\n", best_thr.round(1))
for p in ["SNR>=8 dB", "SNR>=11 dB"]:
    x = (w["TAPRA"] - w[p]).groupby(level=[0, 1, 2]).mean()
    print("TAPRA -", p, x.round(2).to_dict())
print("\nnever served share of UEs\n", g.ue_never.unstack().round(3)[POL])
gap = d[d.policy == "_gap"]
print("\nbest subset: slots equal min/max", gap.slots_equal.min(), gap.slots_equal.max(),
      "objective gap max %", ((gap.planner_best - gap.planner_tapra) / gap.planner_best * 100).max())
S = d[d.nodes.isin([8, 10, 12])].pivot_table(index=["nodes", "traffic", "seed"], columns="policy", values="served_share") * 100
print("served TAPRA - best subset (pp): max abs", (S["TAPRA"] - S["Best subset"]).abs().max())

# ---- channel and elevation (results/rev2_channel.csv, rev2_elevation.csv)
c = pd.read_csv("results/rev2_channel.csv")
print("\nchannel methods:", c.method.unique(), c.channel.unique())
CH = list(c.channel.unique())
LAB = {CH[0]: "A2G (planning model)", CH[1]: "A2G + shadowing", CH[2]: "ITU-R P.1411"}
lines = []; wins = {}
for ch in CH:
    cs = []
    for e, a, t, _ in SC:
        q = c[(c.env == e) & (c.area == a) & (c.traffic == t) & (c.channel == ch)].pivot(index="seed", columns="method", values="served_share") * 100
        comp = [m for m in q.columns if m not in ("TAPRA", "Full search every slot")]
        diffs = {m: (q["TAPRA"] - q[m]) for m in comp}
        best = min(diffs, key=lambda m: diffs[m].mean())
        wins[ch] = wins.get(ch, 0) + int((q["TAPRA"] > q[comp].max(axis=1)).sum())
        fs = (q["TAPRA"] - q["Full search every slot"]).mean()
        cs.append(f"{q['TAPRA'].mean():.1f} ({diffs[best].mean():+.1f}, {fs:+.1f})".replace("+-", "-"))
        order = q.mean().sort_values(ascending=False).index.tolist()
        print(ch, e, a, t, "best comp", best, "ranking", order)
    lines.append(LAB[ch] + " & " + " & ".join(cs) + r" \\")
open("results/tables/channel.tex", "w").write("\n".join(lines) + "\n")
print("\n".join(lines)); print("cities where TAPRA beats every competitor:", wins)
E = pd.read_csv("results/rev2_elevation.csv")
E = E[E.method == "TAPRA"]
BINS = ["0-15", "15-30", "30-45", "45-60", "60-90"]
BL = {"0-15": r"0--15$^\circ$", "15-30": r"15--30$^\circ$", "30-45": r"30--45$^\circ$", "45-60": r"45--60$^\circ$", "60-90": r"60--90$^\circ$"}
lines = []
for b in BINS:
    cs = []
    for e, a in [("suburban", 500), ("urban", 500), ("urban", 1000)]:
        x = E[(E.env == e) & (E.area == a)]
        tot = x[x.channel == CH[0]].link_seconds.sum()
        y0 = x[(x.channel == CH[0]) & (x.elev_bin == b)]
        y2 = x[(x.channel == CH[2]) & (x.elev_bin == b)]
        wavg = lambda y, col: (y[col] * y.link_seconds).sum() / y.link_seconds.sum()
        cs += [f"{y0.link_seconds.sum() / tot * 100:.0f}", f"{wavg(y0, 'los_prob') * 100:.0f}",
               f"{wavg(y0, 'satisfaction') * 100:.0f}", f"{wavg(y2, 'satisfaction') * 100:.0f}"]
    lines.append(BL[b] + " & " + " & ".join(cs) + r" \\")
open("results/tables/elevation.tex", "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
