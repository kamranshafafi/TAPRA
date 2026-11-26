"""
Figures for Section VI from results/batch.csv (and one example city).

  python plots.py --batch results/batch.csv --out figures/
Writes:
  fig_served.pdf          realised served share per method and scenario (95% CI)
  fig_planned_realised.pdf planned vs realised share: slot-average vs slot-start objective
  fig_city_example.pdf    one urban city: buildings, node tracks, TAPRA trajectory
  fig_robustness.pdf      loss of served share under imperfect information (needs --robustness)
  fig_ns3.pdf             served share in ns-3 vs. the model, one point per run (needs --ns3)
  fig_runtime.pdf         time per slot vs. number of UEs and per city type (needs --runtime)
  fig_energy.pdf          propulsion energy and served data per kJ (needs --energy)
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

plt.rcParams.update({"font.size": 8, "font.family": "serif", "axes.grid": True,
                     "grid.alpha": 0.3, "figure.dpi": 150})
SCEN = [("suburban", 500, "homogeneous", "Suburban 500 m\nhomogeneous"),
        ("suburban", 500, "heterogeneous", "Suburban 500 m\nheterogeneous"),
        ("urban", 500, "homogeneous", "Urban 500 m\nhomogeneous"),
        ("urban", 500, "heterogeneous", "Urban 500 m\nheterogeneous"),
        ("urban", 1000, "homogeneous", "Urban 1 km\nhomogeneous"),
        ("urban", 1000, "heterogeneous", "Urban 1 km\nheterogeneous")]
METHODS = [("TAPRA", "#1b6ca8"), ("Full search every slot", "#7fb3d5"),
           ("VTOPA rule", "#e67e22"), ("RLTOPA", "#c0392b"),
           ("Traffic centroid", "#7f8c8d"), ("Fixed rooftop AP", "#2c3e50")]


def ci95(x):
    x = np.asarray(x, float)
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0


def fig_served(d, out):
    fig, ax = plt.subplots(figsize=(7.16, 2.4))
    w = 0.13
    for j, (m, col) in enumerate(METHODS):
        mu, ci = [], []
        for env, area, tr, _ in SCEN:
            x = d[(d.env == env) & (d.area == area) & (d.traffic == tr) & (d.method == m)].realised_share * 100
            mu.append(x.mean()); ci.append(ci95(x))
        pos = np.arange(len(SCEN)) + (j - (len(METHODS) - 1) / 2) * w
        ax.bar(pos, mu, w, yerr=ci, capsize=1.5, color=col, label=m.replace(" every slot", ""),
               error_kw={"lw": 0.6})
    ax.set_xticks(range(len(SCEN)))
    ax.set_xticklabels([s[3] for s in SCEN])
    ax.set_ylabel("Served demand (%)")
    ax.set_ylim(0, 100)
    ax.legend(ncol=6, loc="upper center", bbox_to_anchor=(0.5, 1.18), frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_served.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_planned_realised(d, out):
    fig, ax = plt.subplots(figsize=(3.5, 2.5))
    labels = []
    for j, (m, col, lab) in enumerate([("TAPRA", "#1b6ca8", "Slot-average (TAPRA)"),
                                       ("Ablation: slot-start objective", "#e67e22", "Slot-start")]):
        pl, re = [], []
        for env, area, tr, _ in SCEN:
            x = d[(d.env == env) & (d.area == area) & (d.traffic == tr) & (d.method == m)]
            pl.append(x.planned_share.mean() * 100); re.append(x.realised_share.mean() * 100)
        pos = np.arange(len(SCEN)) + (j - 0.5) * 0.36
        ax.bar(pos, pl, 0.34, color=col, alpha=0.35, edgecolor=col, label=f"{lab}: planned")
        ax.bar(pos, re, 0.34, color=col, label=f"{lab}: realized")
    ax.set_xticks(range(len(SCEN)))
    ax.set_xticklabels(["S-ho", "S-he", "U-ho", "U-he", "U1k-ho", "U1k-he"])
    ax.set_ylabel("Served demand (%)")
    ax.set_ylim(0, 100)
    ax.legend(fontsize=6.5, frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_planned_realised.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_city(out, seed=1):
    from link_model import LinkModel
    from optimizer import Config, run_trajectory
    from scenario_gen import generate
    from matplotlib.patches import Rectangle
    sc = generate("urban", 500, 20, "heterogeneous", 50, 5.0, seed=seed)
    res = run_trajectory(sc, LinkModel(env="urban"), Config(), "adaptive")
    fig, ax = plt.subplots(figsize=(3.5, 3.3))
    hmax = max(b["height"] for b in sc["Buildings"])
    for b in sc["Buildings"]:
        xs = [p[0] for p in b["corners"]] if "corners" in b else None
        if xs is None:
            continue
        ys = [p[1] for p in b["corners"]]
        ax.add_patch(Rectangle((min(xs), min(ys)), max(xs) - min(xs), max(ys) - min(ys),
                               color=plt.cm.Greys(0.25 + 0.6 * b["height"] / hmax), lw=0))
    for u in sc["UENodes"]:
        p = np.array(u["position"])
        ax.plot(p[:, 0], p[:, 1], lw=0.6, color="#2e86c1" if u["type"] == "UE" else "#27ae60", alpha=0.8)
        ax.plot(p[0, 0], p[0, 1], "o", ms=2, color="k")
    q = np.array([r.q for r in res])
    # one UAV: its positions at the 50 slots, joined in time order
    ax.plot(q[:, 0], q[:, 1], "-", color="#c0392b", lw=0.9, zorder=4, label="UAV trajectory (one UAV)")
    sca = ax.scatter(q[:, 0], q[:, 1], c=q[:, 2], cmap="plasma", s=5, zorder=5, vmin=30, vmax=100)
    ax.plot(q[0, 0], q[0, 1], marker="*", ms=9, color="#c0392b", mec="k", mew=0.5, zorder=6,
            ls="none", label="UAV start ($k=1$)")
    ax.plot(q[-1, 0], q[-1, 1], marker="s", ms=5, color="#c0392b", mec="k", mew=0.5, zorder=6,
            ls="none", label="UAV end ($k=50$)")
    ax.legend(fontsize=6, loc="lower left", framealpha=0.9)
    fig.colorbar(sca, ax=ax, label="UAV altitude (m)", shrink=0.8)
    ax.set_xlim(-250, 250); ax.set_ylim(-250, 250); ax.set_aspect("equal")
    ax.set_xlabel("x (m)"); ax.set_ylabel("y (m)")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_city_example.pdf"), bbox_inches="tight")
    plt.close(fig)


COND = [("gps_2m", "GPS\n2 m"), ("gps_5m", "GPS\n5 m"), ("gps_10m", "GPS\n10 m"),
        ("stale", "Stale\nreports"), ("map_heights", "Wrong\nheights"),
        ("map_missing", "Missing\nbuildings"), ("shadowing", "Shadow-\ning"),
        ("itu_p1411", "P.1411\nchannel")]
RMETH = [("TAPRA", "#1b6ca8", "TAPRA"), ("TAPRA uncertainty-aware", "#27ae60", "TAPRA uncertainty-aware"),
         ("VTOPA rule", "#e67e22", "VTOPA rule")]


def fig_robustness(r, out):
    """Change of the served share vs. perfect information, paired by city, pooled over
    the three robustness scenarios (30 cities per bar)."""
    k = ["env", "traffic", "seed", "condition", "method"]
    r = r.drop_duplicates(k, keep="last")          # restarted runs append duplicate rows
    base = r[r.condition == "perfect"][["env", "traffic", "seed", "method", "realised_share"]]
    m = r.merge(base.rename(columns={"realised_share": "base"}),
                on=["env", "traffic", "seed", "method"])
    m["delta"] = (m.realised_share - m.base) * 100
    fig, ax = plt.subplots(figsize=(7.16, 2.3))
    w = 0.26
    for j, (meth, col, lab) in enumerate(RMETH):
        mu, ci = [], []
        for c, _ in COND:
            x = m[(m.condition == c) & (m.method == meth)].delta
            mu.append(x.mean()); ci.append(ci95(x))
        pos = np.arange(len(COND)) + (j - 1) * w
        ax.bar(pos, mu, w, yerr=ci, capsize=1.5, color=col, label=lab, error_kw={"lw": 0.6})
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(COND)))
    ax.set_xticklabels([c[1] for c in COND])
    ax.set_ylabel("Change in served\nshare (points)")
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.2), frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_robustness.pdf"), bbox_inches="tight")
    plt.close(fig)


def fig_ns3(v, out):
    v = v.copy()
    v["method"] = v.run.str.split("_").str[-1]
    v["scen"] = v.run.str.extract(r"^(\w+?)500_20n_(\w{4})").agg(" ".join, axis=1)
    fig, ax = plt.subplots(figsize=(3.5, 3.0))
    style = {"tapra": ("#1b6ca8", "o", "TAPRA"), "full": ("#7fb3d5", "s", "Full search"),
             "vtopa": ("#e67e22", "^", "VTOPA rule")}
    for m, (col, mk, lab) in style.items():
        x = v[v.method == m]
        ax.scatter(x.python_realised_share * 100, x.ns3_share * 100, s=18, c=col, marker=mk,
                   label=lab, edgecolor="k", linewidth=0.3, zorder=3)
    lo, hi = 30, 90
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.7, label="ns-3 = model")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("Served share, model (%)"); ax.set_ylabel("Served share, ns-3 (%)")
    ax.legend(fontsize=6.5, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_ns3.pdf"), bbox_inches="tight")
    plt.close(fig)


RT = [("TAPRA", "#1b6ca8", "o", "TAPRA"),
      ("Full search every slot", "#7fb3d5", "s", "Full search every slot"),
      ("TAPRA uncertainty-aware (2 samples)", "#27ae60", "^", "Uncertainty-aware, $M=2$"),
      ("TAPRA uncertainty-aware (3 samples)", "#145a32", "v", "Uncertainty-aware, $M=3$")]


def fig_runtime(r, out):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.4), gridspec_kw={"width_ratios": [1, 1.2]})
    n = r[r.sweep == "nodes"]
    for m, col, mk, lab in RT:
        g = n[n.method == m].groupby("nodes").per_slot_s
        a1.errorbar(g.mean().index, g.mean(), yerr=[ci95(x) for _, x in g], color=col, marker=mk,
                    ms=4, lw=1, capsize=2, label=lab)
    a1.set_xlabel("Number of UEs $N$ (urban, 500 m)")
    a1.set_ylabel("Time per slot (s)")
    cases = [(("density", "suburban", 500, 20), "Suburban\n500 m, 20 UEs"),
             (("density", "urban", 500, 20), "Urban\n500 m, 20 UEs"),
             (("density", "dense_urban", 500, 20), "Dense urban\n500 m, 20 UEs"),
             (("area", "urban", 1000, 50), "Urban\n1 km, 50 UEs")]
    w = 0.2
    for j, (m, col, mk, lab) in enumerate(RT):
        mu, ci = [], []
        for (sw, env, area, nn), _ in cases:
            x = r[(r.sweep == sw) & (r.env == env) & (r.area == area) & (r.nodes == nn) & (r.method == m)].per_slot_s
            mu.append(x.mean()); ci.append(ci95(x))
        a2.bar(np.arange(len(cases)) + (j - 1.5) * w, mu, w, yerr=ci, color=col, capsize=1.5,
               error_kw={"lw": 0.6})
    a2.set_xticks(range(len(cases))); a2.set_xticklabels([c[1] for c in cases], fontsize=7)
    for ax in (a1, a2):
        ax.axhline(5, color="#c0392b", ls="--", lw=0.8)
        ax.set_ylim(0, 5.6)
    a1.text(52, 5.12, "slot duration $\\tau$ = 5 s", color="#c0392b", fontsize=7)
    a1.legend(fontsize=6.5, frameon=False, loc="upper left", bbox_to_anchor=(0, 0.93))
    a1.set_title("(a)", fontsize=8); a2.set_title("(b)", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_runtime.pdf"), bbox_inches="tight")
    plt.close(fig)


EN = [("TAPRA", "#1b6ca8"), ("Full search every slot", "#7fb3d5"), ("VTOPA rule", "#e67e22"),
      ("Traffic centroid", "#7f8c8d"), ("Fixed rooftop AP", "#2c3e50")]
ESC = [("suburban", "homogeneous", "Suburban\nhomogeneous"), ("urban", "homogeneous", "Urban\nhomogeneous"),
       ("urban", "heterogeneous", "Urban\nheterogeneous")]


def fig_energy(e, out):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(3.5, 3.9), sharex=True)
    w = 0.16
    for j, (m, col) in enumerate(EN):
        lab = "Hovering (fixed position)" if m == "Fixed rooftop AP" else m.replace(" every slot", "")
        for ax, colname in ((a1, "energy_kj"), (a2, "planned_mbit_per_kj")):
            mu, ci = [], []
            for env, tr, _ in ESC:
                x = e[(e.env == env) & (e.traffic == tr) & (e.method == m)][colname]
                mu.append(x.mean()); ci.append(ci95(x))
            ax.bar(np.arange(len(ESC)) + (j - 2) * w, mu, w, yerr=ci, color=col, capsize=1.2,
                   error_kw={"lw": 0.5}, label=lab if ax is a1 else None)
    a1.set_ylim(35, 50)
    a1.set_ylabel("Energy over 250 s (kJ)")
    a2.set_ylabel("Served data per energy\n(Mbit/kJ)")
    a2.set_xticks(range(len(ESC))); a2.set_xticklabels([c[2] for c in ESC], fontsize=7)
    a1.legend(fontsize=6, ncol=2, frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_energy.pdf"), bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", default="results/batch.csv")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--no-city", action="store_true")
    ap.add_argument("--robustness", default=None, help="results/robustness.csv")
    ap.add_argument("--ns3", default=None, help="results/ns3_validation.csv")
    ap.add_argument("--runtime", default=None, help="results/runtime.csv")
    ap.add_argument("--energy", default=None, help="results/energy.csv")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    d = pd.read_csv(a.batch)
    fig_served(d, a.out)
    fig_planned_realised(d, a.out)
    if not a.no_city:
        fig_city(a.out)
    if a.runtime:
        fig_runtime(pd.read_csv(a.runtime), a.out)
    if a.energy:
        fig_energy(pd.read_csv(a.energy), a.out)
    if a.ns3:
        fig_ns3(pd.read_csv(a.ns3), a.out)
    if a.robustness:
        fig_robustness(pd.read_csv(a.robustness), a.out)
    print("figures written to", a.out)
