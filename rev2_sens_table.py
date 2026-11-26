"""Rev. 2: sensitivity table (urban 500 m) from results/rev2_sens.csv -> results/tables/sensitivity.tex."""
import numpy as np, pandas as pd
from scipy import stats

d = pd.read_csv("results/rev2_sens.csv")
d["value"] = d.value.astype(float)


def tci(x):
    x = np.asarray(x, float)
    return stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else np.nan


def sg(x):
    r = f"{x:+.1f}"
    return "0.0" if r in ("+0.0", "-0.0") else r


def get(sw, v, tr, m):
    x = d[(d.sweep == sw) & (d.value == v) & (d.traffic == tr) & (d.method == m)]
    return x.set_index("seed")


ROWS = [("base", 1.0, "Base (Table~\\ref{tab:ns3_config})"),
        ("speed", 0.5, "UE speed $\\times 0.5$"), ("speed", 2.0, "UE speed $\\times 2$"), ("speed", 4.0, "UE speed $\\times 4$"),
        ("tau", 2.0, "$\\tau = 2$~s"), ("tau", 10.0, "$\\tau = 10$~s"),
        ("J", 1, "$J = 1$"), ("J", 5, "$J = 5$"),
        ("mu", 0.0, "$\\mu = 0$"), ("mu", 0.4, "$\\mu = 0.4$"), ("mu", 1.0, "$\\mu = 1$"),
        ("omega", 0.0, "$\\omega = 0$"), ("omega", 0.6, "$\\omega = 0.6$"),
        ("Delta", 0.02, "$\\Delta = 2$ points"), ("Delta", 0.10, "$\\Delta = 10$ points"),
        ("K_f", 5, "$K_f = 5$"), ("K_f", 20, "$K_f = 20$"),
        ("Q_xy", 11, "$Q_{xy} = 11$"), ("Q_xy", 31, "$Q_{xy} = 31$"),
        ("E_a", 30, "$E_a = 30$"), ("E_a", 120, "$E_a = 120$")]

lines, rep = [], []
for sw, v, lab in ROWS:
    cells = []
    for tr in ("homogeneous", "heterogeneous"):
        t = get(sw, v, tr, "TAPRA")
        if len(t) == 0:
            cells += ["--"] * (4 if tr == "homogeneous" else 2)
            continue
        b = get("base", 1.0, tr, "TAPRA").realised_share.reindex(t.index)
        served = t.realised_share.mean() * 100
        dif = (t.realised_share - b) * 100
        vt = get(sw, v, tr, "VTOPA rule")
        vs = (t.realised_share - vt.realised_share.reindex(t.index)).dropna() * 100
        ss = get(sw, v, tr, "Slot-start objective")
        so = (t.realised_share - ss.realised_share.reindex(t.index)).dropna() * 100
        gap = (t.planned_share - t.realised_share).mean() * 100
        if tr == "homogeneous":
            cells.append(f"{served:.1f}" + ("" if sw == "base" else f" ({sg(dif.mean())})"))
            cells.append(sg(vs.mean()) if len(vs) else "--")
            cells.append(sg(so.mean()) if len(so) else "--")
            cells.append(f"{t.evals_per_slot.mean():.0f}")
        else:
            cells.append(f"{served:.1f}" + ("" if sw == "base" else f" ({sg(dif.mean())})"))
            cells.append(sg(vs.mean()) if len(vs) else "--")
        rep.append(dict(sweep=sw, value=v, traffic=tr, n=len(t), served=served, d_base=dif.mean(), ci_base=tci(dif),
                        vs_vtopa=vs.mean() if len(vs) else np.nan, vs_vtopa_ci=tci(vs) if len(vs) > 1 else np.nan,
                        vtopa_wins=int((vs > 0).sum()), vs_slotstart=so.mean() if len(so) else np.nan,
                        vs_slotstart_ci=tci(so) if len(so) > 1 else np.nan, overpromise=gap,
                        evals=t.evals_per_slot.mean(), sec=t.sec_per_slot.mean(), full=t.full_searches.mean()))
    lines.append(lab + " & " + " & ".join(cells).replace("-", "$-$").replace("$-$$-$", "--") + r" \\")
open("results/tables/sensitivity.tex", "w").write("\n".join(lines) + "\n")
R = pd.DataFrame(rep)
R.to_csv("results/rev2_sens_summary.csv", index=False)
pd.set_option("display.width", 250)
print(R.round(2).to_string())
print("\n".join(lines))
