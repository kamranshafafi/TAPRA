import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from plots import ci95
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
r=pd.read_csv("results/robustness.csv")
r=r.drop_duplicates(["env","traffic","seed","condition","method"],keep="last")
base=r[r.condition=="perfect"][["env","traffic","seed","method","realised_share"]]
m=r.merge(base.rename(columns={"realised_share":"base"}),on=["env","traffic","seed","method"])
m["delta"]=(m.realised_share-m.base)*100
COND=[("gps_2m","GPS error 2 m"),("gps_5m","GPS error 5 m"),("gps_10m","GPS error 10 m"),("stale","Reports one slot old"),
      ("map_heights","Wrong building heights"),("map_missing","10% of buildings missing"),("shadowing","Shadowing"),("itu_p1411","ITU-R P.1411 channel")]
RM=[("TAPRA","TAPRA"),("TAPRA uncertainty-aware","Uncertainty-\naware"),("VTOPA rule","VTOPA rule")]
M=np.zeros((len(COND),len(RM))); C=np.zeros_like(M)
for i,(c,_) in enumerate(COND):
    for j,(meth,_) in enumerate(RM):
        x=m[(m.condition==c)&(m.method==meth)].delta; M[i,j]=x.mean(); C[i,j]=ci95(x)
fig,ax=plt.subplots(figsize=(3.5,3.7))
im=ax.imshow(M,cmap="RdBu",norm=TwoSlopeNorm(vmin=-16,vcenter=0,vmax=4),aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        v=M[i,j]; col="white" if v<-8 else "black"
        ax.text(j,i-0.12,f"{v:+.1f}".replace("-","\u2212"),ha="center",va="center",fontsize=7,fontweight="bold",color=col)
        ax.text(j,i+0.22,f"\u00b1{C[i,j]:.1f}",ha="center",va="center",fontsize=5.5,color=col)
ax.set_xticks(range(len(RM))); ax.set_xticklabels([l for _,l in RM],fontsize=6,rotation=90,va="bottom",linespacing=1.0); ax.xaxis.tick_top()
ax.set_yticks(range(len(COND))); ax.set_yticklabels([l for _,l in COND],fontsize=6.3)
ax.set_xticks(np.arange(-.5,len(RM),1),minor=True); ax.set_yticks(np.arange(-.5,len(COND),1),minor=True)
ax.grid(False); ax.grid(which="minor",color="white",lw=1.5); ax.tick_params(which="minor",length=0); ax.tick_params(length=0)
for s in ax.spines.values(): s.set_visible(False)
cb=fig.colorbar(im,ax=ax,fraction=0.05,pad=0.03); cb.ax.tick_params(labelsize=6); cb.set_ticks([-15,-10,-5,0,4])
cb.set_label("Change in served share\n(percentage points)",fontsize=6.3)
fig.tight_layout(); os.makedirs("figures", exist_ok=True); fig.savefig("figures/fig_robustness_heatmap.pdf",bbox_inches="tight")
