import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from plots import SCEN
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
d=pd.read_csv("results/main_results.csv")
M=[("TAPRA","#1b6ca8","TAPRA (slot average)"),("Ablation: no UAV travel","#16a085","No UAV travel"),
   ("Ablation: no motion prediction","#8e44ad","No motion prediction"),("Ablation: slot-start objective","#e67e22","Slot start")]
lab=["Suburban 500 m\nhomogeneous","Suburban 500 m\nheterogeneous","Urban 500 m\nhomogeneous","Urban 500 m\nheterogeneous","Urban 1 km\nhomogeneous","Urban 1 km\nheterogeneous"]
fig,ax=plt.subplots(figsize=(3.5,4.4))
off=[0.3,0.1,-0.1,-0.3]
for r,(env,area,tr,_) in enumerate(SCEN):
    y0=len(SCEN)-1-r
    if r%2==0: ax.axhspan(y0-0.5,y0+0.5,color="#f2f4f4",zorder=0)
    for j,(m,c,_) in enumerate(M):
        x=d[(d.env==env)&(d.area==area)&(d.traffic==tr)&(d.method==m)]
        pl,re=x.planned_share.mean()*100,x.realised_share.mean()*100; y=y0+off[j]
        ax.plot([re,pl],[y,y],color=c,lw=1.4,zorder=2)
        ax.plot(pl,y,"o",ms=3.8,mfc="white",mec=c,mew=1.0,zorder=3)
        ax.plot(re,y,"o",ms=3.8,color=c,zorder=4)
        ax.text(pl+1.2,y,f"{pl-re:.1f}",va="center",ha="left",fontsize=5.6,color=c)
ax.set_yticks(range(len(SCEN))); ax.set_yticklabels(lab[::-1],fontsize=6,linespacing=1.0)
ax.set_ylim(-0.5,len(SCEN)-0.5); ax.set_xlim(10,100)
ax.set_xlabel("Share of the offered demand (%)")
ax.grid(axis="x",alpha=0.3); ax.grid(axis="y",visible=False)
for s in ("top","right"): ax.spines[s].set_visible(False)
h=[plt.Line2D([],[],color=c,lw=1.4,label=l) for _,c,l in M]+\
  [plt.Line2D([],[],marker="o",ls="",mfc="white",mec="k",ms=3.8,label="planned share"),plt.Line2D([],[],marker="o",ls="",color="k",ms=3.8,label="served share")]
ax.legend(handles=h,loc="lower right",fontsize=5.8,frameon=True,framealpha=0.9,edgecolor="none",ncol=1)
fig.tight_layout(); os.makedirs("figures", exist_ok=True); fig.savefig("figures/fig_planned_realised.pdf",bbox_inches="tight")
