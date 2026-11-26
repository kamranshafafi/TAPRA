import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from plots import SCEN, ci95
plt.rcParams.update({"font.size": 8, "font.family": "serif"})
b=pd.read_csv("results/main_results.csv")
w=b.pivot_table(index=["env","area","traffic","seed"],columns="method",values="realised_share")*100
TBLUE="#1b6ca8"
BL=[("Full search every slot","#7fb3d5","D","Full search"),("VTOPA rule","#e67e22","o","VTOPA rule"),
    ("RLTOPA","#c0392b","s","RLTOPA"),("DQN (TAPRA objective)","#8e44ad","P","DQN, TAPRA objective"),("Traffic centroid","#7f8c8d","^","Traffic centroid"),
    ("Fixed rooftop AP","#2c3e50","v","Fixed rooftop AP")]
fig,ax=plt.subplots(figsize=(7.16,4.6))
off=np.linspace(0.34,-0.34,len(BL)+1)
rng=np.random.default_rng(0)
for r,(env,area,tr,lab) in enumerate(SCEN):
    g=w.loc[(env,area,tr)]; y0=len(SCEN)-1-r
    if r%2==0: ax.axhspan(y0-0.5,y0+0.5,color="#f2f4f4",zorder=0)
    ax.plot(0,y0+off[0],marker="*",ms=8,color=TBLUE,mec="white",mew=0.5,zorder=6)
    ax.text(8.5,y0,f"{g['TAPRA'].mean():.1f}%",fontsize=7.5,color=TBLUE,va="center",ha="right",fontweight="bold",zorder=6)
    rel={m:(g[m]-g["TAPRA"]) for m,_,_,_ in BL}
    strongest=max([m for m,_,_,_ in BL if m!="Full search every slot"],key=lambda m:rel[m].mean())
    for (m,c,mk,_),dy in zip(BL,off[1:]):
        d=rel[m]; y=y0+dy; mu=d.mean(); h=ci95(d)
        ax.scatter(d,np.full(len(d),y)+rng.uniform(-0.03,0.03,len(d)),s=4,color=c,alpha=0.25,lw=0,zorder=2)
        filled=(m==strongest) or m=="Full search every slot"
        ax.errorbar(mu,y,xerr=h,fmt=mk,ms=4.2,color=c,mfc=c if filled else "white",mew=1.0,elinewidth=1.0,capsize=1.5,zorder=4)
        txt=f"{mu:+.1f}".replace("-","\u2212")
        if mu>=0: ax.text(mu+h+0.6,y,txt,va="center",ha="left",fontsize=6.2,color=c,fontweight="bold" if m==strongest else "normal",zorder=5)
        else:     ax.text(mu-h-0.6,y,txt,va="center",ha="right",fontsize=6.2,color=c,fontweight="bold" if m==strongest else "normal",zorder=5)
ax.axvline(0,color=TBLUE,lw=1.6,zorder=1)
ax.text(0,len(SCEN)-0.45,"TAPRA",color=TBLUE,fontsize=8,fontweight="bold",ha="center",va="bottom")
ax.text(8.5,len(SCEN)-0.45,"TAPRA\nserves",color=TBLUE,fontsize=7,ha="right",va="bottom")
ax.annotate("",xy=(-59,-0.36),xytext=(-45,-0.36),arrowprops=dict(arrowstyle="-|>",color="#555",lw=0.8))
ax.text(-44.5,-0.36,"baseline serves less",fontsize=7,color="#555",va="center",ha="left")
ax.set_yticks(range(len(SCEN))); ax.set_yticklabels([s[3].replace("\n"," ") for s in SCEN][::-1])
ax.set_ylim(-0.5,len(SCEN)-0.5); ax.set_xlim(-60,9)
ax.set_xlabel("Served share relative to TAPRA (percentage points)")
ax.grid(axis="x",alpha=0.3); ax.grid(axis="y",visible=False)
for s in ("top","right"): ax.spines[s].set_visible(False)
h=[plt.Line2D([],[],marker="*",color=TBLUE,ls="",ms=7,label="TAPRA (served share)")]+\
  [plt.Line2D([],[],marker=mk,color=c,mfc=c if m=="Full search every slot" else "white",ls="",ms=4.5,mew=1,label=l) for m,c,mk,l in BL]
ax.legend(handles=h,loc="upper center",bbox_to_anchor=(0.5,1.2),ncol=4,frameon=False,fontsize=7.5,handletextpad=0.3,columnspacing=1.2)
fig.tight_layout(); os.makedirs("figures", exist_ok=True); fig.savefig("figures/fig_served.pdf")
