import os
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from plots import ci95
plt.rcParams.update({"font.size": 7, "font.family": "serif"})
e=pd.read_csv("results/energy.csv")
EN=[("TAPRA","#1b6ca8","o","TAPRA"),("Full search every slot","#7fb3d5","D","Full search"),("VTOPA rule","#e67e22","s","VTOPA rule"),
    ("Traffic centroid","#7f8c8d","^","Traffic centroid"),("Fixed rooftop AP","#2c3e50","v","Hovering")]
ESC=[("suburban","homogeneous","Suburban 500 m\nhomogeneous"),("urban","homogeneous","Urban 500 m\nhomogeneous"),("urban","heterogeneous","Urban 500 m\nheterogeneous")]
fig,(a1,a2)=plt.subplots(1,2,figsize=(3.5,3.5),sharey=True,gridspec_kw={"width_ratios":[1,1.15]})
off=np.linspace(0.3,-0.3,len(EN))
for r,(env,tr,lab) in enumerate(ESC):
    y0=len(ESC)-1-r
    for ax in (a1,a2):
        if r%2==0: ax.axhspan(y0-0.5,y0+0.5,color="#f2f4f4",zorder=0)
    for (m,c,mk,_),dy in zip(EN,off):
        x=e[(e.env==env)&(e.traffic==tr)&(e.method==m)]
        y=y0+dy
        for ax,col in ((a1,"energy_kj"),(a2,"planned_mbit_per_kj")):
            v=x[col]; ax.errorbar(v.mean(),y,xerr=ci95(v),fmt=mk,ms=3.6,color=c,mew=0.8,elinewidth=0.8,capsize=1.2,zorder=4)
        if m=="TAPRA":
            a2.text(x.planned_mbit_per_kj.mean()+60,y,f"{x.planned_mbit_per_kj.mean():.0f}",fontsize=5.8,color=c,va="center")
hov=e[e.method=="Fixed rooftop AP"].energy_kj.mean()
a1.axvline(hov,color="#2c3e50",ls="--",lw=0.8,zorder=1)
a1.text(hov,len(ESC)-0.45,f"hovering\n{hov:.1f} kJ",fontsize=5.8,ha="center",va="bottom",color="#2c3e50")
a1.set_xlim(39,47); a1.set_xlabel("Energy over 250 s (kJ)",fontsize=6.5)
a2.set_xlabel("Served data per kJ (Mbit/kJ)",fontsize=6.5)
a1.set_yticks(range(len(ESC))); a1.set_yticklabels([s[2] for s in ESC][::-1],fontsize=5.8,linespacing=1.0,rotation=90,va="center",ha="center")
a1.tick_params(axis="y",pad=7)
a1.set_ylim(-0.5,len(ESC)-0.5)
for ax in (a1,a2):
    ax.grid(axis="x",alpha=0.3); ax.grid(axis="y",visible=False)
    for s in ("top","right"): ax.spines[s].set_visible(False)
a2.set_xlim(0,a2.get_xlim()[1]*1.1)
h=[plt.Line2D([],[],marker=mk,color=c,ls="",ms=4,label=l) for _,c,mk,l in EN]
fig.legend(handles=h,loc="upper center",bbox_to_anchor=(0.55,1.07),ncol=3,frameon=False,fontsize=6.3,handletextpad=0.2,columnspacing=0.8)
fig.tight_layout(); fig.subplots_adjust(wspace=0.12); os.makedirs("figures", exist_ok=True); fig.savefig("figures/fig_energy.pdf",bbox_inches="tight")
