import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle, Ellipse, FancyArrowPatch, Polygon
import numpy as np

BLUE="#1f5f9e"; LBLUE="#7fb2df"; ORANGE="#e8892b"; RED="#c0392b"; GRAY="#7f8c8d"; DARK="#2c3e50"; GREEN="#2e8b57"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":11})
fig=plt.figure(figsize=(13.2,5.9),dpi=100)
fig.patch.set_facecolor("white")

# ---------- title ----------
fig.text(0.5,0.94,"TAPRA: positioning one UAV for moving users in a city",ha="center",va="center",
         fontsize=19,fontweight="bold",color=DARK)
fig.text(0.5,0.885,"Traffic-aware positioning and airtime allocation, planned over each time slot",
         ha="center",va="center",fontsize=12.5,color=GRAY)

def panel_title(ax,t,x=None):
    b=ax.get_position(); fig.text(x if x is not None else (b.x0+b.x1)/2,0.80,t,ha="center",va="bottom",fontsize=13.5,fontweight="bold",color=BLUE)

# ---------- left: scene ----------
ax=fig.add_axes([0.02,0.15,0.32,0.62]); ax.set_xlim(0,10); ax.set_ylim(0,7.4); ax.axis("off")
panel_title(ax,"The problem")
ax.add_patch(Rectangle((0,0),10,0.35,color="#d5d8dc",zorder=0))
for x,w,h in [(0.6,1.4,3.2),(3.3,1.1,2.2),(6.9,1.6,3.6),(8.8,0.9,1.6)]:
    ax.add_patch(Rectangle((x,0.35),w,h,facecolor="#b3b6b7",edgecolor="#5d6d7e",lw=1,zorder=2))
    for yy in np.arange(0.75,0.35+h-0.3,0.55):
        for xx in np.arange(x+0.2,x+w-0.25,0.4):
            ax.add_patch(Rectangle((xx,yy),0.18,0.25,color="#eaeded",zorder=3))
# UAV
ux,uy=5.0,6.2
ax.add_patch(Rectangle((ux-0.35,uy-0.12),0.7,0.24,color=DARK,zorder=6))
for dx in (-0.8,0.8):
    ax.plot([ux,ux+dx],[uy+0.12,uy+0.12],color=DARK,lw=2.2,zorder=6)
    ax.add_patch(Ellipse((ux+dx,uy+0.2),0.9,0.16,color=DARK,alpha=0.85,zorder=6))
ax.text(ux,uy+0.62,"one UAV (Wi-Fi 6 access point)",ha="center",fontsize=10.5,color=DARK)
# UEs
ues=[(2.9,0.55,"p",LBLUE,(0.6,0)),(5.6,0.55,"v",GREEN,(1.1,0)),(8.4,0.55,"p",LBLUE,(-0.6,0)),(2.3,0.55,"p",LBLUE,(-0.5,0))]
for x,y,k,c,(dx,dy) in ues:
    if k=="p": ax.add_patch(Ellipse((x,y+0.15),0.32,0.32,color=c,zorder=7,ec=DARK,lw=0.8))
    else: ax.add_patch(FancyBboxPatch((x-0.28,y),0.56,0.3,boxstyle="round,pad=0.03",color=c,zorder=7,ec=DARK,lw=0.8))
    ax.annotate("",xy=(x+dx,y+0.15),xytext=(x+(0.25 if dx>0 else -0.25),y+0.15),
                arrowprops=dict(arrowstyle="-|>",color=DARK,lw=1.4),zorder=8)
# links
ax.plot([ux,2.9],[uy-0.12,0.72],color=GREEN,lw=2,zorder=5)
ax.plot([ux,5.6],[uy-0.12,0.85],color=GREEN,lw=2,zorder=5)
ax.plot([ux,8.4],[uy-0.12,0.72],color=RED,lw=2,ls="--",zorder=5)
ax.text(1.0,4.75,"LoS",color=GREEN,fontsize=11,fontweight="bold")
ax.text(7.9,4.75,"blocked",color=RED,fontsize=11,fontweight="bold")
fig.text(0.18,0.075,"Users move during each slot while the UAV flies,\nbuildings block links, and all users share one channel.",
        ha="center",va="center",fontsize=10.5,color=DARK)

# ---------- middle: method ----------
ax=fig.add_axes([0.355,0.04,0.30,0.75]); ax.set_xlim(0,10); ax.set_ylim(0,10); ax.axis("off")
panel_title(ax,"TAPRA, every 5 s slot")
steps=[("Predict","where each user will be\nduring the slot"),
       ("Score","each UAV position by the demand\nserved on average over the slot"),
       ("Admit","users to the shared airtime\n(model calibrated in ns-3)"),
       ("Search","grid + pattern search;\nlight update between full searches")]
ys=[8.3,6.0,3.7,1.4]
for (h,t),y in zip(steps,ys):
    ax.add_patch(FancyBboxPatch((0.4,y-0.85),9.2,1.7,boxstyle="round,pad=0.05,rounding_size=0.25",
                                fc="#eaf2fb",ec=BLUE,lw=1.4))
    ax.text(0.7,y,h,fontsize=13,fontweight="bold",color=BLUE,va="center")
    ax.text(2.95,y,t,fontsize=10.2,color=DARK,va="center")
for y0,y1 in zip(ys[:-1],ys[1:]):
    ax.annotate("",xy=(5,y1+0.9),xytext=(5,y0-0.9),arrowprops=dict(arrowstyle="-|>",color=BLUE,lw=1.6))

# ---------- right: results ----------
ax=fig.add_axes([0.79,0.43,0.17,0.33])
panel_title(ax,"Results over 60 random cities",x=0.835)
meth=["TAPRA","VTOPA rule","RLTOPA","Fixed rooftop AP"]
vals=[53.0,49.1,38.2,8.7]
cols=[BLUE,ORANGE,"#c0504d",GRAY]
y=np.arange(len(meth))[::-1]
ax.barh(y,vals,color=cols,height=0.62)
for yi,v in zip(y,vals): ax.text(v+1.2,yi,f"{v:.0f}%",va="center",fontsize=11,color=DARK)
ax.set_yticks(y); ax.set_yticklabels(meth,fontsize=11)
ax.set_xlim(0,68); ax.set_xticks([]); 
for s in ("top","right","bottom"): ax.spines[s].set_visible(False)
ax.set_xlabel("Demand served (urban, 500 m, 20 users)",fontsize=10,color=DARK,x=0.35)
txt=[("+3.3 to 5.7 points","over the strongest baseline"),
     ("within 1.3 points","of a full search, 30–54% fewer evaluations"),
     ("1.5 points","mean gap to packet-level ns-3")]
for i,(a,b) in enumerate(txt):
    yy=0.31-i*0.085
    fig.text(0.705,yy,a,fontsize=12.5,fontweight="bold",color=BLUE,va="center")
    fig.text(0.705,yy-0.037,b,fontsize=10.2,color=DARK,va="center")

fig.savefig("graphical_abstract.png",dpi=100)
fig.savefig("graphical_abstract_hires.png",dpi=300)
fig.savefig("graphical_abstract.pdf")
print("ok")
