from config import MAIN, D, W, U
# Step 12: figures for Paper 1.
import json, numpy as np, pandas as pd, geopandas as gpd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
W=W+""; D=D+""; FG=W+"figs/"
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.5,"axes.titlesize":9,"axes.labelsize":8.5,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":300,"savefig.bbox":"tight"})
RED,BLUE,GREY,NAVY="#b2182b","#2166ac","#7f7f7f","#1b2a41"
J=lambda f: json.load(open(W+"tabs/"+f)); pct=lambda b: 100*(np.exp(b)-1); ci=lambda b,s: (pct(b-1.96*s),pct(b+1.96*s))
mm,rl,js,ql,rb,sp,cl=[J(f) for f in ["main_models.json","redlining.json","jobscope_channel.json","quality.json","robustness.json","spatial.json","collinearity.json"]]
# ---- Figure 1: maps
T=pd.read_csv(W+"tract.csv",dtype={"GEOID":str}); A=pd.read_csv(W+"tract_adjusted.csv",dtype={"GEOID":str})
g=gpd.read_file(D+"tracts/cb_2023_17_tract_500k.shp").merge(T,on="GEOID").merge(A[["GEOID","adj"]],on="GEOID",how="left").to_crs(26916)
city=gpd.read_file(MAIN+"ca.gpkg").to_crs(26916).dissolve()
h=gpd.read_file(D+"holc/holc_chicago.gpkg").to_crs(26916); h=h[h.grade.isin(list("ABCD"))]
fig,ax=plt.subplots(2,2,figsize=(8.2,9.4)); ax=ax.ravel()
g[g.analytic].plot("hardship",ax=ax[0],cmap="Reds",legend=True,legend_kwds={"shrink":.55,"label":"Hardship index"},linewidth=0)
g[g.analytic].plot("pct_black",ax=ax[1],cmap="Purples",legend=True,legend_kwds={"shrink":.55,"label":"% Black residents"},linewidth=0)
city.boundary.plot(ax=ax[2],color=GREY,linewidth=.5)
for gr,c in zip("ABCD",["#76a865","#7cb5bd","#e8d25a","#d9534f"]): h[h.grade==gr].plot(ax=ax[2],color=c,linewidth=0,label=gr)
ax[2].legend(handles=[plt.Rectangle((0,0),1,1,color=c) for c in ["#76a865","#7cb5bd","#e8d25a","#d9534f"]],labels=list("ABCD"),title="1930s HOLC grade",fontsize=7,title_fontsize=7,frameon=False,loc="lower left")
g[g.analytic].plot(g.loc[g.analytic,"adj"].clip(-.5,.5),ax=ax[3],cmap="RdBu_r",norm=TwoSlopeNorm(0,-.5,.5),legend=True,legend_kwds={"shrink":.55,"label":"Adjusted wait (log points)"},linewidth=0)
for a,t in zip(ax,["(a) Hardship index","(b) Black residents","(c) 1930s HOLC map","(d) Adjusted completion time"]):
    city.boundary.plot(ax=a,color="k",linewidth=.4); a.set_axis_off(); a.set_title(t)
    x0,y0,x1,y1=city.total_bounds; a.set_xlim(x0-500,x1+500); a.set_ylim(y0-500,y1+500)
fig.savefig(FG+"fig1_maps.png"); plt.close()
# ---- Figure 2: race x poverty heatmap
rows=["<10% Black","10-60% Black",">60% Black"]; cols=["low poverty","middle poverty","high poverty"]; ref="<10% Black | low poverty"
fig,ax=plt.subplots(1,2,figsize=(9.6,3.5),constrained_layout=True)
for a,(k,title) in zip(ax,[("S7 race-class cells","(a) Citywide comparison"),("S8 cells+CA FE","(b) Within community areas")]):
    M=np.zeros((3,3)); S=np.full((3,3),"",dtype=object)
    for i,rw in enumerate(rows):
        for j,c in enumerate(cols):
            key=f"{rw} | {c}"; b=0.0 if key==ref else mm[k]["coef"][key]; s=None if key==ref else mm[k]["se_ca"][key]
            M[i,j]=pct(b); n=cl["cells"][c][rw]; S[i,j]=("reference" if s is None else f"{pct(b):+.1f}%\n[{ci(b,s)[0]:+.0f}, {ci(b,s)[1]:+.0f}]")+f"\n{n} tracts"
    im=a.imshow(M,cmap="Reds",vmin=0,vmax=32)
    for i in range(3):
        for j in range(3): a.text(j,i,S[i,j],ha="center",va="center",fontsize=6.8,color="white" if M[i,j]>20 else "black")
    a.set_xticks(range(3),[c.capitalize() for c in cols]); a.set_yticks(range(3),rows if a is ax[0] else [""]*3); a.set_title(title); a.spines[:].set_visible(False)
fig.colorbar(im,ax=ax,shrink=.8,label="% longer completion time vs. reference cell"); fig.savefig(FG+"fig2_race_class_grid.png"); plt.close()
# ---- Figure 3: model comparison
labs=[("S1 hardship","Hardship only"),("S2 race","Race only"),("S3 both","Hardship + race"),("S4 both+CA FE","+ community-area FE"),("S5 +controls","+ need and housing controls"),("S6 poverty,income+race","Poverty and income + race")]
fig,a=plt.subplots(figsize=(6.4,3.4)); y=np.arange(len(labs))[::-1]
for off,var,c,name in [(-.18,"z_hardship",GREY,"Hardship index"),(0,"z_pct_black",RED,"Black share"),(.18,"z_pct_hisp",BLUE,"Hispanic share"),(-.18,"z_poverty",GREY,None)]:
    for yi,(k,_) in zip(y,labs):
        e=mm[k]
        if var in e["coef"]:
            b,s=e["coef"][var],e["se_ca"][var]; a.errorbar(pct(b),yi+off,xerr=[[pct(b)-ci(b,s)[0]],[ci(b,s)[1]-pct(b)]],fmt="o",color=c,ms=4,capsize=2,label=name if (name and yi==y[0+(0 if var!="z_pct_black" else 1)]) else None)
a.axvline(0,color="k",lw=.6); a.set_yticks(y,[l for _,l in labs]); a.set_xlabel("% change in completion time per +1 SD (95% CI, clustered by community area)")
from matplotlib.lines import Line2D
a.legend([Line2D([],[],marker="o",ls="",color=c) for c in [GREY,RED,BLUE]],["Class (hardship, or poverty and income)","Black share","Hispanic share"],frameon=False,fontsize=7.5,loc="lower right"); fig.savefig(FG+"fig3_models.png"); plt.close()
# ---- Figure 4: bigger-jobs test
fig,ax=plt.subplots(1,2,figsize=(9,3.4),gridspec_kw={"width_ratios":[1,1.35]})
grp=[("fixed","Fixed-scope jobs"),("variable","Variable-scope jobs"),("other","Other services")]
for off,tag,c,name in [(.12,"",RED,"Citywide"),(-.12," +CA",NAVY,"Within community areas")]:
    for yi,(k,l) in zip([2,1,0],grp):
        e=js[f"scope|{k}{tag}"]; b,s=e["coef"]["z_pct_black"],e["se"]["z_pct_black"]
        ax[0].errorbar(pct(b),yi+off,xerr=[[pct(b)-ci(b,s)[0]],[ci(b,s)[1]-pct(b)]],fmt="o",color=c,ms=4,capsize=2,label=name if yi==2 else None)
ax[0].set_yticks([2,1,0],[l for _,l in grp]); ax[0].axvline(0,color="k",lw=.6); ax[0].legend(frameon=False,fontsize=7.5); ax[0].set_title("(a) By job scope"); ax[0].set_xlabel("% change per +1 SD Black share")
types=[k.split("|",1)[1] for k in js if k.startswith("type|")]; yy=np.arange(len(types))[::-1]
for yi,t in zip(yy,types):
    e=js["type|"+t]; b,s=e["coef"]["z_pct_black"],e["se"]["z_pct_black"]; c=RED if e["scope"]=="variable" else NAVY
    ax[1].errorbar(pct(b),yi,xerr=[[pct(b)-ci(b,s)[0]],[ci(b,s)[1]-pct(b)]],fmt="o",color=c,ms=4,capsize=2)
ax[1].set_yticks(yy,[t.replace(" Complaint","").replace(" Request","") for t in types]); ax[1].axvline(0,color="k",lw=.6)
ax[1].set_title("(b) Single service types (red = variable scope)"); ax[1].set_xlabel("% change per +1 SD Black share"); fig.tight_layout(); fig.savefig(FG+"fig4_jobscope.png"); plt.close()
# ---- Figure 6: robustness; Figure 7: quality and channel
fig,a=plt.subplots(figsize=(6.4,3.6)); ks=[k for k in rb if k not in ("percentile rank outcome","slower-than-median outcome (pp)")]; y=np.arange(len(ks))[::-1]
for yi,k in zip(y,ks):
    e=rb[k]; bv=[x for x in e["coef"] if x.startswith("z_") and x not in ("z_pct_black","z_pct_hisp")][0]
    if k in ("percentile rank outcome","slower-than-median outcome (pp)"): continue
    for var,c,off in [("z_pct_black",RED,.12),(bv,GREY,-.12)]:
        b,s=e["coef"][var],e["se"][var]; a.errorbar(pct(b),yi+off,xerr=[[pct(b)-ci(b,s)[0]],[ci(b,s)[1]-pct(b)]],fmt="o",color=c,ms=3.5,capsize=2)
a.set_yticks(y,ks,fontsize=7.5); a.axvline(0,color="k",lw=.6); a.set_xlabel("% change per +1 SD (red: Black share; grey: class measure)"); fig.savefig(FG+"fig6_robustness.png"); plt.close()
fig,ax=plt.subplots(1,2,figsize=(9,3))
qs=[("recur30","Re-reported within 30 days"),("recur90","Re-reported within 90 days"),("canceled","Canceled"),("unresolved","Still unresolved")]
for off,var,c,name in [(.12,"z_pct_black",RED,"Black share"),(-.12,"z_hardship",GREY,"Hardship index")]:
    for yi,(k,l) in zip([3,2,1,0],qs):
        b,s=ql[k]["coef"][var],ql[k]["se"][var]; ax[0].errorbar(b,yi+off,xerr=1.96*s,fmt="o",color=c,ms=4,capsize=2,label=name if yi==3 else None)
ax[0].set_yticks([3,2,1,0],[l for _,l in qs]); ax[0].axvline(0,color="k",lw=.6); ax[0].legend(frameon=False,fontsize=7.5); ax[0].set_title("(a) Quality of resolution"); ax[0].set_xlabel("Percentage points per +1 SD")
chs=[("phone","Phone"),("digital","Web or app"),("alderman","Aldermanic office")]
for yi,(k,l) in zip([2,1,0],chs):
    b,s=js[f"chan|{k}"]["coef"]["z_pct_black"],js[f"chan|{k}"]["se"]["z_pct_black"]; ax[1].errorbar(pct(b),yi,xerr=[[pct(b)-ci(b,s)[0]],[ci(b,s)[1]-pct(b)]],fmt="o",color=RED,ms=4,capsize=2)
ax[1].set_yticks([2,1,0],[l for _,l in chs]); ax[1].axvline(0,color="k",lw=.6); ax[1].set_title("(b) Black-share gradient by reporting channel"); ax[1].set_xlabel("% change in completion time per +1 SD")
fig.tight_layout(); fig.savefig(FG+"fig7_quality_channel.png"); plt.close(); print("figures 1-4, 6-7 written")
