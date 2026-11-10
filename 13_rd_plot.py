from config import MAIN, D, W, U
# Step 13: Figure 5, binned regression-discontinuity plots at C|D and B|C HOLC borders (residualized on FE incl. border pair).
import json, numpy as np, pandas as pd, matplotlib, common1
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from common1 import W
from fe import demean
common1.TV=["hardship"]; plt.rcParams.update({"font.family":"DejaVu Sans","font.size":8.5,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":300,"savefig.bbox":"tight"})
r=common1.load(["lhrs"]); b=pd.read_parquet(W+"boundary_inputs.parquet").set_index("row"); rl=json.load(open(W+"tabs/redlining.json"))
fig,ax=plt.subplots(1,2,figsize=(8.6,3.2),sharey=True)
for a,(col,lo,hi) in zip(ax,[("CD","C","D"),("BC","B","C")]):
    d=b[b.grade.isin([lo,hi]) & (b[f"dist_{col}"]<=400)].copy()
    d["run"]=np.where(d.grade==hi,1,-1)*d[f"dist_{col}"]; d["pair"]=np.where(d.grade==hi,d.area.astype(str)+"|"+d[f"nbr_{col}"].astype(str),d[f"nbr_{col}"].astype(str)+"|"+d.area.astype(str))
    s=r.loc[r.index.intersection(d.index),["lhrs","ty","ym","dow","hod","chan"]].copy(); s["run"]=d.run.reindex(s.index).values; s["pair"]=pd.factorize(d.pair.reindex(s.index).values)[0]
    s["res"]=demean(s[["lhrs"]].to_numpy(float),[s[f].to_numpy() for f in ["ty","ym","dow","hod","chan","pair"]])[:,0]
    s["bin"]=(np.floor(s.run/25)*25+12.5); g=s.groupby("bin").res.agg(["mean","size","sem"]).reset_index()
    a.errorbar(g["bin"],g["mean"],yerr=1.96*g["sem"],fmt="o",ms=3,color="#1b2a41",ecolor="#b0b0b0",capsize=0)
    for side in [s.run<0, s.run>=0]:
        q=s[side]; k=np.polyfit(q.run,q.res,1); xs=np.linspace(q.run.min(),q.run.max(),50); a.plot(xs,np.polyval(k,xs),color="#b2182b",lw=1.4)
    a.axvline(0,color="k",lw=.7,ls="--"); e=rl[f"RD {col} bw100"]
    a.set_title(f"({'a' if col=='CD' else 'b'}) {lo} | {hi} borders"); a.text(0.02,0.95,f"Jump at 100 m: {e['b']:.3f} (SE {e['se']:.3f})\nJump at 400 m: {rl[f'RD {col} bw400']['b']:.3f} (SE {rl[f'RD {col} bw400']['se']:.3f})",transform=a.transAxes,va="top",fontsize=7.5)
    a.set_xlabel(f"Distance to border (m); negative = {lo} side, positive = {hi} side")
ax[0].set_ylabel("Adjusted log completion time"); fig.savefig(W+"figs/fig5_rd.png"); plt.close(); print("fig5 written")
