from config import MAIN, D, W, U
# Step 6: redlining. (a) request-level HOLC grade and tract HRI models; (b) geographic regression-discontinuity at C|D and B|C
# borders: within border-pair fixed effects, local linear in signed distance, bandwidths 100/200/400 m, clustered by pair.
import json, os, numpy as np, pandas as pd
from common1 import load, W, tracts
from fe import feols
out=json.load(open(W+"tabs/redlining.json")) if os.path.exists(W+"tabs/redlining.json") else {}
def save(): json.dump(out,open(W+"tabs/redlining.json","w"),default=float)
r=load(["lhrs"]); F=["ty","ym","dow","hod","chan"]
for g in ["A","B","D"]: r["grade_"+g]=(r.holc_grade==g).astype("float32")
r["graded"]=r.holc_grade.isin(["A","B","C","D"])
gr=r[r.graded].copy()
for k,X in [("grade only",["grade_A","grade_B","grade_D","lwl"]),("grade+race+hardship",["grade_A","grade_B","grade_D","z_pct_black","z_pct_hisp","z_hardship","lwl"])]:
    if k in out: continue
    m=feols(gr,"lhrs",X,fes=F,cluster="ca"); out[k]={"coef":m["coef"].to_dict(),"se":m["se"].to_dict(),"p":m["p"].to_dict(),"n":m["n"]}; save()
    print(k,{x:f"{m['coef'][x]:.3f} ({m['se'][x]:.3f})" for x in X if x!="lwl"},flush=True)
for k,X in [("hri only",["z_hri","lwl"]),("hri+race+hardship",["z_hri","z_pct_black","z_pct_hisp","z_hardship","lwl"])]:
    if k in out: continue
    m=feols(r,"lhrs",X,fes=F,cluster="ca"); out[k]={"coef":m["coef"].to_dict(),"se":m["se"].to_dict(),"p":m["p"].to_dict(),"n":m["n"]}; save()
    print(k,{x:f"{m['coef'][x]:.3f} ({m['se'][x]:.3f})" for x in X if x!="lwl"},flush=True)
del gr
b=pd.read_parquet(W+"boundary_inputs.parquet")
core_rows=np.where(pd.read_parquet(MAIN+"core.parquet",columns=["speed_s"]).speed_s.values)[0]
pos=pd.Series(np.arange(len(core_rows)),index=core_rows)          # map core row -> speed-sample position
r=r.reset_index(drop=True) if False else r
T=tracts()
for col,lo,hi in [("CD","C","D"),("BC","B","C")]:
    d=b[b.grade.isin([lo,hi]) & b[f"dist_{col}"].notna()].copy()
    d["run"]=np.where(d.grade==hi,1,-1)*d[f"dist_{col}"]; d["pair"]=np.where(d.grade==hi, d.area.astype(str)+"|"+d[f"nbr_{col}"].astype(str), d[f"nbr_{col}"].astype(str)+"|"+d.area.astype(str))
    d=d.set_index("row")
    for bw in [100,200,400]:
        k=f"RD {col} bw{bw}"
        if k in out: continue
        s=d[d[f"dist_{col}"]<=bw]; sub=r.loc[r.index.intersection(s.index)].copy()
        sub["run"]=s.run.reindex(sub.index).values/100; sub["treat"]=(sub.run>0).astype("float32"); sub["tr_run"]=sub.treat*sub.run
        sub["pair"]=pd.factorize(s.pair.reindex(sub.index).values)[0]; sub["pair_fe"]=sub["pair"]
        m=feols(sub,"lhrs",["treat","run","tr_run","lwl"],fes=F+["pair_fe"],cluster="pair")
        bal={}
        for v in ["z_pct_black","z_poverty","z_hardship"]:
            mb=feols(sub,v,["treat","run","tr_run"],fes=["pair_fe"],cluster="pair"); bal[v]={"b":float(mb["coef"]["treat"]),"se":float(mb["se"]["treat"])}
        m2=feols(sub,"lhrs",["treat","run","tr_run","z_pct_black","z_pct_hisp","z_hardship","lwl"],fes=F+["pair_fe"],cluster="pair")
        out[k]={"b":float(m["coef"]["treat"]),"se":float(m["se"]["treat"]),"p":float(m["p"]["treat"]),"n":m["n"],"pairs":m["G"],"balance":bal,
                "b_ctrl":float(m2["coef"]["treat"]),"se_ctrl":float(m2["se"]["treat"])}; save()
        print(k,f"jump {m['coef']['treat']:.3f} ({m['se']['treat']:.3f}) n={m['n']:,} pairs={m['G']} | with tract controls {m2['coef']['treat']:.3f} ({m2['se']['treat']:.3f}) | balance black {bal['z_pct_black']['b']:.2f} pov {bal['z_poverty']['b']:.2f}",flush=True)
