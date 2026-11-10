from config import MAIN, D, W, U
# Step 8: quality outcomes at the tract level: re-report within 30/90 days (speed sample), cancellation and unresolved (resident, non-duplicate).
import json, numpy as np, pandas as pd, common1
from common1 import W
from fe import feols
common1.TV=["hardship","pct_black","pct_hisp"]; X=["z_hardship","z_pct_black","z_pct_hisp","lwl"]; F=["ty","ym","dow","hod","chan"]; out={}
r=common1.load(["recur30","recur90"])
for y in ["recur30","recur90"]:
    r[y]=100*r[y]; m=feols(r,y,X,fes=F,cluster="ca"); out[y]={"coef":m["coef"].to_dict(),"se":m["se"].to_dict(),"n":m["n"],"mean":m["ymean"]}
    print(y,{k:f"{m['coef'][k]:.3f}({m['se'][k]:.3f})" for k in X[:3]},f"mean={m['ymean']:.2f}",flush=True)
del r; r=common1.load(["canceled","unresolved","dup"],filt="resident"); r=r[~r.dup]
for y in ["canceled","unresolved"]:
    r[y]=100*r[y].astype("float32"); m=feols(r,y,X,fes=F,cluster="ca"); out[y]={"coef":m["coef"].to_dict(),"se":m["se"].to_dict(),"n":m["n"],"mean":m["ymean"]}
    print(y,{k:f"{m['coef'][k]:.3f}({m['se'][k]:.3f})" for k in X[:3]},f"mean={m['ymean']:.2f}",flush=True)
json.dump(out,open(W+"tabs/quality.json","w"),default=float)
