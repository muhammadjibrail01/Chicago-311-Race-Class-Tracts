from config import MAIN, D, W, U
# Step 4: request-level models of log completion time on tract hardship and racial composition (type x year, month, weekday, hour, channel FE).
import json, numpy as np, pandas as pd
from scipy import stats
from common1 import load, W
from fe import feols
def se_by(m, cl):
    g=pd.factorize(np.asarray(cl)[m["idx"]] if False else cl)[0]; X=m["xt"]; e=m["resid"]; G=g.max()+1; n,k=X.shape
    S=np.column_stack([np.bincount(g,weights=X[:,j]*e,minlength=G) for j in range(k)]); Ai=np.linalg.inv(X.T@X)
    V=Ai@(S.T@S)@Ai*(G/(G-1))*((n-1)/(n-k)); return pd.Series(np.sqrt(np.diag(V)),index=m["coef"].index), G
import common1, gc
common1.TV=["hardship","pct_black","pct_hisp","poverty","pci","own","year_built","violent_rate","bviol_rate","density"]
r=load(["lhrs"]); r=r[~r.ty.isna()]
from common1 import tracts
r["ca_fe"]=r.ca.values
F=["ty","ym","dow","hod","chan"]; C=["z_own","z_year_built","ldens","z_violent_rate","z_bviol_rate"]
specs={"S1 hardship":(["z_hardship","lwl"],F),"S2 race":(["z_pct_black","z_pct_hisp","lwl"],F),"S3 both":(["z_hardship","z_pct_black","z_pct_hisp","lwl"],F),
 "S4 both+CA FE":(["z_hardship","z_pct_black","z_pct_hisp","lwl"],F+["ca_fe"]),"S5 +controls":(["z_hardship","z_pct_black","z_pct_hisp","lwl"]+C,F+["ca_fe"]),
 "S6 poverty,income+race":(["z_poverty","z_pci","z_pct_black","z_pct_hisp","lwl"],F)}
import os
out=json.load(open(W+'tabs/main_models.json')) if os.path.exists(W+'tabs/main_models.json') else {}
for k,(X,FE) in specs.items():
    if k in out: continue
    m=feols(r,"lhrs",X,fes=FE,cluster="ca"); d=r.loc[m["idx"]]
    se_t,Gt=se_by(m,d.tract.values)
    out[k]={"coef":m["coef"].to_dict(),"se_ca":m["se"].to_dict(),"p_ca":m["p"].to_dict(),"se_tract":se_t.to_dict(),"n":m["n"],"G_ca":m["G"],"G_tract":int(Gt)}
    show=[x for x in X if x!="lwl" and not x.startswith("z_own") and x not in C][:4]
    print(k, {x:f"{m['coef'][x]:.3f} (ca {m['se'][x]:.3f}; tr {se_t[x]:.3f})" for x in show}, f"n={m['n']:,}", flush=True)
    json.dump(out,open(W+"tabs/main_models.json","w"),default=float); del m; gc.collect()
