from config import MAIN, D, W, U
# Step 3: does tract-level data separate race from class? Collinearity at CA vs tract level; within-CA variation; race-class cells.
import json, numpy as np, pandas as pd
from common1 import tracts, W
t=tracts(); ca=pd.read_csv(MAIN+"ca.csv")
out={}
out["corr_ca_hardship_black"]=float(ca[["hardship","pct_black"]].corr().iloc[0,1])
out["corr_ca_lowinc_black"]=float(ca[["lowinc","pct_black"]].corr().iloc[0,1])
out["corr_tract_hardship_black"]=float(t[["hardship","pct_black"]].corr().iloc[0,1])
out["corr_tract_poverty_black"]=float(t[["poverty","pct_black"]].corr().iloc[0,1])
out["corr_tract_hardship_hisp"]=float(t[["hardship","pct_hisp"]].corr().iloc[0,1])
for v in ["hardship","pct_black","pct_hisp","hri"]:
    s=t[v].dropna(); g=t.loc[s.index,"ca"]; out[f"within_ca_share_{v}"]=float(((s-s.groupby(g).transform("mean"))**2).sum()/((s-s.mean())**2).sum())
# residual race variation after hardship, at each level
def resid_share(df,y,x):
    b=np.polyfit(df[x],df[y],1); e=df[y]-np.polyval(b,df[x]); return float(e.var()/df[y].var())
out["black_var_unexplained_by_hardship_ca"]=resid_share(ca,"pct_black","hardship"); out["black_var_unexplained_by_hardship_tract"]=resid_share(t,"pct_black","hardship")
t["black_grp"]=pd.cut(t.pct_black,[-1,10,60,101],labels=["<10% Black","10-60% Black",">60% Black"])
t["pov_grp"]=pd.qcut(t.poverty,3,labels=["low poverty","middle poverty","high poverty"])
cells=pd.crosstab(t.black_grp,t.pov_grp); out["cells"]=cells.to_dict()
t[["black_grp","pov_grp"]].to_csv(W+"tract_groups.csv")
json.dump(out,open(W+"tabs/collinearity.json","w"),indent=1,default=str)
print(json.dumps({k:(round(v,3) if isinstance(v,float) else v) for k,v in out.items() if k!="cells"},indent=0)); print(cells)
