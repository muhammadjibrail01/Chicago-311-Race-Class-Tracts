from config import MAIN, D, W, U
# Step 7: the "bigger jobs" test (fixed-scope vs variable-scope services; key single types) and heterogeneity by reporting channel.
import json, gc, numpy as np, pandas as pd, common1
from common1 import W
from fe import feols
common1.TV=["hardship","pct_black","pct_hisp"]
FIXED={"Street Light Out Complaint","Alley Light Out Complaint","Traffic Signal Out Complaint","Sign Repair Request - All Other Signs","Sign Repair Request - Stop Sign",
       "Sign Repair Request - One Way Sign","Street Light Pole Damage Complaint","Garbage Cart Maintenance","Blue Recycling Cart"}
VARIABLE={"Graffiti Removal Request","Weed Removal Request","Tree Debris Clean-Up Request","Sanitation Code Violation","Fly Dumping Complaint","Clean Vacant Lot Request",
          "Street Cleaning Request","Yard Waste Pick-Up Request","Pothole in Street Complaint","Alley Pothole Complaint"}
r=common1.load(["lhrs"]); r["ca_fe"]=r.ca.values; ty=r.type.astype(str)
r["scope"]=np.where(ty.isin(FIXED),"fixed",np.where(ty.isin(VARIABLE),"variable","other"))
X=["z_hardship","z_pct_black","z_pct_hisp","lwl"]; F=["ty","ym","dow","hod","chan"]; out={}
def fit(d,FE): m=feols(d,"lhrs",X,fes=FE,cluster="ca"); return {"coef":m["coef"].to_dict(),"se":m["se"].to_dict(),"p":m["p"].to_dict(),"n":m["n"]}
for g in ["fixed","variable","other"]:
    for tag,FE in [("",F),(" +CA",F+["ca_fe"])]:
        out[f"scope|{g}{tag}"]=fit(r[r.scope==g],FE); e=out[f"scope|{g}{tag}"]
        print(f"{g}{tag}: black {e['coef']['z_pct_black']:.3f} ({e['se']['z_pct_black']:.3f}) hardship {e['coef']['z_hardship']:.3f} ({e['se']['z_hardship']:.3f}) n={e['n']:,}",flush=True)
for t in ["Street Light Out Complaint","Alley Light Out Complaint","Traffic Signal Out Complaint","Garbage Cart Maintenance","Fly Dumping Complaint","Weed Removal Request","Graffiti Removal Request","Pothole in Street Complaint"]:
    d=r[ty==t]; m=feols(d,"lhrs",X,fes=["ym","dow","hod","chan"],cluster="ca")
    out[f"type|{t}"]={"coef":m["coef"].to_dict(),"se":m["se"].to_dict(),"n":m["n"],"scope":d.scope.iloc[0]}
    print(f"  {t[:32]:32s} black {m['coef']['z_pct_black']:.3f} ({m['se']['z_pct_black']:.3f})",flush=True)
for c in ["phone","digital","alderman"]:
    out[f"chan|{c}"]=fit(r[r.chan==c],["ty","ym","dow","hod"]); e=out[f"chan|{c}"]
    print(f"channel {c}: black {e['coef']['z_pct_black']:.3f} ({e['se']['z_pct_black']:.3f})",flush=True)
json.dump(out,open(W+"tabs/jobscope_channel.json","w"),default=float)
