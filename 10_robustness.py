from config import MAIN, D, W, U
# Step 10: robustness of the race-vs-class result (baseline = hardship + Black + Hispanic shares, type x year etc. FE).
import json, numpy as np, pandas as pd, geopandas as gpd, common1
from common1 import W, tracts
from fe import feols
common1.TV=["hardship","hardship_pc","pct_black","pct_hisp","mhi","pci"]
r=common1.load(["lhrs","prank","slow","hrs"]); T=tracts(); F=["ty","ym","dow","hod","chan"]
t=gpd.read_file(D+"tracts/cb_2023_17_tract_500k.shp").set_index("GEOID").loc[T.index].to_crs(26916)
city=gpd.read_file(MAIN+"ca.gpkg").to_crs(26916).dissolve()
share=gpd.overlay(t.reset_index()[["GEOID","geometry"]],city[["geometry"]],how="intersection").set_index("GEOID").area/t.area
border=np.asarray(share.reindex(T.index).fillna(0)<0.99); r["border"]=border[r.tract.values]
r["ca_fe"]=r.ca.values; r["wy_fe"]=r.wy.values
n_t=r.groupby("tract").tract.transform("size"); r["w_eq"]=(1/n_t).astype("float64")
base=["z_hardship","z_pct_black","z_pct_hisp","lwl"]; out={}
def run(k,d,X=base,y="lhrs",FE=F,cl="ca",w=None):
    m=feols(d,y,X,fes=FE,cluster=cl,w=w); out[k]={"coef":m["coef"].to_dict(),"se":m["se"].to_dict(),"p":m["p"].to_dict(),"n":m["n"],"G":m["G"]}
    print(k,{x:f"{m['coef'][x]:.3f}({m['se'][x]:.3f})" for x in X[:3]},flush=True); json.dump(out,open(W+"tabs/robustness.json","w"),default=float)
run("baseline",r)
run("exclude border tracts",r[~r.border])
run("percentile rank outcome",r,y="prank")
r["slow100"]=100*r.slow; run("slower-than-median outcome (pp)",r,y="slow100")
run("cluster by ward",r[r.ward>0],cl="ward")
run("exclude 2020",r[r.year!=2020])
run("exclude downtown (CA 8, 32, 33)",r[~r.ca.isin([8,32,33])])
run("PCA hardship index",r,X=["z_hardship_pc","z_pct_black","z_pct_hisp","lwl"])
run("median household income",r,X=["z_mhi","z_pct_black","z_pct_hisp","lwl"])
run("ward x year FE",r,FE=F+["wy_fe"])
run("tracts weighted equally",r,w="w_eq")
r["lraw"]=np.log1p(r.hrs.clip(lower=0)); run("unwinsorized log hours",r,y="lraw")
