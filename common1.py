from config import MAIN, D, W, U
import sys, numpy as np, pandas as pd, pyarrow.parquet as pq

TV=["hardship","hardship_pc","pct_black","pct_hisp","pct_asian","pct_white","poverty","pci","own","year_built","violent_rate","bviol_rate","density","hri","mhi"]
def tracts():
    t=pd.read_csv(W+"tract.csv",dtype={"GEOID":str}).set_index("GEOID"); return t[t.analytic]
def load(cols, filt="speed_s"):
    base=["type","chan","year","ym","dow","hod","ca","ward","lwl","dept"]
    c=list(dict.fromkeys(base+cols+([filt] if filt else [])))
    r=pq.read_table(MAIN+"core.parquet",columns=c,read_dictionary=["type","chan","dept"]).to_pandas()
    loc=pd.read_parquet(W+"core_locations.parquet"); t=tracts()
    r["tract"]=pd.Categorical(loc.GEOID.values,categories=t.index).codes.astype(np.int32)
    r["holc_grade"]=loc.holc_grade.values; r["holc_area"]=loc.holc_area.values
    if filt: r=r[r[filt].values]
    r=r[r.tract>=0].copy()
    for v in TV:
        s=t[v]; z=(s-s.mean())/s.std(); r["z_"+v]=z.values[r.tract.values].astype("float32")
    r["ldens"]=np.log(t.density.values[r.tract.values]).astype("float32")
    r["type"]=r.type.astype(str).astype("category")
    r["ty"]=(r.type.cat.codes.astype(np.int32)*10+(r.year-2019)).astype(np.int32)
    r["wy"]=(r.ward.astype(np.int32)*10+(r.year-2019)).astype(np.int32)
    return r
