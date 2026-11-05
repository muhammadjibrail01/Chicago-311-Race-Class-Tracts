from config import MAIN, D, W, U
# Step 11: tract-level adjusted wait index (mean residual from FE model without tract variables); WLS/HC3; Moran's I; spatial lag/error models.
import json, numpy as np, pandas as pd, geopandas as gpd, statsmodels.api as sm, common1
from libpysal.weights import Queen
from esda.moran import Moran
from spreg import ML_Error, ML_Lag
from common1 import W, tracts
from fe import feols
common1.TV=["hardship","pct_black","pct_hisp"]
r=common1.load(["lhrs"]); m=feols(r,"lhrs",["lwl"],fes=["ty","ym","dow","hod","chan"],cluster="ca")
d=r.loc[m["idx"]]; adj=pd.Series(m["resid"],index=d.index).groupby(d.tract.values).agg(["mean","size"])
T=tracts().reset_index(); T["adj"]=adj["mean"].reindex(T.index).values; T["n_req"]=adj["size"].reindex(T.index).values
g=gpd.read_file(D+"tracts/cb_2023_17_tract_500k.shp").set_index("GEOID").loc[T.GEOID].reset_index()
g=g.merge(T,on="GEOID"); g=g[g.adj.notna()].reset_index(drop=True)
for v in ["hardship","pct_black","pct_hisp"]: g["z_"+v]=(g[v]-g[v].mean())/g[v].std()
g[["GEOID","adj","n_req","z_hardship","z_pct_black","z_pct_hisp","ca"]].to_csv(W+"tract_adjusted.csv",index=False)
w=Queen.from_dataframe(g,use_index=False); w.transform="r"; out={"n":len(g),"islands":len(w.islands)}
for v in ["adj","z_pct_black","z_hardship"]: mi=Moran(g[v].values,w,permutations=999); out[f"moran|{v}"]={"I":float(mi.I),"p":float(mi.p_sim)}
Xv=["z_hardship","z_pct_black","z_pct_hisp"]; X=sm.add_constant(g[Xv])
ols=sm.WLS(g.adj,X,weights=g.n_req).fit(cov_type="HC3"); out["wls"]={"coef":ols.params.to_dict(),"se":ols.bse.to_dict()}
e=ML_Error(g.adj.values[:,None],g[Xv].values,w=w,name_x=Xv); l=ML_Lag(g.adj.values[:,None],g[Xv].values,w=w,name_x=Xv)
out["ml_error"]={"coef":dict(zip(["const"]+Xv+["lambda"],e.betas.ravel().tolist())),"se":dict(zip(["const"]+Xv+["lambda"],e.std_err.tolist()))}
out["ml_lag"]={"coef":dict(zip(["const"]+Xv+["rho"],l.betas.ravel().tolist())),"se":dict(zip(["const"]+Xv+["rho"],l.std_err.tolist()))}
mi=Moran(ols.resid.values,w,permutations=999); out["moran|wls_resid"]={"I":float(mi.I),"p":float(mi.p_sim)}
json.dump(out,open(W+"tabs/spatial.json","w"),default=float); print(json.dumps(out,default=float)[:900])
