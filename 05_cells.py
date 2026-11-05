from config import MAIN, D, W, U
# Step 4c: race-by-poverty cell models, memory-lean (minimal columns). Reference cell: <10% Black, low poverty.
import json, gc, numpy as np, pandas as pd, common1
from common1 import W, tracts
from fe import feols
common1.TV=["hardship"]                     # load only one tract variable to save memory
r=common1.load(["lhrs"]); T=tracts()
grp=pd.read_csv(W+"tract_groups.csv",dtype={"GEOID":str}).set_index("GEOID").reindex(T.index)
cells=(grp.black_grp.astype(str)+" | "+grp.pov_grp.astype(str)).values; ref="<10% Black | low poverty"
cc=pd.Categorical(cells[r.tract.values]); names=[c for c in cc.categories if c!=ref]
for i,c in enumerate(names): r[f"g{i}"]=(cc==c).astype("float32")
r["ca_fe"]=r.ca.values; G=[f"g{i}" for i in range(len(names))]
out=json.load(open(W+"tabs/main_models.json")); F=["ty","ym","dow","hod","chan"]
for k,FE in [("S7 race-class cells",F),("S8 cells+CA FE",F+["ca_fe"])]:
    m=feols(r[["lhrs","lwl","ca","ca_fe","tract"]+G+F].copy(),"lhrs",G+["lwl"],fes=FE,cluster="ca")
    out[k]={"coef":{names[i]:float(m["coef"][f"g{i}"]) for i in range(len(names))},"se_ca":{names[i]:float(m["se"][f"g{i}"]) for i in range(len(names))},"n":m["n"]}
    json.dump(out,open(W+"tabs/main_models.json","w"),default=float)
    print(k,{names[i][:24]:f"{m['coef'][f'g{i}']:.3f}({m['se'][f'g{i}']:.3f})" for i in range(len(names))},flush=True); del m; gc.collect()
