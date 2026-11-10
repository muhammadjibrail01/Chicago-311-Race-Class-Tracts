from config import MAIN, D, W, U
# Step 5a: for each speed-sample request inside a graded HOLC area, find the nearest area of the adjacent grade
# (C<->D and B<->C), the distance to it, and a boundary-pair identifier (geographic regression-discontinuity inputs).
import numpy as np, pandas as pd, geopandas as gpd, shapely, pyarrow.parquet as pq
h=gpd.read_file(D+"holc/holc_chicago.gpkg"); h=h[h.grade.isin(["A","B","C","D"])].to_crs(26916).reset_index(drop=True)
core=pq.read_table(MAIN+"core.parquet",columns=["lat","lon","speed_s"]).to_pandas()
loc=pd.read_parquet(W+"core_locations.parquet")
keep=core.speed_s.values & loc.holc_grade.isin(["B","C","D"]).values
idx=np.where(keep)[0]
pts=gpd.GeoSeries(gpd.points_from_xy(core.lon.values[idx],core.lat.values[idx]),crs=4326).to_crs(26916).values
own_area=loc.holc_area.values[idx]; own_grade=loc.holc_grade.values[idx]
res=pd.DataFrame({"row":idx,"grade":own_grade,"area":own_area})
for lo,hi in [("C","D"),("B","C")]:
    for g,other in [(lo,hi),(hi,lo)]:
        m=(own_grade==g); P=h[h.grade==other].reset_index(drop=True); tree=shapely.STRtree(P.geometry.values)
        sel=np.where(m)[0]; near=np.full(len(sel),-1); dist=np.full(len(sel),np.nan)
        for s in range(0,len(sel),400_000):
            ii,dd=tree.query_nearest(pts[sel[s:s+400_000]],return_distance=True,all_matches=False)
            near[s+ii[0]]=ii[1]; dist[s+ii[0]]=dd
        col=f"{lo}{hi}"
        res.loc[sel,f"dist_{col}"]=dist; res.loc[sel,f"nbr_{col}"]=P.area_id.values[np.maximum(near,0)]
res.to_parquet(W+"boundary_inputs.parquet",index=False)
for col in ["CD","BC"]:
    d=res[f"dist_{col}"]; print(f"{col}: requests with a neighbor-grade area within 100m {int((d<=100).sum()):,}, 200m {int((d<=200).sum()):,}, 400m {int((d<=400).sum()):,}")
