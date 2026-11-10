from config import MAIN, D, W, U
# Step 1: assign each core 311 request to its 2020 census tract and to its 1930s HOLC area (grade), by point-in-polygon.
import numpy as np, pandas as pd, geopandas as gpd, shapely, pyarrow.parquet as pq
r=pq.read_table(MAIN+"core.parquet",columns=["ca","lat","lon"]).to_pandas()
t=gpd.read_file(D+"tracts/cb_2023_17_tract_500k.shp"); t=t[t.COUNTYFP=="031"].to_crs(4326).reset_index(drop=True)
h=gpd.read_file(D+"holc/holc_chicago.gpkg").to_crs(4326).reset_index(drop=True)
def assign(polys, lat, lon, chunk=500_000):
    tree=shapely.STRtree(polys.geometry.values); out=np.full(len(lat),-1,dtype=np.int32)
    for s in range(0,len(lat),chunk):
        la,lo=lat[s:s+chunk],lon[s:s+chunk]; ok=~(np.isnan(la)|np.isnan(lo))
        pts=shapely.points(lo[ok],la[ok]); pi,ti=tree.query(pts,predicate="within")
        idx=np.where(ok)[0]; tmp=np.full(ok.sum(),-1,dtype=np.int32); tmp[pi[::-1]]=ti[::-1]   # first match wins
        out[s+idx]=tmp
    return out
lat=r.lat.values.astype(float); lon=r.lon.values.astype(float)
ti=assign(t,lat,lon); hi=assign(h,lat,lon)
geo=np.where(ti>=0, t.GEOID.values[np.maximum(ti,0)], None)
grade=np.where(hi>=0, h.grade.values[np.maximum(hi,0)], None); area=np.where(hi>=0, h.area_id.values[np.maximum(hi,0)], None)
out=pd.DataFrame({"GEOID":geo,"holc_area":area,"holc_grade":grade})
out.to_parquet(W+"core_locations.parquet",index=False)
chi=pd.read_csv(D+"chicago_tracts.csv",dtype={"GEOID":str})
m=out.GEOID.isin(chi.GEOID); tca=out.GEOID.map(chi.set_index("GEOID").ca)
print(f"requests: {len(r):,} | with coordinates: {(~np.isnan(lat)).mean():.2%} | in a Cook tract: {(ti>=0).mean():.2%} | in a Chicago tract: {m.mean():.2%}")
print(f"tract's community area equals request's recorded community area: {(tca[m]==r.ca[m]).mean():.2%}")
print(f"in a graded HOLC area: {pd.Series(grade).notna().mean():.2%} | by grade:", pd.Series(grade).value_counts(normalize=True).round(3).to_dict())
