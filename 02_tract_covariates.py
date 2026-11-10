from config import MAIN, D, W, U
# Step 2: tract-level covariates for Chicago's 792 tracts: six-indicator hardship index (Nathan & Adams method),
# race/ethnicity, controls (ownership, housing age, density), violent crime and building violations (point-in-tract), redlining (HRI).
import numpy as np, pandas as pd, geopandas as gpd, shapely, glob
A=lambda t: pd.read_csv(D+f"acs/{t}.csv").assign(GEOID=lambda d:d.GEO_ID.str[-11:]).set_index("GEOID")
R,P,C,E,U2,AG,I,O,Y,M=[A(t) for t in ["B03002","B17017","B25014","B15003","B23025","B01001","B19301","B25003","B25035","B19013"]]
chi=pd.read_csv(D+"chicago_tracts.csv",dtype={"GEOID":str}).set_index("GEOID")
x=pd.DataFrame(index=chi.index); x["ca"]=chi.ca; x["pop"]=R.B03002_001E
v=lambda df,num,den: df[num].sum(1)/df[den].replace(0,np.nan)*100
x["crowded"]=v(C,[f"B25014_{i:03d}E" for i in (5,6,7,11,12,13)],"B25014_001E")
x["poverty"]=v(P,["B17017_002E"],"B17017_001E")
x["unemp"]=v(U2,["B23025_005E"],"B23025_003E")
x["nohs"]=v(E,[f"B15003_{i:03d}E" for i in range(2,17)],"B15003_001E")
kids=[f"B01001_{i:03d}E" for i in list(range(3,7))+list(range(27,31))]; old=[f"B01001_{i:03d}E" for i in list(range(20,26))+list(range(44,50))]
x["dependency"]=v(AG,kids+old,"B01001_001E"); x["pci"]=I.B19301_001E.where(I.B19301_001E>0)
x["mhi"]=M.B19013_001E.where(M.B19013_001E>0)
for k,c in [("pct_black","B03002_004E"),("pct_hisp","B03002_012E"),("pct_white","B03002_003E"),("pct_asian","B03002_006E")]: x[k]=v(R,[c],"B03002_001E")
x["own"]=v(O,["B25003_002E"],"B25003_001E"); x["year_built"]=Y.B25035_001E.where(Y.B25035_001E>0)
x["year_built"]=x.year_built.fillna(x.groupby("ca").year_built.transform("median"))
x["analytic"]=(x["pop"]>=100)&x[["crowded","poverty","unemp","nohs","dependency","pci"]].notna().all(1)
a=x[x.analytic]
# Nathan & Adams (1976) standardization: min-max to 0-100 (income inverted), averaged
comp={}
for c in ["crowded","poverty","unemp","nohs","dependency"]: comp[c]=100*(a[c]-a[c].min())/(a[c].max()-a[c].min())
comp["pci"]=100*(a.pci.max()-a.pci)/(a.pci.max()-a.pci.min())
x.loc[a.index,"hardship"]=pd.DataFrame(comp).mean(1)
Z=pd.DataFrame({c:(a[c]-a[c].mean())/a[c].std() for c in ["crowded","poverty","unemp","nohs","dependency"]}).assign(pci=-(a.pci-a.pci.mean())/a.pci.std())
w,vec=np.linalg.eigh(np.cov(Z.T)); pc=Z.values@vec[:,-1]; pc*=np.sign(np.corrcoef(pc,x.loc[a.index,"hardship"])[0,1])
x.loc[a.index,"hardship_pc"]=pc; share_pc=w[-1]/w.sum()
t=gpd.read_file(D+"tracts/cb_2023_17_tract_500k.shp").set_index("GEOID"); x["area_km2"]=t.ALAND.reindex(x.index)/1e6
x["density"]=x["pop"]/x.area_km2
def count_points(lat,lon):
    tt=t.loc[x.index].to_crs(4326); tree=shapely.STRtree(tt.geometry.values); ok=~(np.isnan(lat)|np.isnan(lon))
    pi,ti=tree.query(shapely.points(lon[ok],lat[ok]),predicate="within")
    return pd.Series(tt.index.values[ti]).value_counts().reindex(x.index).fillna(0)
cr=pd.concat([pd.read_csv(f,usecols=["ID","FBI Code","Latitude","Longitude","Year"],dtype={"FBI Code":str}) for f in sorted(glob.glob(U+"Crimes*2001_to_Present*.csv"))]).drop_duplicates("ID")
cr=cr[cr.Year.between(2019,2024)&cr["FBI Code"].isin(["01A","02","03","04A","04B"])]
x["violent_rate"]=1000*count_points(cr.Latitude.values,cr.Longitude.values)/x["pop"]/6
bv=pd.concat([pd.read_csv(f,usecols=["ID","LATITUDE","LONGITUDE"]) for f in sorted(glob.glob(U+"Building_Violations*.csv"))]).drop_duplicates("ID")
x["bviol_rate"]=1000*count_points(bv.LATITUDE.values,bv.LONGITUDE.values)/x["pop"]/6
h=pd.read_excel(D+"hri/Historic Redlining Indicator 2020.xlsx",dtype={"GEOID20":str}).set_index("GEOID20")
x["hri"]=h.HRI2020.reindex(x.index); x["hri_interval"]=h.INTERVAL2020.reindex(x.index)
x.to_csv(W+"tract.csv")
a=x[x.analytic]
print(f"Chicago tracts: {len(x)} | analytic (pop>=100, complete indicators): {len(a)} | first principal component explains {share_pc:.0%} of indicator variance")
print(f"hardship index: mean {a.hardship.mean():.1f}, sd {a.hardship.std():.1f}, range {a.hardship.min():.1f}-{a.hardship.max():.1f} | corr with PC version {a[['hardship','hardship_pc']].corr().iloc[0,1]:.3f}")
print("violent crimes geocoded to tracts:", int(1000*0+(x.violent_rate*x['pop']*6/1000).sum()), "| building violations:", int((x.bviol_rate*x['pop']*6/1000).sum()))
print("tracts with HRI:", int(a.hri.notna().sum()))
print(a[["hardship","pct_black","pct_hisp","poverty","pci","own","year_built","violent_rate","bviol_rate","hri"]].describe().round(1).T[["mean","std","min","max"]].to_string())
