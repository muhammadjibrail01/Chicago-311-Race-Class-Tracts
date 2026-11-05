from config import MAIN, D, W, U
# Step 0: prepare inputs: the 10 ACS tract tables saved from the Census API (any file extension), the 2023 tract boundaries,
# the Historic Redlining Indicator (2020 tracts), the Chicago HOLC map, and the list of Chicago tracts.
import os, re, json, glob, zipfile, pandas as pd, geopandas as gpd
for sub in ["acs","tracts","hri","holc"]: os.makedirs(D+sub, exist_ok=True)
found={}
for f in sorted(glob.glob(U+"*")):
    if os.path.isdir(f) or os.path.getsize(f)>50_000_000 or f.lower().endswith((".pdf",".zip",".gpkg",".csv",".xlsx",".shp",".7z",".docx")): continue
    try: d=json.load(open(f))
    except Exception: continue
    if not (isinstance(d,list) and d and isinstance(d[0],list)): continue
    tabs={m.group(1) for c in d[0] for m in [re.match(r"(B\d{5})_",str(c))] if m}
    if len(tabs)!=1: continue
    t=tabs.pop(); df=pd.DataFrame(d[1:],columns=d[0]); df=df.loc[:,~df.columns.duplicated()]
    est=[c for c in df.columns if re.match(r"B\d{5}_\d{3}E$",c)]
    out=df[["GEO_ID","NAME","tract"]+est].copy(); out[est]=out[est].apply(pd.to_numeric,errors="coerce"); out.to_csv(D+f"acs/{t}.csv",index=False); found[t]=len(out)
need=["B03002","B19013","B19301","B17017","B25014","B15003","B23025","B01001","B25003","B25035"]
miss=[t for t in need if t not in found]
if miss: raise SystemExit(f"Missing ACS tables {miss}: save each Census API link from the README into RAW_DIR.")
print("ACS tables:", {k:found[k] for k in need})
z=glob.glob(U+"cb_2023_17_tract_500k.zip")
if z: zipfile.ZipFile(z[0]).extractall(D+"tracts")
elif not glob.glob(D+"tracts/*.shp"): raise SystemExit("Tract boundary file cb_2023_17_tract_500k.zip not found in RAW_DIR.")
hz=glob.glob(U+"*141121*.zip"); hx=glob.glob(U+"Historic Redlining Indicator 2020.xlsx")
if hz:
    zz=zipfile.ZipFile(hz[0]); m=[n for n in zz.namelist() if n.endswith("Historic Redlining Indicator 2020.xlsx")][0]
    open(D+"hri/Historic Redlining Indicator 2020.xlsx","wb").write(zz.read(m))
elif hx: open(D+"hri/Historic Redlining Indicator 2020.xlsx","wb").write(open(hx[0],"rb").read())
else: raise SystemExit("Historic Redlining Indicator (ICPSR 141121) not found in RAW_DIR.")
hg=glob.glob(U+"mappinginequality*.gpkg")
if not hg: raise SystemExit("Mapping Inequality file mappinginequality.gpkg not found in RAW_DIR.")
h=gpd.read_file(hg[0]); h=h[h.city.astype(str).str.contains("Chicago",case=False)&(h.state.astype(str).str.upper()=="IL")].copy(); h["grade"]=h.grade.str.strip()
h.to_file(D+"holc/holc_chicago.gpkg",driver="GPKG")
t=gpd.read_file(D+"tracts/cb_2023_17_tract_500k.shp"); cook=t[t.COUNTYFP=="031"].to_crs(26916)
ca=gpd.read_file(MAIN+"ca.gpkg")[["ca","name","geometry"]].to_crs(26916)
pts=cook.copy(); pts["geometry"]=cook.representative_point(); j=gpd.sjoin(pts,ca,predicate="within",how="inner")
j[["GEOID","ca","name"]].to_csv(D+"chicago_tracts.csv",index=False)
print(f"Cook tracts {len(cook)} | Chicago tracts {len(j)} | HOLC Chicago areas {len(h)} | HRI file ready")
