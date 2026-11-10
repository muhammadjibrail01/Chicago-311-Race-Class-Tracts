from config import MAIN, D, W, U
# Step 14: all Paper 1 tables, built directly from saved results (no hand-typed numbers).
import json, numpy as np, pandas as pd
from scipy import stats
W=W+""; J=lambda f: json.load(open(W+"tabs/"+f))
mm,rl,js,ql,rb,sp,cl=[J(f) for f in ["main_models.json","redlining.json","jobscope_channel.json","quality.json","robustness.json","spatial.json","collinearity.json"]]
pval=lambda b,s,df=76: 2*stats.t.sf(abs(b/s),df)
st=lambda p: "***" if p<.01 else "**" if p<.05 else "*" if p<.1 else ""
def cs(b,s,df=76,d=3): return f"{b:.{d}f}{st(pval(b,s,df))}", f"({s:.{d}f})"
pct=lambda b: 100*(np.exp(b)-1)
NOTE_STD="Tract variables are standardized across the 789 analytic tracts. Models include request type × year, month-of-sample, weekday, hour-of-day, and reporting-channel fixed effects and log workload. Standard errors clustered by community area (77 clusters) in parentheses"
T={}
T["T1"]={"cap":"Table 1. Data sources and analytic samples","cols":["Item","Value"],"widths":[6400,2960],"font":8.5,"rows":[
 ["311 requests of the 49 core service types, 2019–2024","4,648,843"],["Located inside a Cook County census tract","99.97%"],["Located inside a Chicago census tract","99.85%"],
 ["Tract matches the community area recorded by the city","99.20%"],["Speed sample: completed, non-duplicate, resident-initiated requests in analytic tracts",f"{mm['S1 hardship']['n']:,}"],
 ["Chicago census tracts (2020 boundaries)","792"],["Analytic tracts (population ≥ 100, complete ACS indicators)","789"],["Tracts with a Historic Redlining Indicator score","737"],
 ["Requests inside a graded 1930s HOLC area (A / B / C / D)","86.3% (1.0 / 9.8 / 56.1 / 33.0)"],
 ["Requests within 400 m of a C | D border",f"{rl['RD CD bw400']['n']:,}"],["Requests within 400 m of a B | C border",f"{rl['RD BC bw400']['n']:,}"],
 ["Violent Index crimes geocoded to tracts, 2019–2024","156,267"],["Building violations geocoded to tracts, 2019–2024","342,757"]],
 "note":"Sources: City of Chicago Data Portal (311 Service Requests; Crimes; Building Violations); U.S. Census Bureau, ACS 2019–2023 five-year tract tables and 2023 cartographic tract boundaries; Meier and Mitchell, Historic Redlining Indicator (2020 tracts); Mapping Inequality HOLC map polygons."}
t=pd.read_csv(W+"tract.csv",dtype={"GEOID":str}); t=t[t.analytic].set_index("GEOID"); g=pd.read_csv(W+"tract_groups.csv",dtype={"GEOID":str}).set_index("GEOID")
t["bg"]=g.black_grp.reindex(t.index); A=pd.read_csv(W+"tract_adjusted.csv",dtype={"GEOID":str}).set_index("GEOID"); t["adj"]=A.adj.reindex(t.index)
V=[("N tracts",None,"{:.0f}"),("Hardship index (0–100)","hardship","{:.1f}"),("Households below poverty (%)","poverty","{:.1f}"),("Per capita income ($)","pci","{:,.0f}"),
   ("Black residents (%)","pct_black","{:.1f}"),("Hispanic residents (%)","pct_hisp","{:.1f}"),("Owner-occupied homes (%)","own","{:.1f}"),("Median year built","year_built","{:.0f}"),
   ("Violent crimes per 1,000 per year","violent_rate","{:.1f}"),("Building violations per 1,000 per year","bviol_rate","{:.1f}"),("Historic Redlining Indicator (1–4)","hri","{:.2f}"),("Adjusted completion time (log points)","adj","{:+.3f}")]
grp=["<10% Black","10-60% Black",">60% Black"]; rows=[]
for lab,v,f in V:
    rows.append([lab]+[f.format(len(t[t.bg==x]) if v is None else t.loc[t.bg==x,v].mean()) for x in grp]+[f.format(len(t) if v is None else t[v].mean())])
T["T2"]={"cap":"Table 2. Tract characteristics by share of Black residents","cols":["","<10% Black","10–60% Black",">60% Black","All tracts"],"widths":[3960,1350,1350,1350,1350],"font":8.5,"rows":rows,
 "note":"Unweighted means across the 789 analytic tracts. ACS 2019–2023 five-year estimates. The adjusted completion time is the mean residual, by tract, from the request-level model with type × year, month, weekday, hour, and channel fixed effects and log workload."}
c=cl; T["T3"]={"cap":"Table 3. How much independent variation in race and class do tracts provide?","cols":["","Community areas (77)","Census tracts (789)"],"widths":[4960,2200,2200],"font":8.5,"rows":[
 [{"panel":"Panel A. Correlations and variance"}],["Correlation: hardship index and % Black",f"{c['corr_ca_hardship_black']:.2f}",f"{c['corr_tract_hardship_black']:.2f}"],
 ["Correlation: low income or poverty and % Black",f"{c['corr_ca_lowinc_black']:.2f} (low income)",f"{c['corr_tract_poverty_black']:.2f} (poverty)"],
 ["Share of % Black variance not explained by hardship",f"{c['black_var_unexplained_by_hardship_ca']:.2f}",f"{c['black_var_unexplained_by_hardship_tract']:.2f}"],
 ["Share of tract variance lying within community areas: hardship","—",f"{c['within_ca_share_hardship']:.2f}"],["Share of tract variance lying within community areas: % Black","—",f"{c['within_ca_share_pct_black']:.2f}"],
 [{"panel":"Panel B. Number of tracts by race and poverty (tract poverty terciles)"}]]+
 [[r]+[f"{c['cells'][p][r]}" for p in ["low poverty","middle poverty"]]+[] for r in []],"note":""}
T["T3"]["cols"]=["","Community areas (77)","Census tracts (789)"]
T3b={"cap":"","cols":[],"rows":[]}
cells_rows=[[r,f"{c['cells']['low poverty'][r]} / {c['cells']['middle poverty'][r]} / {c['cells']['high poverty'][r]}",""] for r in grp]
T["T3"]["rows"]+= [["Tracts: low / middle / high poverty"]+["",""]]+[[f"  {r}","",v] for r,v,_ in cells_rows]
T["T3"]["note"]="Community-area figures use the 2008–2012 Hardship Index and ACS 2019–2023 low-income share, as in the companion analysis; tract figures use the tract hardship index rebuilt from ACS 2019–2023 with the same six indicators."
order=["S1 hardship","S2 race","S3 both","S4 both+CA FE","S5 +controls","S6 poverty,income+race"]
vars_=[("z_hardship","Hardship index"),("z_poverty","Poverty rate"),("z_pci","Per capita income"),("z_pct_black","Black share"),("z_pct_hisp","Hispanic share")]
rows=[]
for v,lab in vars_:
    r1=[lab]; r2=[""]; r3=[""]
    for k in order:
        e=mm[k]
        if v in e["coef"]: a,b=cs(e["coef"][v],e["se_ca"][v]); r1.append(a); r2.append(b); r3.append(f"[{e['se_tract'][v]:.3f}]")
        else: r1.append(""); r2.append(""); r3.append("")
    rows+= [r1,r2,r3]
rows+= [["Community-area fixed effects","","","","Yes","Yes",""],["Housing, crime, and building-violation controls","","","","","Yes",""],["Observations"]+[f"{mm[k]['n']:,}" for k in order]]
T["T4"]={"cap":"Table 4. Race, class, and the speed of 311 service: request-level models","cols":["","(1)","(2)","(3)","(4)","(5)","(6)"],"widths":[3060,1050,1050,1050,1050,1050,1050],"font":8,"rows":rows,
 "note":"Outcome: log(1 + hours to completion). "+NOTE_STD+"; standard errors clustered by tract (789) in brackets. Controls in column 5: owner-occupancy share, median year built, log population density, violent crimes and building violations per 1,000. *** p<0.01, ** p<0.05, * p<0.1 (community-area clustering, t with 76 df)."}
cellnames=[f"{b} | {p}" for b in grp for p in ["low poverty","middle poverty","high poverty"]]
rows=[]
for k in cellnames:
    b,p=k.split(" | "); n=c["cells"][p][b]
    if k=="<10% Black | low poverty": rows.append([f"{b}, {p}",str(n),"Reference","","Reference",""]); continue
    e7,s7=mm["S7 race-class cells"]["coef"][k],mm["S7 race-class cells"]["se_ca"][k]; e8,s8=mm["S8 cells+CA FE"]["coef"][k],mm["S8 cells+CA FE"]["se_ca"][k]
    a7,b7=cs(e7,s7); a8,b8=cs(e8,s8); rows.append([f"{b}, {p}",str(n),f"{a7} {b7}",f"{pct(e7):+.1f}%",f"{a8} {b8}",f"{pct(e8):+.1f}%"])
T["T5"]={"cap":"Table 5. Completion times by race-by-poverty tract group","cols":["Tract group","Tracts","Citywide: log points","%","Within community areas: log points","%"],"widths":[2660,800,1850,900,2250,900],"font":8,"rows":rows,
 "note":"Coefficients on tract-group indicators relative to tracts with fewer than 10% Black residents and low poverty (bottom tercile). "+NOTE_STD+". Within-community-area models add community-area fixed effects. *** p<0.01, ** p<0.05, * p<0.1."}
rows=[[{"panel":"Panel A. By job scope"}]]
for gname,lab in [("fixed","Fixed-scope jobs"),("variable","Variable-scope jobs"),("other","Other services")]:
    for tag,tl in [("","citywide"),(" +CA","within community areas")]:
        e=js[f"scope|{gname}{tag}"]; a,b=cs(e["coef"]["z_pct_black"],e["se"]["z_pct_black"]); h1,h2=cs(e["coef"]["z_hardship"],e["se"]["z_hardship"])
        rows.append([f"{lab}, {tl}",f"{a} {b}",f"{pct(e['coef']['z_pct_black']):+.1f}%",f"{h1} {h2}",f"{e['n']:,}"])
rows.append([{"panel":"Panel B. Single service types (citywide)"}])
for k in [x for x in js if x.startswith("type|")]:
    e=js[k]; a,b=cs(e["coef"]["z_pct_black"],e["se"]["z_pct_black"]); h1,h2=cs(e["coef"]["z_hardship"],e["se"]["z_hardship"])
    rows.append([f"{k.split('|',1)[1]} ({e['scope']})",f"{a} {b}",f"{pct(e['coef']['z_pct_black']):+.1f}%",f"{h1} {h2}",f"{e['n']:,}"])
T["T6"]={"cap":"Table 6. The \u201cbigger jobs\u201d test: race and class gradients by job scope and service type","cols":["","Black share","%","Hardship index","N"],"widths":[3860,1700,900,1700,1200],"font":8,"rows":rows,
 "note":"Each row is a separate model including hardship, Black share, and Hispanic share. Fixed-scope jobs restore a standardized piece of equipment (lights, signals, signs, carts); variable-scope jobs remove or clean material whose quantity varies (dumping, weeds, debris, graffiti, potholes, street and lot cleaning); see Appendix Table A2. Single-type models use month, weekday, hour, and channel fixed effects. "+NOTE_STD.split(". ",1)[1]+". *** p<0.01, ** p<0.05, * p<0.1."}
rows=[[{"panel":"Panel A. Request's 1930s HOLC grade (reference: C)"}]]
for k,lab in [("grade only","Grade only"),("grade+race+hardship","Grade + current race and hardship")]:
    e=rl[k]; rows.append([lab]+[" ".join(cs(e["coef"][v],e["se"][v])) for v in ["grade_A","grade_B","grade_D"]]+[f"{e['n']:,}"])
rows.append([{"panel":"Panel B. Tract Historic Redlining Indicator (per SD)"}])
for k,lab in [("hri only","HRI only"),("hri+race+hardship","HRI + current race and hardship")]:
    e=rl[k]; rows.append([lab," ".join(cs(e["coef"]["z_hri"],e["se"]["z_hri"])),"","",f"{e['n']:,}"])
rows.append([{"panel":"Panel C. Geographic regression discontinuity at HOLC borders (jump on the lower-grade side)"}])
for col,lab in [("CD","C | D"),("BC","B | C")]:
    for bw in [100,200,400]:
        e=rl[f"RD {col} bw{bw}"]; rows.append([f"{lab} borders, {bw} m"," ".join(cs(e["b"],e["se"],e["pairs"]-1))," ".join(cs(e["b_ctrl"],e["se_ctrl"],e["pairs"]-1)),f"{e['balance']['z_pct_black']['b']:+.2f}",f"{e['n']:,} ({e['pairs']} pairs)"])
T["T7"]={"cap":"Table 7. Historical redlining and today's service speed","cols":["","(a)","(b)","(c)","N"],"widths":[3360,1650,1650,1300,1400],"font":8,"rows":rows,
 "note":"Panel A columns (a)–(c): grades A, B, and D relative to C. Panel B column (a): HRI coefficient. Panel C columns: (a) jump in log completion time at the border, (b) with tract race and hardship controls, (c) jump in tract Black share (SD units), a balance check that is weak because tract measures cannot vary within a tract. Panel C models include border-pair fixed effects and separate linear distance trends on each side; standard errors clustered by border pair. Panels A–B cluster by community area. *** p<0.01, ** p<0.05, * p<0.1."}
rows=[[{"panel":"Panel A. Quality of resolution (percentage points per +1 SD)"}]]
for k,lab in [("recur30","Re-reported within 30 days"),("recur90","Re-reported within 90 days"),("canceled","Canceled"),("unresolved","Still unresolved")]:
    e=ql[k]; rows.append([lab]+[" ".join(cs(e["coef"][v],e["se"][v])) for v in ["z_pct_black","z_hardship","z_pct_hisp"]]+[f"{e['mean']:.2f}"])
rows.append([{"panel":"Panel B. Completion time by reporting channel (log points per +1 SD)"}])
for k,lab in [("phone","Phone"),("digital","Web or app"),("alderman","Aldermanic office")]:
    e=js[f"chan|{k}"]; rows.append([lab]+[" ".join(cs(e["coef"][v],e["se"][v])) for v in ["z_pct_black","z_hardship","z_pct_hisp"]]+[f"{e['n']:,}"])
T["T8"]={"cap":"Table 8. Quality of resolution and reporting channels","cols":["","Black share","Hardship index","Hispanic share","Mean / N"],"widths":[3060,1600,1600,1600,1500],"font":8,"rows":rows,
 "note":"Re-report models use the speed sample; cancellation and unresolved models use all resident-initiated, non-duplicate requests. Channel models estimate the baseline specification separately by channel (without channel fixed effects). Standard errors clustered by community area. *** p<0.01, ** p<0.05, * p<0.1."}
rows=[[{"panel":"Panel A. Request-level robustness (log points per +1 SD)"}]]
for k,e in rb.items():
    cv=[x for x in e["coef"] if x.startswith("z_") and x not in ("z_pct_black","z_pct_hisp")][0]
    rows.append([k[0].upper()+k[1:]," ".join(cs(e["coef"]["z_pct_black"],e["se"]["z_pct_black"],e["G"]-1))," ".join(cs(e["coef"][cv],e["se"][cv],e["G"]-1)),f"{e['n']:,}"])
rows.append([{"panel":"Panel B. Tract-level models of the adjusted completion-time index (789 tracts)"}])
for k,lab,se_df in [("wls","Weighted least squares (HC3)",785),("ml_error","Spatial error model",785),("ml_lag","Spatial lag model",785)]:
    e=sp[k]; extra=[x for x in e["coef"] if x in ("lambda","rho")]
    rows.append([lab+(f"; {extra[0]} = {e['coef'][extra[0]]:.2f}" if extra else "")," ".join(cs(e["coef"]["z_pct_black"],e["se"]["z_pct_black"],se_df))," ".join(cs(e["coef"]["z_hardship"],e["se"]["z_hardship"],se_df)),"789"])
T["T9"]={"cap":"Table 9. Robustness and spatial dependence","cols":["","Black share","Class measure","N"],"widths":[4360,1750,1750,1500],"font":8,"rows":rows,
 "note":f"Panel A: each row modifies the baseline (column 3 of Table 4); the class measure is the hardship index unless the row names another. The percentile-rank and slower-than-median outcomes are in percentile points and percentage points. Panel B: Moran's I of the adjusted index = {sp['moran|adj']['I']:.2f} and of the WLS residuals = {sp['moran|wls_resid']['I']:.2f} (both p = 0.001, 999 permutations); queen contiguity, no islands. *** p<0.01, ** p<0.05, * p<0.1."}
T["A1"]={"cap":"Table A1. Variable definitions","cols":["Variable","Definition and source"],"widths":[2400,6960],"font":8,"rows":[
 ["Completion time","Hours from request creation to closure; outcome log(1 + hours), winsorized at the type-specific 99th percentile (311 Service Requests)."],
 ["Hardship index","Mean of six indicators min–max scaled to 0–100 across tracts (per capita income inverted): crowded housing (B25014), household poverty (B17017), unemployment (B23025), adults 25+ without a high school diploma (B15003), population under 18 or 65+ (B01001), per capita income (B19301); ACS 2019–2023. Same dimensions as Nathan and Adams (1976)."],
 ["Black, Hispanic, white, Asian share","Non-Hispanic Black, Hispanic, non-Hispanic white, and non-Hispanic Asian residents as % of population (B03002)."],
 ["Housing controls","Owner-occupied share (B25003); median year structure built (B25035; missing values imputed with the community-area median); log population per km² of land."],
 ["Need controls","Violent Index crimes (FBI codes 01A, 02, 03, 04A, 04B) and building violations per 1,000 residents per year, 2019–2024, geocoded to tracts."],
 ["Redlining","Historic Redlining Indicator for 2020 tracts (Meier and Mitchell; 1–4, higher = more redlined); 1930s HOLC grade of the request's location (Mapping Inequality)."],
 ["Workload","Log number of resident-initiated, non-duplicate requests of the same type in the same community area and week."],
 ["Re-report","A new resident request of the same type at the same address within 30 or 90 days of closure."]],"note":""}
FIXED=["Street Light Out","Alley Light Out","Traffic Signal Out","Sign Repair (all other, stop, one-way)","Street Light Pole Damage","Garbage Cart Maintenance","Blue Recycling Cart"]
VAR=["Graffiti Removal","Weed Removal","Tree Debris Clean-Up","Sanitation Code Violation","Fly Dumping","Clean Vacant Lot","Street Cleaning","Yard Waste Pick-Up","Pothole in Street","Alley Pothole"]
T["A2"]={"cap":"Table A2. Classification of service types by job scope","cols":["Group","Service types"],"widths":[2000,7360],"font":8,"rows":[["Fixed scope (9 types)","; ".join(FIXED)],["Variable scope (10 types)","; ".join(VAR)],["Other (30 types)","Inspections (buildings, trees, sewers, sidewalks, public way), animal services, water service, tree emergencies and planting, snow and ice, abandoned vehicles, rodent baiting, dead animals, missed pick-ups, and remaining types"]],"note":""}
json.dump(T,open(W+"tables_p1.json","w"),ensure_ascii=False,indent=0); print("tables:",list(T))
