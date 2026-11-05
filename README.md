# Replication package: "Race or Class? Neighborhood Composition, Historical Redlining, and the Speed of 311 Service in Chicago"

This package reproduces every table and figure in Paper 1 from public data. **Tested end to end on 1 October 2026** in an empty folder (Python 3.12, Linux, 4 GB RAM, one CPU): all 15 steps ran in about 22 minutes, `check_results.py` reported 29 of 29 key numbers matching the paper, and all 541 table cells were identical. It builds on the main-paper package
(`chicago311_replication`), which constructs the cleaned 311 request file; run that package first.

## 1. Before you start
1. Run the main-paper package to completion (`python run_all.py` in `chicago311_replication`). Paper 1 uses its
   `work/core.parquet`, `work/ca.csv`, and `work/ca.gpkg`.
2. Install Python 3.10+ and run `pip install -r requirements.txt` in this folder.
3. If the two packages are not side by side, set `MAIN_WORK` in `config.py` to the main package's `work` folder.

## 2. Put the Paper 1 inputs in `raw_data/`
| Input | Where to get it | File(s) |
|---|---|---|
| 10 ACS 2019–2023 tract tables | Open each Census API link below in a browser and save the page (any file name or extension) | e.g. `acs5.json` … |
| 2023 tract boundaries (Illinois) | https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_17_tract_500k.zip | `cb_2023_17_tract_500k.zip` |
| Historic Redlining Indicator | https://doi.org/10.3886/E141121V3 (free ICPSR account) | `ICPSR_141121-V3*.zip` |
| 1930s HOLC map polygons | https://dsl.richmond.edu/panorama/redlining/data | `mappinginequality.gpkg` |
| Crimes, Building Violations | Same files as the main package (Chicago Data Portal) | `Crimes*2001_to_Present*.csv`, `Building_Violations*.csv` |

Census API links (all Cook County tracts; table name after `group(`):
`https://api.census.gov/data/2023/acs/acs5?get=NAME,group(TABLE)&for=tract:*&in=state:17&in=county:031`
with TABLE = B03002, B19013, B19301, B17017, B25014, B15003, B23025, B01001, B25003, B25035.

## 3. Run and check
`python run_all.py` runs all 15 steps in order (about 40–60 minutes; logs in `work/logs/`; resume with `--from N`).
`python check_results.py` compares 29 key numbers with the paper and every cell of the 11 tables with
`expected/tables_in_paper.json`. All lines should read PASS. The Moran's I permutation p-values are not seeded and can
vary slightly; every other random step is fixed. If a step reports that the system stopped it, the computer ran out of
memory: close other programs and resume with `--from N` (the scripts need about 3 GB of free RAM).

## 4. What each step produces
| Step | Script | Produces | In the paper |
|---|---|---|---|
| 00 | `00_prepare_inputs.py` | ACS tables, boundaries, HRI, Chicago HOLC map, list of Chicago tracts | §3 |
| 01 | `01_assign_locations.py` | Each request's tract and 1930s HOLC area (point-in-polygon) | §3.2, Table 1 |
| 02 | `02_tract_covariates.py` | Tract hardship index, race shares, controls, crime and building-violation rates, HRI | §3.3, Table 2, Table A1 |
| 03 | `03_collinearity.py` | Race–class correlations, within-area variance, race-by-poverty cells | Table 3 |
| 04 | `04_main_models.py` | Main request-level models (CA- and tract-clustered SEs) | Table 4, Figure 2 |
| 05 | `05_cells.py` | Race-by-poverty group models | Table 5, Figure 3 |
| 06 | `06_boundaries.py` | Distances to adjacent-grade HOLC borders | Appendix B |
| 07 | `07_redlining.py` | HOLC grade, HRI, and boundary-discontinuity models | Table 7, Figure 5 |
| 08 | `08_jobscope_channel.py` | Job-scope test, single types, channels | Tables 6 and 8, Figures 4 and 6 |
| 09 | `09_quality.py` | Re-report, cancellation, unresolved | Table 8, Figure 6 |
| 10 | `10_robustness.py` | Robustness battery | Table 9, Figure 7 |
| 11 | `11_spatial.py` | Tract adjusted index, Moran's I, spatial error and lag models | Table 9, Figure 1(d) |
| 12 | `12_figures.py` | Figures 1–4, 6, 7 | Figures |
| 13 | `13_rd_plot.py` | Figure 5 (border plots) | Figure 5 |
| 14 | `14_tables.py` | All tables → `work/tables_p1.json` | Tables 1–9, A1–A2 |

`common1.py` loads requests with tract variables; `fe.py` is the fixed-effects estimator with clustered standard errors
(validated against pyfixest in the main package). Figure files are named in the order they were created; the paper's
figure numbers are: fig1 → 1, fig3_models → 2, fig2_race_class_grid → 3, fig4 → 4, fig5_rd → 5, fig7 → 6, fig6 → 7.

## 5. Disclosure
AI-assisted tools were used solely for grammar editing and correcting minor coding errors. All data processing, analytical decisions, and interpretations remain the author's own.
