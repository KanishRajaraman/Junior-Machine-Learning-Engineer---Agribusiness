"""Week 3 - Part A: data-side checks that feed the modelling plan (units, spellings, spatial and temporal support)."""
import json, numpy as np, pandas as pd, difflib
from rapidfuzz import fuzz
R = {}
raw = pd.read_csv("agri_raw_messy.csv"); mr = pd.read_csv("agri_model_ready.csv")
KEY = ["field_id", "season"]
d = raw.drop_duplicates(KEY).copy()
R["rows"] = {"raw": len(raw), "dedup": len(d)}

# ---------- 1. UNITS: acres entered as hectares
ac = d[d.area_unit == "ac"]
R["ac_rows"] = int(len(ac)); R["ac_share"] = round(len(ac) / len(d) * 100, 1)
d["area_fixed"] = np.where(d.area_unit == "ac", d.area_ha / 2.471, d.area_ha)
# cross-season consistency check WITHOUT using the unit flag: area of a field should be the same each season
g = d.groupby("field_id").area_ha
d["area_med_field"] = g.transform("median"); d["n_seas"] = g.transform("count")
d["ratio"] = d.area_ha / d.area_med_field
# robust: compare to the field's lowest reported area (acre values are always larger)
d["ratio_min"] = d.area_ha / g.transform("min")
cand = (d.n_seas >= 2) & (d.ratio_min.between(2.35, 2.60))
R["xseason_detect"] = {"flagged": int(cand.sum()), "tp": int((cand & (d.area_unit == "ac")).sum()),
                       "fp": int((cand & (d.area_unit != "ac")).sum()), "fn": int((~cand & (d.area_unit == "ac")).sum()),
                       "ac_with_single_season": int(((d.area_unit == "ac") & (d.n_seas < 2)).sum())}
# fields where 2 of 3 seasons are acres? handle: use min -> if two seasons both ac, min is also ac -> missed
both = d[d.area_unit == "ac"].groupby("field_id").size()
R["xseason_detect"]["fields_with_2plus_ac_seasons"] = int((both >= 2).sum())
# what if the flag is absent: distribution-based - fraction of area above 4 ha
R["area_raw_mean"] = round(d.area_ha.mean(), 3); R["area_fixed_mean"] = round(d.area_fixed.mean(), 3)
R["area_raw_p99"] = round(d.area_ha.quantile(.99), 2); R["area_fixed_p99"] = round(d.area_fixed.quantile(.99), 2)
# production distortion: sum(area*yield) over rows with a valid yield
v = d[d.yield_t_ha.between(0.5, 15)]
prod_raw = (v.area_ha * v.yield_t_ha).sum(); prod_fix = (v.area_fixed * v.yield_t_ha).sum()
R["production_t"] = {"raw": round(prod_raw, 0), "fixed": round(prod_fix, 0), "overstatement_pct": round((prod_raw / prod_fix - 1) * 100, 1)}
# acre rows: how much each region's tonnage is overstated
v2 = v.assign(pr=v.area_ha * v.yield_t_ha, pf=v.area_fixed * v.yield_t_ha)
rg = v2.assign(region=v2.region.str.strip().str.title()).groupby("region")[["pr", "pf"]].sum()
R["production_by_region_overstatement_pct"] = ((rg.pr / rg.pf - 1) * 100).round(1).to_dict()

# ---------- 2. SPELLINGS
for c in ["crop", "region", "soil_type", "variety"]:
    R[f"spell_{c}"] = d[c].value_counts(dropna=False).to_dict()
n_raw_dummies = sum(d[c].nunique() for c in ["crop", "region", "soil_type"])
R["dummy_columns"] = {"raw_crop_region_soil": int(n_raw_dummies), "clean": 2 + 4 + 4}
# share of rows with a non-canonical spelling
noncanon = ((d.crop != d.crop.str.strip().str.lower()) | (d.region != d.region.str.strip().str.title()) | d.soil_type.isin(["Clayey", "sand"]))
R["rows_with_any_bad_spelling"] = {"n": int(noncanon.sum()), "pct": round(noncanon.mean() * 100, 1)}
# fuzzy similarity examples
vocab = ["clay", "loam", "sandy", "silt"]
ex = {}
for w in ["Clayey", "sand", "clay ", "loamy", "slit", "sandy loam"]:
    sc = {v_: round(fuzz.ratio(w.strip().lower(), v_), 1) for v_ in vocab}
    ex[w] = sc
R["fuzzy_examples"] = ex
# india-style names (illustrative, not in dataset)
districts = ["Thanjavur", "Tiruchirappalli", "Kancheepuram", "Nagapattinam", "Tiruvarur"]
vars_ = ["Tanjore", "Thanjavur ", "THANJAVUR", "Trichy", "Tiruchirapalli", "Tiruchchirappalli", "Kanchipuram", "Kancheepuram", "Nagapatinam"]
fm = {}
for w in vars_:
    best = max(districts, key=lambda x: fuzz.WRatio(w.strip().lower(), x.lower()))
    fm[w] = (best, round(fuzz.WRatio(w.strip().lower(), best.lower()), 1), round(fuzz.ratio(w.strip().lower(), best.lower()), 1))
R["fuzzy_district_demo"] = fm

# ---------- 3. SPATIAL SUPPORT: field size vs pixel footprint
fa = mr.drop_duplicates("field_id")  # model-ready area_ha has been unit-corrected
area = fa.area_ha
pix = {"Sentinel-2 (10 m)": 0.01, "Landsat 8/9 (30 m)": 0.09, "MODIS (250 m)": 6.25, "CHIRPS (0.05 deg ~5.5 km)": 30.0 * 30.0 * 0 + 5.5 * 5.5 * 100,
       "ERA5-Land (0.1 deg ~9 km)": 9 * 9 * 100, "NASA POWER (0.5 deg ~55 km)": 55 * 55 * 100}
fp = {}
for k, ha in pix.items():
    fp[k] = {"pixel_ha": round(ha, 2), "median_field_pct_of_pixel": round(float(area.median() / ha * 100), 3),
             "fields_larger_than_pixel_pct": round(float((area >= ha).mean() * 100), 1),
             "fields_per_pixel_at_median_size": round(float(ha / area.median()), 1)}
R["footprints"] = fp
R["field_area"] = {"n_fields": int(len(area)), "median": round(float(area.median()), 2), "p10": round(float(area.quantile(.1)), 2),
                   "p90": round(float(area.quantile(.9)), 2), "max": round(float(area.max()), 2)}
# pure-pixel feasibility for Sentinel-2: field must hold a 10 m pixel after a 10 m inner buffer: square field side >= 30 m => 0.09 ha
R["s2_pure_pixel_ok_pct"] = round(float((area >= 0.09).mean() * 100), 1)
R["landsat_pure_pixel_ok_pct"] = round(float((area >= (30 * 3) ** 2 / 1e4).mean() * 100), 1)   # 30 m pixel + 30 m buffer each side -> 90x90 m = 0.81 ha
R["modis_pure_pixel_ok_pct"] = round(float((area >= (250 * 2) ** 2 / 1e4).mean() * 100), 1)    # 250 m pixel + 125 m buffer -> 500x500 m? conservative 25 ha
# GPS offset: a 10 m shift on a square 1.2 ha field (side ~110 m)
side = np.sqrt(area.median() * 1e4); 
R["gps_offset_example"] = {"median_field_side_m": round(float(side), 0), "overlap_pct_if_shifted_10m": round(float(max(0, (side - 10)) / side * 100), 1),
                           "overlap_pct_if_shifted_30m_diag": round(float((max(0, side - 30) / side) ** 2 * 100), 1), "overlap_pct_if_shifted_30m_axis": round(float(max(0, side - 30) / side * 100), 1)}
json.dump(R, open("results_A.json", "w"), indent=1, default=str)
print(json.dumps(R, indent=1, default=str))
