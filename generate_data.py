"""Generate a hypothetical, deliberately messy agribusiness dataset (rice/maize, 3 seasons).
Every defect is injected at a known rate so cleaning can be measured against ground truth."""
import numpy as np, pandas as pd
rng = np.random.default_rng(42)

N_FIELDS, SEASONS = 600, [2022, 2023, 2024]
regions = ["North", "South", "East", "West"]
soil_types = ["clay", "loam", "sandy", "silt"]
varieties = {"rice": ["R-101", "R-205", "R-330"], "maize": ["M-12", "M-47", "M-90"]}

fields = pd.DataFrame({
    "field_id": [f"F{i:04d}" for i in range(N_FIELDS)],
    "region": rng.choice(regions, N_FIELDS, p=[.3, .25, .25, .2]),
    "crop": rng.choice(["rice", "maize"], N_FIELDS, p=[.55, .45]),
    "soil_type": rng.choice(soil_types, N_FIELDS, p=[.3, .35, .2, .15]),
    "area_ha": np.round(rng.lognormal(0.2, 0.55, N_FIELDS), 2),
    "soil_ph": np.round(rng.normal(6.4, 0.6, N_FIELDS).clip(4.5, 8.5), 2),
    "soil_om_pct": np.round(rng.normal(2.1, 0.7, N_FIELDS).clip(0.4, 5), 2),
    "soil_n_kg_ha": np.round(rng.normal(240, 45, N_FIELDS).clip(80, 420), 0),
})
region_rain = {"North": 1.0, "South": 1.15, "East": 0.9, "West": 0.8}

rows = []
for s in SEASONS:
    season_shock = {2022: 0.0, 2023: -0.8, 2024: 0.3}[s]       # 2023 = drought-ish year
    for _, f in fields.iterrows():
        planting_doy = int(rng.normal(170, 8))
        variety = rng.choice(varieties[f.crop])
        rain = max(250, rng.normal(620, 110) * region_rain[f.region] * (1 + 0.12 * season_shock))
        tmax = rng.normal(31.5, 1.2) + 0.5 * -season_shock
        heat_days = max(0, int(rng.poisson(max(0.5, 7 - 4 * season_shock + (tmax - 31.5) * 2))))
        dry_spell = max(0, int(rng.poisson(max(1, 9 + 6 * -season_shock))))
        ndvi_peak = np.clip(rng.normal(0.74, 0.06) + 0.04 * season_shock - 0.004 * heat_days, 0.35, 0.92)
        ndvi_mid = np.clip(ndvi_peak - rng.normal(0.08, 0.03), 0.3, 0.9)
        n_applied = max(0, rng.normal(110, 25))
        irrig = max(0, rng.normal(260, 70) - 0.18 * (rain - 620))
        base = 4.4 if f.crop == "rice" else 5.6
        y = (base + 4.2 * (ndvi_peak - 0.74) + 0.0011 * (rain - 620) - 0.045 * heat_days
             - 0.03 * dry_spell + 0.006 * (n_applied - 110) + 0.0015 * (f.soil_n_kg_ha - 240)
             - 0.35 * abs(f.soil_ph - 6.5) + 0.12 * (f.soil_om_pct - 2.1)
             + 0.0008 * (180 - planting_doy) + rng.normal(0, 0.35))
        rows.append(dict(field_id=f.field_id, season=s, variety=variety, planting_doy=planting_doy,
                         rain_mm=round(rain, 1), tmax_mean_c=round(tmax, 2), heat_stress_days=heat_days,
                         max_dry_spell_days=dry_spell, ndvi_peak=round(ndvi_peak, 3),
                         ndvi_mid=round(ndvi_mid, 3), n_applied_kg_ha=round(n_applied, 1),
                         irrigation_mm=round(irrig, 1), yield_t_ha=round(max(0.8, y), 2)))
clean = fields.merge(pd.DataFrame(rows), on="field_id")
clean.to_csv("agri_clean_truth.csv", index=False)

# ---------------- inject real-world defects ----------------
d = clean.copy()
n = len(d)
def idx(frac): return rng.choice(n, int(frac * n), replace=False)

# 1. Missing values with different mechanisms
d.loc[idx(0.12), "ndvi_mid"] = np.nan                                   # cloud cover (MCAR-ish)
cloudy = d.index[(d.region == "South") & (d.season == 2023)]
d.loc[rng.choice(cloudy, int(0.35 * len(cloudy)), replace=False), "ndvi_peak"] = np.nan  # monsoon clouds (MAR)
d.loc[idx(0.09), "soil_ph"] = np.nan
d.loc[idx(0.07), "n_applied_kg_ha"] = np.nan
# farmers with low inputs under-report -> MNAR-like: small applications are more often blank
low = d.index[d.n_applied_kg_ha < 85]
d.loc[rng.choice(low, int(0.4 * len(low)), replace=False), "n_applied_kg_ha"] = np.nan
d.loc[idx(0.05), "irrigation_mm"] = np.nan
d.loc[idx(0.04), "variety"] = np.nan
# 2. Unit/entry errors
d["area_unit"] = "ha"
acre = idx(0.06); d.loc[acre, "area_ha"] = np.round(d.loc[acre, "area_ha"] * 2.471, 2); d.loc[acre, "area_unit"] = "ac"
d.loc[idx(0.015), "yield_t_ha"] *= 10                                   # kg/ha-style decimal slip
d.loc[idx(0.01), "yield_t_ha"] = 0                                      # zero-yield placeholder
d.loc[idx(0.01), "rain_mm"] = -999                                      # sensor sentinel
i=idx(0.01); d.loc[i, "soil_ph"] = d.loc[i, "soil_ph"] * 10                 # missing decimal point
i=idx(0.02); d.loc[i, "tmax_mean_c"] = d.loc[i, "tmax_mean_c"] + rng.choice([12, -14], len(i))  # sensor drift
# 3. Inconsistent categoricals
d["crop"] = d["crop"].where(rng.random(n) > 0.08, d["crop"].str.title())
d["region"] = d["region"].where(rng.random(n) > 0.05, d["region"].str.upper() + " ")
d["soil_type"] = d["soil_type"].replace({"loam": "loam"}); 
m = rng.random(n) < 0.04; d.loc[m, "soil_type"] = d.loc[m, "soil_type"].replace({"clay": "Clayey", "sandy": "sand"})
# 4. Duplicates
dups = d.sample(int(0.025 * n), random_state=1); d = pd.concat([d, dups], ignore_index=True)
d = d.sample(frac=1, random_state=3).reset_index(drop=True)
d.to_csv("agri_raw_messy.csv", index=False)
print("raw rows:", len(d), "| clean-truth rows:", len(clean))
print(d.isna().mean().round(3).sort_values(ascending=False).head(8))
