"""
Week 2 - Data preprocessing & feature engineering pipeline for crop-yield forecasting.
Run:  python agri_pipeline.py     (needs agri_raw_messy.csv, agri_clean_truth.csv from generate_data.py)
Outputs: results.json + figures/*.png
"""
import json, warnings, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest, RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import mutual_info_regression
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
warnings.filterwarnings("ignore"); np.random.seed(0)
import os; os.makedirs("figures", exist_ok=True)
R = {}                                     # collected results for the report
GREEN, LIGHT, GREY = "#2E5E3E", "#4F9D69", "#9AA5A0"
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.spines.top": False, "axes.spines.right": False})

raw   = pd.read_csv("agri_raw_messy.csv")
truth = pd.read_csv("agri_clean_truth.csv")
KEY = ["field_id", "season"]

# ------------------------------------------------------------------ 1. AUDIT
R["audit"] = {
    "rows_raw": len(raw), "duplicate_rows": int(raw.duplicated(KEY).sum()),
    "missing_pct": (raw.isna().mean() * 100).round(1)[lambda s: s > 0].sort_values(ascending=False).to_dict(),
    "acre_rows": int((raw.area_unit == "ac").sum()),
    "crop_spellings": sorted(raw.crop.unique().tolist()),
    "region_spellings": sorted(raw.region.unique().tolist()),
    "soil_spellings": sorted(raw.soil_type.unique().tolist()),
}

# ------------------------------------------------------------------ 2. CLEANING
d = raw.drop_duplicates(KEY).copy()
# 2a categoricals
d["crop"] = d.crop.str.strip().str.lower()
d["region"] = d.region.str.strip().str.title()
d["soil_type"] = d.soil_type.str.strip().str.lower().replace({"clayey": "clay", "sand": "sandy"})
# 2b units
d["area_ha"] = np.where(d.area_unit == "ac", d.area_ha / 2.471, d.area_ha).round(2)
d = d.drop(columns="area_unit")
# 2c domain-rule checks (hard physical limits) -> set to NaN, keep a log
rules = {"rain_mm": (50, 2500), "soil_ph": (3.5, 9.5), "tmax_mean_c": (15, 45),
         "ndvi_peak": (0.1, 0.95), "ndvi_mid": (0.1, 0.95), "n_applied_kg_ha": (0, 400),
         "irrigation_mm": (0, 1500), "yield_t_ha": (0.5, 15)}
rule_log = {}
for c, (lo, hi) in rules.items():
    bad = d[c].notna() & ~d[c].between(lo, hi)
    rule_log[c] = int(bad.sum()); d.loc[bad, c] = np.nan
R["rule_violations"] = rule_log
# rows with no usable label cannot be used for supervised learning
n_before = len(d); d = d[d.yield_t_ha.notna()].reset_index(drop=True)
R["rows_dropped_no_label"] = n_before - len(d)
R["rows_after_cleaning"] = len(d)

# ------------------------------------------------------------------ 3. OUTLIER DETECTION (benchmarked against known truth)
rawd = raw.drop_duplicates(KEY).merge(truth[KEY + ["tmax_mean_c", "yield_t_ha"]], on=KEY, suffixes=("", "_true"))
rawd["region"] = rawd.region.str.strip().str.title()
def prf(flag, truth_flag):
    tp = int((flag & truth_flag).sum()); fp = int((flag & ~truth_flag).sum()); fn = int((~flag & truth_flag).sum())
    return {"flagged": int(flag.sum()), "tp": tp, "fp": fp, "fn": fn,
            "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / max(tp + fn, 1), 3)}
# tmax_mean_c: injected sensor drift (+12 / -14 C)
t = rawd.tmax_mean_c; tt = (rawd.tmax_mean_c - rawd.tmax_mean_c_true).abs() > 5
z = (t - t.mean()) / t.std()
q1, q3 = t.quantile([.25, .75]); iqr = q3 - q1
mad = (t - t.median()).abs().median(); mz = 0.6745 * (t - t.median()) / mad
grp = rawd.groupby(["region"]).tmax_mean_c.transform("median")
iso = IsolationForest(contamination=0.02, random_state=0).fit_predict(rawd[["tmax_mean_c", "rain_mm", "heat_stress_days"]].clip(lower=0)) == -1
R["outliers_tmax"] = {"z_score_3": prf(z.abs() > 3, tt), "iqr_1.5": prf((t < q1 - 1.5 * iqr) | (t > q3 + 1.5 * iqr), tt),
                      "modified_z_3.5": prf(mz.abs() > 3.5, tt), "isolation_forest": prf(pd.Series(iso, index=rawd.index), tt),
                      "n_true_anomalies": int(tt.sum())}
# yield: decimal slip (x10) and zero placeholder
y = rawd.yield_t_ha; yt = (rawd.yield_t_ha - rawd.yield_t_ha_true).abs() > 1
R["outliers_yield"] = {"z_score_3": prf(((y - y.mean()) / y.std()).abs() > 3, yt),
                       "iqr_1.5": prf((y < y.quantile(.25) - 1.5 * (y.quantile(.75) - y.quantile(.25))) |
                                      (y > y.quantile(.75) + 1.5 * (y.quantile(.75) - y.quantile(.25))), yt),
                       "domain_rule_0.5-15": prf(~y.between(0.5, 15), yt), "n_true_anomalies": int(yt.sum())}
# apply: tmax drift flagged with modified z within the cleaned frame -> NaN (then imputed)
mzc = 0.6745 * (d.tmax_mean_c - d.tmax_mean_c.median()) / (d.tmax_mean_c - d.tmax_mean_c.median()).abs().median()
R["tmax_flagged_applied"] = int((mzc.abs() > 3.5).sum()); d.loc[mzc.abs() > 3.5, "tmax_mean_c"] = np.nan
# winsorise the remaining continuous inputs at 1st/99th percentile (keeps rows, tames tails)
for c in ["rain_mm", "irrigation_mm", "n_applied_kg_ha", "area_ha"]:
    lo, hi = d[c].quantile([.01, .99]); d[c] = d[c].clip(lo, hi)

# ------------------------------------------------------------------ 4. MISSING-VALUE STRATEGY BENCHMARK
R["missing_after_cleaning_pct"] = (d.isna().mean() * 100).round(1)[lambda s: s > 0].sort_values(ascending=False).to_dict()
num_in = ["rain_mm", "tmax_mean_c", "heat_stress_days", "max_dry_spell_days", "ndvi_peak", "ndvi_mid", "n_applied_kg_ha",
          "irrigation_mm", "soil_ph", "soil_om_pct", "soil_n_kg_ha", "area_ha", "planting_doy"]   # NOTE: yield excluded -> no leakage
d["_grp"] = d.crop + "|" + d.region + "|" + d.season.astype(str)
def impute_eval(col, frac=0.15, seed=0):
    rs = np.random.default_rng(seed); obs = d.index[d[col].notna()]
    mask = rs.choice(obs, int(frac * len(obs)), replace=False)
    X = d[num_in].copy(); X["_g"] = d["_grp"]; true = X.loc[mask, col].copy(); X.loc[mask, col] = np.nan
    out = {}
    out["mean"] = SimpleImputer(strategy="mean").fit_transform(X[num_in])[:, num_in.index(col)]
    out["median"] = SimpleImputer(strategy="median").fit_transform(X[num_in])[:, num_in.index(col)]
    gm = X.groupby("_g")[col].transform("median").fillna(X[col].median()); out["group_median"] = np.where(X[col].isna(), gm, X[col])
    sc = StandardScaler().fit(X[num_in]); Xs = pd.DataFrame(sc.transform(X[num_in]), columns=num_in)
    k = KNNImputer(n_neighbors=7).fit_transform(Xs); out["knn_7"] = sc.inverse_transform(k)[:, num_in.index(col)]
    it = IterativeImputer(max_iter=10, random_state=0).fit_transform(X[num_in]); out["iterative_mice"] = it[:, num_in.index(col)]
    pos = [X.index.get_loc(m) for m in mask]
    return {m: round(float(np.sqrt(np.mean((v[pos] - true.values) ** 2))), 4) for m, v in out.items()}
R["imputation_rmse"] = {c: impute_eval(c) for c in ["ndvi_mid", "soil_ph", "n_applied_kg_ha", "ndvi_peak"]}
R["imputation_rmse_scale"] = {c: round(float(d[c].std()), 3) for c in R["imputation_rmse"]}
# final choice: missing flags + iterative imputer for numerics, mode for categoricals
for c in ["n_applied_kg_ha", "ndvi_peak", "ndvi_mid", "irrigation_mm", "soil_ph"]:
    d[f"{c}_missing"] = d[c].isna().astype(int)
d["variety"] = d.variety.fillna("unknown")
d[num_in] = IterativeImputer(max_iter=10, random_state=0).fit_transform(d[num_in])
d = d.drop(columns="_grp")

# ------------------------------------------------------------------ 5. FEATURE ENGINEERING
WATER_DEMAND = {"rice": 900.0, "maize": 600.0}                      # assumed seasonal crop water demand (mm) - planning assumption
def engineer(df):
    f = df.copy()
    f["ndvi_drop"] = f.ndvi_peak - f.ndvi_mid                       # senescence/stress between mid-season and peak
    f["ndvi_peak_anom"] = f.ndvi_peak - f.groupby(["crop", "region", "season"]).ndvi_peak.transform("mean")   # vs peer fields, same season (label-free)
    f["total_water_mm"] = f.rain_mm + f.irrigation_mm
    f["water_balance_mm"] = f.total_water_mm - f.crop.map(WATER_DEMAND)
    f["heat_x_dry"] = f.heat_stress_days * f.max_dry_spell_days / 10    # compound stress
    f["n_supply_kg_ha"] = f.n_applied_kg_ha + 0.1 * f.soil_n_kg_ha
    f["n_use_ratio"] = f.n_applied_kg_ha / f.soil_n_kg_ha
    f["ph_dist_opt"] = (f.soil_ph - 6.5).abs()
    f["plant_delay_d"] = f.planting_doy - f.groupby("crop").planting_doy.transform("median")
    f["log_area"] = np.log1p(f.area_ha)
    f["ph_band"] = pd.cut(f.soil_ph, [0, 5.5, 6.5, 7.5, 14], labels=["acidic", "slightly_acidic", "neutral", "alkaline"]).astype(str)
    f["plant_window"] = pd.cut(f.plant_delay_d, [-99, -7, 7, 99], labels=["early", "normal", "late"]).astype(str)
    f["n_band"] = pd.qcut(f.n_applied_kg_ha, 4, labels=["Q1_low", "Q2", "Q3", "Q4_high"]).astype(str)
    return f
E = engineer(d)
cat_cols = ["crop", "region", "soil_type", "variety", "ph_band", "plant_window", "n_band"]
E_ohe = pd.get_dummies(E, columns=cat_cols, dtype=int)
base_num = num_in + [c for c in E.columns if c.endswith("_missing")]
eng_new = ["ndvi_drop", "ndvi_peak_anom", "total_water_mm", "water_balance_mm", "heat_x_dry", "n_supply_kg_ha",
           "n_use_ratio", "ph_dist_opt", "plant_delay_d", "log_area"]
ohe_cols = [c for c in E_ohe.columns if any(c.startswith(k + "_") for k in cat_cols)]
all_feats = base_num + eng_new + ohe_cols
R["n_features"] = {"raw_numeric": len(base_num), "engineered_new": len(eng_new), "one_hot": len(ohe_cols), "total_candidate": len(all_feats)}

# ------------------------------------------------------------------ 6. FEATURE SELECTION
X, yv = E_ohe[all_feats], E_ohe.yield_t_ha
corr = X.corr().abs(); upper = corr.where(np.triu(np.ones(corr.shape), 1).astype(bool))
drop_corr = [c for c in upper.columns if (upper[c] > 0.90).any()]
R["dropped_collinear"] = drop_corr
Xc = X.drop(columns=drop_corr)
mi = pd.Series(mutual_info_regression(Xc, yv, random_state=0), index=Xc.columns).sort_values(ascending=False)
keep_mi = mi[mi > 0.01].index.tolist()
train_mask = E_ohe.season != 2023     # importance computed on training seasons only
rf_sel = RandomForestRegressor(300, min_samples_leaf=3, n_jobs=-1, random_state=0).fit(Xc.loc[train_mask, keep_mi], yv[train_mask])
pi = permutation_importance(rf_sel, Xc.loc[~train_mask, keep_mi], yv[~train_mask], n_repeats=8, random_state=0, n_jobs=-1)
pi_s = pd.Series(pi.importances_mean, index=keep_mi).sort_values(ascending=False)
selected = pi_s[pi_s > 0.002].index.tolist()
R["selection"] = {"after_collinearity": Xc.shape[1], "after_mutual_info": len(keep_mi), "after_permutation": len(selected),
                  "top10_permutation": pi_s.head(10).round(4).to_dict(), "top10_mutual_info": mi.head(10).round(3).to_dict(),
                  "selected": selected}

# ------------------------------------------------------------------ 7. MODEL COMPARISON - leave-one-season-out
def loso(train_df, feats, test_df, model_fn):
    res = []
    for s in [2022, 2023, 2024]:
        tr = train_df[train_df.season != s]; te = test_df[test_df.season == s]
        m = model_fn().fit(tr[feats], tr.yield_t_ha); p = m.predict(te[feats])
        res.append({"season": s, "rmse": float(np.sqrt(mean_squared_error(te.yield_t_ha, p))), "mae": float(mean_absolute_error(te.yield_t_ha, p)), "r2": float(r2_score(te.yield_t_ha, p))})
    r = pd.DataFrame(res); return {"per_season": r.round(3).to_dict("records"), "mean_rmse": round(r.rmse.mean(), 3), "mean_mae": round(r.mae.mean(), 3), "mean_r2": round(r.r2.mean(), 3)}
gb = lambda: HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0)
# honest test set: cleaned labels from ground-truth table for the same field-seasons (isolates effect of TRAINING-data quality)
tt_ = truth.set_index(KEY).yield_t_ha
test_df = E_ohe.copy(); test_df["yield_t_ha"] = [tt_.loc[(a, b)] for a, b in zip(test_df.field_id, test_df.season)]
# A: naive - raw messy numerics, sentinels left in, median fill, labels as-is
nv = raw.drop_duplicates(KEY).reset_index(drop=True); nv_feats = ["rain_mm", "tmax_mean_c", "heat_stress_days", "max_dry_spell_days", "ndvi_peak", "ndvi_mid",
          "n_applied_kg_ha", "irrigation_mm", "soil_ph", "soil_om_pct", "soil_n_kg_ha", "area_ha", "planting_doy"]
nv[nv_feats] = nv[nv_feats].fillna(nv[nv_feats].median())
nv_test = nv[KEY].merge(truth[KEY + ["yield_t_ha"]], on=KEY).join(nv[nv_feats]).rename(columns={})
nv_test["yield_t_ha"] = nv_test.yield_t_ha
R["models"] = {}
hist = lambda tr, te: float(np.sqrt(mean_squared_error(te.yield_t_ha, te.crop_rice.map(tr.groupby("crop_rice").yield_t_ha.mean()))))
r = [];
for s in [2022, 2023, 2024]:
    tr = test_df[test_df.season != s]; te = test_df[test_df.season == s]; r.append(hist(tr, te))
R["models"]["0_baseline_crop_mean"] = {"mean_rmse": round(float(np.mean(r)), 3)}
nv_train = nv.copy(); nv_train["season"] = nv.season
def naive_run():
    out = []
    for s in [2022, 2023, 2024]:
        tr = nv_train[nv_train.season != s]; te = nv_test[nv_test.season == s]
        m = gb().fit(tr[nv_feats], tr.yield_t_ha); p = m.predict(te[nv_feats])
        out.append(float(np.sqrt(mean_squared_error(te.yield_t_ha, p))))
    return round(float(np.mean(out)), 3)
nv_test["season"] = nv.season.values
R["models"]["A_naive_raw"] = {"mean_rmse": naive_run()}
R["models"]["B_cleaned_raw_features"] = loso(E_ohe, [c for c in base_num + [c for c in ohe_cols if c.startswith(("crop_", "region_", "soil_", "variety_"))]], test_df, gb)
R["models"]["C_cleaned_engineered"] = loso(E_ohe, all_feats, test_df, gb)
R["models"]["D_engineered_selected"] = loso(E_ohe, selected, test_df, gb)
R["models"]["D_selected_random_forest"] = loso(E_ohe, selected, test_df, lambda: RandomForestRegressor(300, min_samples_leaf=3, n_jobs=-1, random_state=0))
R["yield_stats"] = {"mean": round(float(truth.yield_t_ha.mean()), 2), "std": round(float(truth.yield_t_ha.std()), 2)}
R["scaling_demo"] = {c: {"min": round(float(d[c].min()), 2), "max": round(float(d[c].max()), 2)} for c in ["area_ha", "rain_mm", "ndvi_peak", "soil_ph"]}
R["discretization_counts"] = {"ph_band": E.ph_band.value_counts().to_dict(), "plant_window": E.plant_window.value_counts().to_dict()}
E_ohe.to_csv("agri_model_ready.csv", index=False)

# ------------------------------------------------------------------ FIGURES
fig, ax = plt.subplots(figsize=(6.4, 3.2)); s = pd.Series(R["audit"]["missing_pct"]).sort_values()
ax.barh(s.index, s.values, color=LIGHT); [ax.text(v + .2, i, f"{v}%", va="center", fontsize=8) for i, v in enumerate(s.values)]
ax.set_xlabel("% of rows missing"); ax.set_title("Missing values in the raw dataset", loc="left", fontsize=10, color=GREEN); plt.tight_layout(); plt.savefig("figures/missing.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(6.4, 3.4)); df_ = pd.DataFrame(R["imputation_rmse"]).T
df_n = df_.div(pd.Series(R["imputation_rmse_scale"]), axis=0)
df_n.plot.bar(ax=ax, color=[GREY, "#C7CFCB", "#7FB28F", LIGHT, GREEN], width=.8); ax.set_ylabel("RMSE / std of column (lower = better)"); ax.set_xticklabels(df_n.index, rotation=0, fontsize=8)
ax.axhline(1, ls="--", lw=.8, color="#999"); ax.set_ylim(0, 1.5); ax.legend(fontsize=7, ncol=5, frameon=False, loc="upper center"); ax.set_title("Imputation error by method (15% extra values masked)", loc="left", fontsize=10, color=GREEN)
plt.tight_layout(); plt.savefig("figures/imputation.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(6.4, 3.4)); p = pi_s.head(12)[::-1]
ax.barh(p.index, p.values.clip(max=0.22), color=[GREEN if i in eng_new else LIGHT for i in p.index]); ax.set_xlim(0, 0.26)
[ax.text(0.225, k, f"{v:.2f} (off scale)", va="center", fontsize=7) for k, (i, v) in enumerate(p.items()) if v > 0.22]
ax.set_xlabel("Permutation importance (drop in R², 2023 held out)"); ax.set_title("Top features - dark green = engineered", loc="left", fontsize=10, color=GREEN)
plt.tight_layout(); plt.savefig("figures/importance.png", dpi=200); plt.close()

fig, ax = plt.subplots(figsize=(6.4, 3.2)); M = R["models"]
labels = ["Crop-mean\nbaseline", "A. Naive\n(raw data)", "B. Cleaned", "C. + Engineered", "D. + Selected"]
vals = [M["0_baseline_crop_mean"]["mean_rmse"], M["A_naive_raw"]["mean_rmse"], M["B_cleaned_raw_features"]["mean_rmse"], M["C_cleaned_engineered"]["mean_rmse"], M["D_engineered_selected"]["mean_rmse"]]
b = ax.bar(labels, vals, color=[GREY, "#C7CFCB", "#7FB28F", LIGHT, GREEN]); [ax.text(x.get_x() + x.get_width() / 2, v + .01, f"{v:.3f}", ha="center", fontsize=8) for x, v in zip(b, vals)]
ax.set_ylabel("Mean RMSE, t/ha (leave-one-season-out)"); ax.set_title("Effect of each preprocessing stage on forecast error", loc="left", fontsize=10, color=GREEN)
plt.xticks(fontsize=8); plt.tight_layout(); plt.savefig("figures/model_progress.png", dpi=200); plt.close()

json.dump(R, open("results.json", "w"), indent=1, default=str)
print(json.dumps({k: R[k] for k in ["audit", "rule_violations", "rows_dropped_no_label", "rows_after_cleaning", "outliers_tmax", "outliers_yield", "imputation_rmse", "imputation_rmse_scale", "n_features", "dropped_collinear", "models", "yield_stats"]}, indent=1, default=str))
print(json.dumps(R["selection"], indent=1))

# ------------------------------------------------------------------ 8. EXTRA: does engineering help a LINEAR model? (Ridge, scaled)
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
ridge = lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20)))
base_feats = [c for c in base_num + [c for c in ohe_cols if c.startswith(("crop_", "region_", "soil_", "variety_"))]]
R["models_ridge"] = {"B_cleaned_raw_features": loso(E_ohe, base_feats, test_df, ridge),
                     "C_cleaned_engineered": loso(E_ohe, all_feats, test_df, ridge),
                     "D_engineered_selected": loso(E_ohe, selected, test_df, ridge)}
# scaling demo: effect on a distance-based method is shown via KNN regression RMSE with/without scaling
from sklearn.neighbors import KNeighborsRegressor
knn_raw = lambda: KNeighborsRegressor(10)
knn_sc = lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(10))
R["scaling_effect_knn"] = {"unscaled": loso(E_ohe, base_feats, test_df, knn_raw)["mean_rmse"], "standardised": loso(E_ohe, base_feats, test_df, knn_sc)["mean_rmse"]}
json.dump(R, open("results.json", "w"), indent=1, default=str)
print("RIDGE", json.dumps({k: v["mean_rmse"] for k, v in R["models_ridge"].items()}), "KNN scaling", R["scaling_effect_knn"])
