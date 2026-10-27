"""Week 3 - Part B2: metrics, classification, cost, spatial/temporal tests, improvement experiments."""
import json, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import HistGradientBoostingRegressor as HGB, HistGradientBoostingClassifier as HGC, RandomForestRegressor
from sklearn.linear_model import RidgeCV, QuantileRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, r2_score, roc_auc_score, average_precision_score, fbeta_score, confusion_matrix
from sklearn.inspection import permutation_importance, partial_dependence
R = {}
df = pd.read_csv("agri_model_ready.csv")
df["region"] = df[[c for c in df if c.startswith("region_")]].idxmax(axis=1).str[7:]
df["crop"] = np.where(df.crop_rice == 1, "rice", "maize")
Y = "yield_t_ha"; SEAS = [2022, 2023, 2024]
FULL = [c for c in df.columns if c not in ["field_id", "season", Y, "region", "crop"]]
ENG = ["ndvi_drop", "ndvi_peak_anom", "total_water_mm", "water_balance_mm", "heat_x_dry", "n_supply_kg_ha", "n_use_ratio", "ph_dist_opt", "plant_delay_d", "log_area"]
BASE = [c for c in FULL if c not in ENG and not c.startswith(("ph_band_", "plant_window_", "n_band_"))]
rmse = lambda a, b: float(np.sqrt(mse(a, b)))
mk = lambda **kw: HGB(**{**dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0), **kw})
LS = [(df.index[df.season != s], df.index[df.season == s]) for s in SEAS]
def oof(fn, feats, data=df, target=None, inv=None):
    p = pd.Series(np.nan, index=data.index)
    for tr, te in LS:
        yt = data.loc[tr, Y] if target is None else target(data.loc[tr, Y])
        pr = fn().fit(data.loc[tr, feats], yt).predict(data.loc[te, feats]); p[te] = pr if inv is None else inv(pr)
    return p
P = pd.read_pickle("preds.pkl")
B1 = json.load(open("results_B1.json"))
PRIMARY = "Elastic Net"
p = P[PRIMARY]; p0 = P["Crop x region mean (proxy for current method)"]; pc = P["Crop-mean baseline"]; y = df[Y]

# ------------------------------------------------ metrics suite
def suite(y, p, base=None, area=None):
    e = p - y
    out = {"n": int(len(y)), "rmse": rmse(y, p), "mae": float(mae(y, p)), "mape_pct": float((np.abs(e) / y).mean() * 100), "r2": float(r2_score(y, p)),
           "bias_t_ha": float(e.mean()), "rel_rmse_pct": rmse(y, p) / float(y.mean()) * 100,
           "within_0.5_pct": float((np.abs(e) <= 0.5).mean() * 100), "within_0.25_pct": float((np.abs(e) <= 0.25).mean() * 100),
           "p90_abs_err": float(np.quantile(np.abs(e), .9))}
    mu_y, mu_p = y.mean(), p.mean(); out["ccc"] = float(2 * np.cov(y, p, ddof=0)[0, 1] / (y.var(ddof=0) + p.var(ddof=0) + (mu_y - mu_p) ** 2))
    if base is not None: out["mse_skill_vs_base"] = float(1 - mse(y, p) / mse(y, base))
    return {k: round(v, 3) if isinstance(v, float) else v for k, v in out.items()}
R["metrics_overall"] = {"primary": suite(y, p, p0), "crop_region_mean": suite(y, p0), "crop_mean": suite(y, pc)}
R["improvement_vs_baselines_pct"] = {"vs_crop_region_mean": round((1 - rmse(y, p) / rmse(y, p0)) * 100, 1), "vs_crop_mean": round((1 - rmse(y, p) / rmse(y, pc)) * 100, 1),
                                     "mae_vs_crop_region_mean": round((1 - mae(y, p) / mae(y, p0)) * 100, 1)}
for key in ["crop", "region", "season"]:
    R[f"metrics_by_{key}"] = {str(k): suite(y[g.index], p[g.index], p0[g.index]) for k, g in df.groupby(key)}
# tail behaviour
q10 = y.quantile(.1); q90 = y.quantile(.9); lo = y <= q10; hi = y >= q90
R["tail"] = {"q10": round(float(q10), 2), "bias_bottom10": round(float((p[lo] - y[lo]).mean()), 3), "rmse_bottom10": round(rmse(y[lo], p[lo]), 3),
             "bias_top10": round(float((p[hi] - y[hi]).mean()), 3), "rmse_top10": round(rmse(y[hi], p[hi]), 3),
             "calibration_slope_obs_on_pred": round(float(np.polyfit(p, y, 1)[0]), 3), "calibration_intercept": round(float(np.polyfit(p, y, 1)[1]), 3)}
# aggregation: errors shrink when predictions are aggregated (what a procurement manager really needs)
agg = df.assign(p=p, pa=p * df.area_ha, ya=y * df.area_ha).groupby(["region", "season"]).agg(ya=("ya", "sum"), pa=("pa", "sum"), area=("area_ha", "sum"), y=(Y, "mean"), p=("p", "mean"), n=(Y, "size"))
agg["prod_err_pct"] = (agg.pa / agg.ya - 1) * 100; agg["yield_err"] = agg.p - agg.y
p0a = df.assign(p=p0 * df.area_ha, ya=y * df.area_ha).groupby(["region", "season"]).agg(pa=("p", "sum"), ya=("ya", "sum"))
agg["base_prod_err_pct"] = (p0a.pa / p0a.ya - 1) * 100
R["aggregate"] = {"field_level_mape_pct": round(float((np.abs(p - y) / y).mean() * 100), 2), "region_season_mean_abs_prod_err_pct": round(float(agg.prod_err_pct.abs().mean()), 2),
                  "region_season_max_abs_prod_err_pct": round(float(agg.prod_err_pct.abs().max()), 2), "baseline_mean_abs_prod_err_pct": round(float(agg.base_prod_err_pct.abs().mean()), 2),
                  "baseline_max_abs_prod_err_pct": round(float(agg.base_prod_err_pct.abs().max()), 2),
                  "table": agg.reset_index().round(2)[["region", "season", "n", "prod_err_pct", "base_prod_err_pct"]].to_dict("records")}
# cluster bootstrap by field
rs = np.random.default_rng(0); fields = df.field_id.unique(); idx_by_f = df.groupby("field_id").indices
diffs = []; ratios = []; maes = []
for _ in range(1000):
    ff = rs.choice(fields, len(fields)); ii = np.concatenate([idx_by_f[f] for f in ff])
    a = rmse(y.values[ii], p.values[ii]); b = rmse(y.values[ii], p0.values[ii]); diffs.append(b - a); ratios.append(1 - a / b)
    maes.append(rmse(y.values[ii], p.values[ii]))
R["bootstrap"] = {"rmse_ci95": [round(float(np.quantile(maes, .025)), 3), round(float(np.quantile(maes, .975)), 3)],
                  "reduction_vs_baseline_ci95_pct": [round(float(np.quantile(ratios, .025)) * 100, 1), round(float(np.quantile(ratios, .975)) * 100, 1)]}

# ------------------------------------------------ classification: low-yield alert
def f2(yt, yp): return fbeta_score(yt, yp, beta=2, zero_division=0)
rows = []; cm_tot = np.zeros((2, 2), int); allp = []; ally = []; allpred_reg = []; prev = {}
for s in SEAS:
    tr = df[df.season != s]; te = df[df.season == s]
    thr = tr.groupby("crop")[Y].quantile(.2); ltr = (tr[Y] < tr.crop.map(thr)).astype(int); lte = (te[Y] < te.crop.map(thr)).astype(int)
    prev[s] = round(float(lte.mean() * 100), 1)
    # inner OOF probs -> pick threshold maximising F2 (no peeking at the test season)
    inner = np.zeros(len(tr))
    for a, b in GroupKFold(4).split(tr, groups=tr.field_id):
        inner[b] = HGC(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0).fit(tr[FULL].iloc[a], ltr.iloc[a]).predict_proba(tr[FULL].iloc[b])[:, 1]
    ts = np.linspace(0.05, 0.8, 31); bt = ts[int(np.argmax([f2(ltr.values, (inner >= t).astype(int)) for t in ts]))]
    clf = HGC(max_iter=200, learning_rate=0.05, max_leaf_nodes=15, random_state=0).fit(tr[FULL], ltr); pr = clf.predict_proba(te[FULL])[:, 1]
    # regression-derived flag: predicted yield below the crop's alert threshold
    reg_flag = (P[PRIMARY][te.index] < te.crop.map(thr)).astype(int)
    def m(flag): 
        tn, fp, fn, tp = confusion_matrix(lte, flag, labels=[0, 1]).ravel()
        return dict(tp=int(tp), fp=int(fp), fn=int(fn), tn=int(tn), accuracy=round((tp + tn) / len(lte), 3), precision=round(tp / max(tp + fp, 1), 3), recall=round(tp / max(tp + fn, 1), 3),
                    f1=round(float(fbeta_score(lte, flag, beta=1, zero_division=0)), 3), f2=round(float(f2(lte, flag)), 3))
    rows.append({"season": s, "prevalence_pct": prev[s], "threshold_t_ha": thr.round(2).to_dict(), "roc_auc": round(float(roc_auc_score(lte, pr)), 3), "pr_auc": round(float(average_precision_score(lte, pr)), 3),
                 "chosen_prob_threshold": round(float(bt), 2), "classifier_at_0.5": m((pr >= .5).astype(int)), "classifier_tuned_thr": m((pr >= bt).astype(int)), "regression_flag": m(reg_flag.values),
                 "always_no_alert_accuracy": round(float(1 - lte.mean()), 3)})
    allp.append(pr); ally.append(lte.values)
R["classification_by_season"] = rows
def pooled(key):
    tp = sum(r[key]["tp"] for r in rows); fp = sum(r[key]["fp"] for r in rows); fn = sum(r[key]["fn"] for r in rows); tn = sum(r[key]["tn"] for r in rows); n = tp + fp + fn + tn
    pr_ = tp / max(tp + fp, 1); rc = tp / max(tp + fn, 1)
    return dict(tp=tp, fp=fp, fn=fn, tn=tn, accuracy=round((tp + tn) / n, 3), precision=round(pr_, 3), recall=round(rc, 3), f1=round(2 * pr_ * rc / max(pr_ + rc, 1e-9), 3),
                f2=round(5 * pr_ * rc / max(4 * pr_ + rc, 1e-9), 3), prevalence_pct=round((tp + fn) / n * 100, 1), always_no_alert_accuracy=round((tn + fp) / n, 3))
R["classification_pooled"] = {k: pooled(k) for k in ["classifier_at_0.5", "classifier_tuned_thr", "regression_flag"]}
pa = np.concatenate(allp); ya_ = np.concatenate(ally)
R["classification_pooled"]["roc_auc"] = round(float(roc_auc_score(ya_, pa)), 3); R["classification_pooled"]["pr_auc"] = round(float(average_precision_score(ya_, pa)), 3)
np.save("clf_pa.npy", pa); np.save("clf_ya.npy", ya_)

# ------------------------------------------------ asymmetric cost + quantile models + intervals
qs = [0.1, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 0.9]
lq = lambda q: (lambda: make_pipeline(StandardScaler(), QuantileRegressor(quantile=q, alpha=0.0005, solver="highs")))
qp = {q: oof(lq(q), FULL) for q in qs}
qh = {q: oof(lambda q=q: HGB(loss="quantile", quantile=q, max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=0), FULL) for q in [0.1, 0.25, 0.5, 0.9]}
CO, CU = 3.0, 1.0     # assumed: over-forecast (promised tonnes not delivered) costs 3x an under-forecast (unused capacity)
def cost(pred): 
    e = (pred - y) * df.area_ha; return float((CO * e.clip(lower=0) + CU * (-e).clip(lower=0)).sum())
costs = {"mean_model": cost(p), "baseline": cost(p0)}
for q in sorted(qp): costs[f"q{q}"] = cost(qp[q])
costs["hgb_q0.25"] = cost(qh[0.25]); costs["hgb_q0.5"] = cost(qh[0.5])
R["cost"] = {"assumed_ratio_over_to_under": CO / CU, "optimal_quantile_theory": CU / (CU + CO), "cost_tonne_units": {k: round(v, 0) for k, v in costs.items()},
             "saving_vs_mean_model_pct_at_q0.25": round((1 - costs["q0.25"] / costs["mean_model"]) * 100, 1), "saving_hgb_q0.25_vs_hgb_q0.5_pct": round((1 - costs["hgb_q0.25"] / costs["hgb_q0.5"]) * 100, 1),
             "bias_t_ha_by_quantile": {str(q): round(float((qp[q] - y).mean()), 3) for q in sorted(qp)}}
cov = ((y >= qp[0.1]) & (y <= qp[0.9])).mean()
covh = ((y >= qh[0.1]) & (y <= qh[0.9])).mean(); R["interval_80_hgb"] = {"coverage_pct": round(float(covh * 100), 1), "mean_width_t_ha": round(float((qh[0.9] - qh[0.1]).mean()), 3)}
R["interval_80"] = {"coverage_pct": round(float(cov * 100), 1), "mean_width_t_ha": round(float((qp[0.9] - qp[0.1]).mean()), 3),
                    "coverage_by_season": {str(s): round(float(((y >= qp[0.1]) & (y <= qp[0.9]))[df.season == s].mean() * 100), 1) for s in SEAS}}
pd.DataFrame({str(q): v for q, v in qp.items()}).to_pickle("qp.pkl"); pd.DataFrame({str(q): v for q, v in qh.items()}).to_pickle("qh.pkl")

# ------------------------------------------------ SPATIAL mismatch simulation (coarse weather, mixed-pixel NDVI)
WD = {"rice": 900.0, "maize": 600.0}
def rebuild(d):
    d = d.copy(); d["total_water_mm"] = d.rain_mm + d.irrigation_mm; d["water_balance_mm"] = d.total_water_mm - d.crop.map(WD)
    d["heat_x_dry"] = d.heat_stress_days * d.max_dry_spell_days / 10; d["ndvi_drop"] = d.ndvi_peak - d.ndvi_mid
    d["ndvi_peak_anom"] = d.ndvi_peak - d.groupby(["crop", "region", "season"]).ndvi_peak.transform("mean"); return d
wx = ["rain_mm", "tmax_mean_c", "heat_stress_days", "max_dry_spell_days"]
cell_mean = df.groupby(["region", "season"])[wx].transform("mean")
sp = {}
for a in [1.0, 0.75, 0.5, 0.25, 0.0]:
    d = df.copy(); d[wx] = a * df[wx] + (1 - a) * cell_mean; d = rebuild(d); sp[str(a)] = round(rmse(y, oof(mk, FULL, d)), 3)
R["spatial_weather_blend_rmse"] = sp
nd = ["ndvi_peak", "ndvi_mid"]; surround = df.groupby(["region", "season"])[nd].transform("mean")
sens = {"Sentinel-2 10 m": 0.01, "Landsat 30 m": 0.09, "MODIS 250 m": 6.25}; sp2 = {}; purity = {}
for k, pix in sens.items():
    f = (df.area_ha / pix).clip(upper=1.0); d = df.copy()
    for c in nd: d[c] = f * df[c] + (1 - f) * surround[c]
    d = rebuild(d); sp2[k] = round(rmse(y, oof(mk, FULL, d)), 3); purity[k] = round(float(f.mean() * 100), 1)
R["spatial_pixel_mixing_rmse"] = sp2; R["spatial_pixel_purity_pct"] = purity
R["spatial_reference_rmse_clean"] = round(rmse(y, p), 3)

# ------------------------------------------------ TEMPORAL misalignment simulation (stage-aligned vs calendar windows)
def sim(sd, seed=0, n_ry=60, nf=30):
    g = np.random.default_rng(seed); Xc, Xs, Yv, G = [], [], [], []
    d0 = 150; days = np.arange(d0, 331)
    for ry in range(n_ry):
        pw = np.clip(0.28 + 0.30 * np.sin(np.pi * (days - d0) / 180) + g.normal(0, 0.06), 0.03, 0.9)
        rain = np.where(g.random(len(days)) < pw, g.exponential(11 * np.exp(g.normal(0, .25)), len(days)), 0)
        cs = np.cumsum(np.r_[0, rain])
        win = lambda a, b: cs[b - d0 + 1] - cs[a - d0]
        for _ in range(nf):
            pl = int(np.clip(g.normal(170, sd), 155, 190)) if sd > 0 else 170
            fl = win(pl + 60, pl + 80); cal = win(230, 250)
            yv = 3.0 * (1 - np.exp(-fl / 120)) + g.normal(0, 0.2)
            Xc.append(cal); Xs.append(fl); Yv.append(yv); G.append(ry)
    return np.array(Xc)[:, None], np.array(Xs)[:, None], np.array(Yv), np.array(G)
tsim = {}
for sd in [0, 4, 8, 12, 16]:
    Xc, Xs, Yv, G = sim(sd); r = {}
    for nm, X in [("calendar_window", Xc), ("stage_aligned_window", Xs)]:
        pr = np.zeros(len(Yv))
        for a, b in GroupKFold(5).split(X, Yv, G): pr[b] = HGB(max_iter=100, learning_rate=0.1, max_leaf_nodes=6, random_state=0).fit(X[a], Yv[a]).predict(X[b])
        r[nm] = round(float(r2_score(Yv, pr)), 3)
    r["corr_calendar"] = round(float(np.corrcoef(Xc[:, 0], Xs[:, 0])[0, 1]), 3); tsim[str(sd)] = r
R["temporal_sim_r2"] = tsim
# boundary double count with inclusive slicing
idx = pd.date_range("2023-06-01", periods=130); s_ = pd.Series(np.random.default_rng(1).exponential(4, 130), index=idx)
pdate = pd.Timestamp("2023-06-10"); edges = [0, 40, 80, 120]
incl = [s_.loc[pdate + pd.Timedelta(days=edges[i]): pdate + pd.Timedelta(days=edges[i + 1])].sum() for i in range(3)]
excl = [s_.loc[pdate + pd.Timedelta(days=edges[i]): pdate + pd.Timedelta(days=edges[i + 1] - 1)].sum() for i in range(3)]
R["boundary_double_count"] = {"days_in_stage_inclusive": [len(s_.loc[pdate + pd.Timedelta(days=edges[i]): pdate + pd.Timedelta(days=edges[i + 1])]) for i in range(3)], "sum_inclusive_mm": round(float(sum(incl)), 1), "sum_exclusive_mm": round(float(sum(excl)), 1), "double_counted_mm": round(float(sum(incl) - sum(excl)), 1)}

# ------------------------------------------------ INFORMATION HORIZON (forecast origin)
STATIC = [c for c in FULL if c.startswith(("crop_", "region_", "soil_type_", "variety_", "ph_band_", "plant_window_")) or c in ["soil_ph", "soil_om_pct", "soil_n_kg_ha", "area_ha", "log_area", "planting_doy", "plant_delay_d", "ph_dist_opt", "soil_ph_missing"]]
MID = STATIC + [c for c in FULL if c in ["ndvi_mid", "ndvi_mid_missing", "n_applied_kg_ha", "n_applied_kg_ha_missing", "n_supply_kg_ha", "n_use_ratio"] or c.startswith("n_band_")]
PEAK = MID + ["ndvi_peak", "ndvi_peak_missing", "ndvi_drop", "ndvi_peak_anom"]
hz = {"0 Pre-season (soil, variety, planting date)": STATIC, "1 Mid-season (+ mid NDVI, N applied)": MID, "2 Peak canopy (+ peak NDVI)": PEAK, "3 Season end (+ weather totals, irrigation)": FULL}
R["horizon_rmse"] = {k: {"n_features": len(v), "rmse": round(rmse(y, oof(mk, v)), 3)} for k, v in hz.items()}
R["horizon_rmse"]["baseline_crop_region_mean"] = round(rmse(y, p0), 3)

# ------------------------------------------------ IMPROVEMENT experiments
imp = {}
mono = {c: 0 for c in FULL}; mono.update({"rain_mm": 1, "heat_stress_days": -1, "max_dry_spell_days": -1, "ndvi_peak": 1, "ndvi_mid": 1, "soil_om_pct": 1, "ph_dist_opt": -1, "n_applied_kg_ha": 1, "heat_x_dry": -1})
imp["monotonic_constraints"] = {"unconstrained": round(rmse(y, p), 3), "constrained": round(rmse(y, oof(lambda: mk(monotonic_cst=mono), FULL)), 3)}
imp["log_target"] = round(rmse(y, oof(mk, FULL, target=np.log, inv=np.exp)), 3)
# native missing handling vs imputed (BASE features, flags used to restore NaN)
dn = df.copy()
for c in ["n_applied_kg_ha", "ndvi_peak", "ndvi_mid", "irrigation_mm", "soil_ph"]: dn.loc[dn[f"{c}_missing"] == 1, c] = np.nan
imp["native_nan_vs_imputed_base_features"] = {"imputed_in_file": round(rmse(y, oof(mk, BASE)), 3), "native_NaN_handling": round(rmse(y, oof(mk, BASE, dn)), 3), "n_restored_nan": int(dn[["n_applied_kg_ha", "ndvi_peak", "ndvi_mid", "irrigation_mm", "soil_ph"]].isna().sum().sum())}
# lag feature (forward chaining only: train 2022-23, test 2024)
dl = df.merge(df[["field_id", "season", Y]].assign(season=lambda x: x.season + 1).rename(columns={Y: "yield_prev"}), on=["field_id", "season"], how="left")
tr = dl[dl.season <= 2023]; te = dl[dl.season == 2024]
r_nolag = rmse(te[Y], mk().fit(tr[FULL], tr[Y]).predict(te[FULL])); r_lag = rmse(te[Y], mk().fit(tr[FULL + ["yield_prev"]], tr[Y]).predict(te[FULL + ["yield_prev"]]))
tm = te.dropna(subset=["yield_prev"]); 
imp["lag_yield_forward_chain_2024"] = {"without": round(r_nolag, 3), "with_prev_season_yield": round(r_lag, 3), "test_rows_with_lag": int(te.yield_prev.notna().sum()), "test_rows": int(len(te)),
                                         "corr_prev_vs_current_yield_in_residuals": None}
# is there any field-level persistence left in the residuals?
res = (y - p).groupby(df.field_id).mean(); rr = pd.DataFrame({"r": y - p, "f": df.field_id, "s": df.season}).pivot_table(index="f", columns="s", values="r")
imp["field_residual_persistence_corr_2023_2024"] = round(float(rr[2023].corr(rr[2024])), 3)
# spelling / unit errors reaching the model
raw = pd.read_csv("agri_raw_messy.csv").drop_duplicates(["field_id", "season"])
dm = df.merge(raw[["field_id", "season", "crop", "region", "soil_type", "area_ha", "area_unit"]].rename(columns={"crop": "crop_raw", "region": "region_raw", "soil_type": "soil_raw", "area_ha": "area_raw"}), on=["field_id", "season"], how="left")
keep = [c for c in FULL if not c.startswith(("crop_", "region_", "soil_type_"))]
dums = pd.get_dummies(dm[["crop_raw", "region_raw", "soil_raw"]].astype(str), dtype=int); dm = pd.concat([dm, dums], axis=1)
dirty_cat = keep + list(dums.columns)
imp["dirty_spellings_onehot"] = {"clean": round(rmse(y, oof(mk, FULL)), 3), "raw_spellings_onehot": round(rmse(y, oof(mk, dirty_cat, dm)), 3), "n_dummy_cols_clean": 10, "n_dummy_cols_raw": int(dums.shape[1])}
dm["ndvi_peak_anom"] = dm.ndvi_peak - dm.groupby(["crop_raw", "region_raw", "season"]).ndvi_peak.transform("mean")
imp["dirty_spellings_plus_fragmented_peer_feature"] = round(rmse(y, oof(mk, dirty_cat, dm)), 3)
# group sizes
imp["fragmented_group_sizes"] = {"clean_min": int(df.groupby(["crop", "region", "season"]).size().min()), "raw_min": int(dm.groupby(["crop_raw", "region_raw", "season"]).size().min()), "raw_n_groups": int(dm.groupby(["crop_raw", "region_raw", "season"]).ngroups), "clean_n_groups": int(df.groupby(["crop", "region", "season"]).ngroups)}
dm["area_ha"] = dm.area_raw; dm["log_area"] = np.log1p(dm.area_raw)
imp["unit_error_area_in_features"] = round(rmse(y, oof(mk, FULL, dm)), 3)
# learning curve on number of training fields
lc = {}; fl = df.field_id.unique(); g = np.random.default_rng(0)
for frac in [0.1, 0.25, 0.5, 0.75, 1.0]:
    rr_ = []
    for rep in range(3):
        sel = set(g.choice(fl, int(frac * len(fl)), replace=False)); pr = pd.Series(np.nan, index=df.index)
        for tr_, te_ in LS:
            tr2 = tr_[df.loc[tr_, "field_id"].isin(sel).values]; pr[te_] = mk().fit(df.loc[tr2, FULL], df.loc[tr2, Y]).predict(df.loc[te_, FULL])
        rr_.append(rmse(y, pr))
    lc[str(frac)] = {"fields": int(frac * len(fl)), "rmse": round(float(np.mean(rr_)), 3), "sd": round(float(np.std(rr_)), 3)}
R["learning_curve"] = lc
R["improvements"] = imp
# ------------------------------------------------ interpretability
tr = df[df.season != 2023]; te = df[df.season == 2023]
m = mk().fit(tr[FULL], tr[Y]); pi = permutation_importance(m, te[FULL], te[Y], n_repeats=8, random_state=0, scoring="neg_root_mean_squared_error")
R["perm_importance_top12"] = pd.Series(pi.importances_mean, index=FULL).sort_values(ascending=False).head(12).round(3).to_dict()
mfull = mk().fit(df[FULL], df[Y]); pdp = {}
for c in ["rain_mm", "heat_stress_days", "ndvi_peak", "n_applied_kg_ha", "max_dry_spell_days", "soil_ph"]:
    r = partial_dependence(mfull, df[FULL], [c], grid_resolution=20); pdp[c] = {"x": [round(float(v), 3) for v in r["grid_values"][0]], "y": [round(float(v), 3) for v in r["average"][0]]}
R["pdp"] = pdp
json.dump(R, open("results_B2.json", "w"), indent=1, default=str)
print(json.dumps({k: v for k, v in R.items() if k not in ["pdp", "aggregate"]}, indent=1, default=str)[:12000])
