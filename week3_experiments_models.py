"""Week 3 - Part B: model-planning experiments on agri_model_ready.csv (hypothetical data)."""
import json, time, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor, HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import RidgeCV, ElasticNetCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GroupKFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error as mse, mean_absolute_error as mae, r2_score, roc_auc_score, average_precision_score, precision_recall_curve
import lightgbm as lgb, xgboost as xgb
R = {}; SEED = 0
df = pd.read_csv("agri_model_ready.csv")
df["region"] = df[[c for c in df if c.startswith("region_")]].idxmax(axis=1).str[7:]
df["crop"] = np.where(df.crop_rice == 1, "rice", "maize")
Y = "yield_t_ha"; SEAS = [2022, 2023, 2024]
FULL = [c for c in df.columns if c not in ["field_id", "season", Y, "region", "crop"]]
ohe = [c for c in FULL if c.startswith(("crop_", "region_", "soil_type_", "variety_", "ph_band_", "plant_window_", "n_band_"))]
ENG = ["ndvi_drop", "ndvi_peak_anom", "total_water_mm", "water_balance_mm", "heat_x_dry", "n_supply_kg_ha", "n_use_ratio", "ph_dist_opt", "plant_delay_d", "log_area"]
BASE = [c for c in FULL if c not in ENG and not c.startswith(("ph_band_", "plant_window_", "n_band_"))]
R["data"] = {"rows": len(df), "fields": df.field_id.nunique(), "n_full": len(FULL), "n_base": len(BASE),
             "yield_mean": round(df[Y].mean(), 3), "yield_sd": round(df[Y].std(), 3),
             "by_season_mean": df.groupby("season")[Y].mean().round(3).to_dict(), "by_crop_mean": df.groupby("crop")[Y].mean().round(3).to_dict(),
             "rows_per_season": df.season.value_counts().sort_index().to_dict()}
def rmse(a, b): return float(np.sqrt(mse(a, b)))
def mk_hgb(**kw): 
    p = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=SEED); p.update(kw); return HistGradientBoostingRegressor(**p)
def mk_rf(**kw):
    p = dict(n_estimators=200, min_samples_leaf=3, n_jobs=1, random_state=SEED); p.update(kw); return RandomForestRegressor(**p)

def cv_oof(model_fn, feats, splits, data=df):
    pred = pd.Series(np.nan, index=data.index)
    for tr, te in splits:
        m = model_fn().fit(data.loc[tr, feats], data.loc[tr, Y]); pred[te] = m.predict(data.loc[te, feats])
    return pred
def loso_splits(data=df): return [(data.index[data.season != s], data.index[data.season == s]) for s in SEAS]
def summarise(pred, data=df):
    ok = pred.notna(); y = data.loc[ok, Y]; p = pred[ok]
    return {"rmse": round(rmse(y, p), 3), "mae": round(float(mae(y, p)), 3), "r2": round(float(r2_score(y, p)), 3)}

# ---------------------------------------------------------------- E1 split strategies (same model: default HistGB, FULL features)
rs = np.random.default_rng(1)
sp = {}
rk = [rmse(df[Y], cv_oof(mk_hgb, FULL, [(df.index[tr], df.index[te]) for tr, te in KFold(5, shuffle=True, random_state=s).split(df)])) for s in range(3)]
sp["random_row_5fold"] = round(float(np.mean(rk)), 3)
sp["group_by_field_5fold"] = round(rmse(df[Y], cv_oof(mk_hgb, FULL, [(df.index[tr], df.index[te]) for tr, te in GroupKFold(5).split(df, groups=df.field_id)])), 3)
sp["leave_one_season_out"] = round(rmse(df[Y], cv_oof(mk_hgb, FULL, loso_splits())), 3)
rsplit = [(df.index[df.region != r], df.index[df.region == r]) for r in ["North", "South", "East", "West"]]
sp["leave_one_region_out"] = round(rmse(df[Y], cv_oof(mk_hgb, FULL, rsplit)), 3)
# hardest: unseen season AND unseen region
hard = []
for s in SEAS:
    for r in ["North", "South", "East", "West"]:
        tr = df.index[(df.season != s) & (df.region != r)]; te = df.index[(df.season == s) & (df.region == r)]
        p = mk_hgb().fit(df.loc[tr, FULL], df.loc[tr, Y]).predict(df.loc[te, FULL]); hard.append((df.loc[te, Y].values, p))
sp["leave_season_and_region_out"] = round(rmse(np.concatenate([a for a, b in hard]), np.concatenate([b for a, b in hard])), 3)
R["split_strategies_rmse"] = sp

# ---------------------------------------------------------------- E2 algorithm comparison (LOSO)
LS = loso_splits()
class GroupMean:
    def __init__(self, cols): self.cols = cols
    def fit(self, X, y): 
        t = X[self.cols].astype(str).agg("|".join, axis=1); self.m = y.groupby(t).mean(); self.g = y.mean(); return self
    def predict(self, X): 
        t = X[self.cols].astype(str).agg("|".join, axis=1); return t.map(self.m).fillna(self.g).values
df["_crop_region"] = df.crop + "|" + df.region
cands = {}
def run_group(name, cols):
    pred = pd.Series(np.nan, index=df.index)
    for tr, te in LS:
        pred[te] = GroupMean(cols).fit(df.loc[tr], df.loc[tr, Y]).predict(df.loc[te])
    return pred
algos = {}
preds = {}
t0 = time.time(); preds["Crop-mean baseline"] = run_group("crop", ["crop"]); algos["Crop-mean baseline"] = {"fit_s": 0.0}
preds["Crop x region mean (proxy for current method)"] = run_group("cr", ["crop", "region"]); algos["Crop x region mean (proxy for current method)"] = {"fit_s": 0.0}
models = {
 "Ridge": (lambda: make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(-2, 3, 20))), FULL),
 "Elastic Net": (lambda: make_pipeline(StandardScaler(), ElasticNetCV(l1_ratio=[.2, .5, .8, .95], cv=5, random_state=0, max_iter=5000)), FULL),
 "k-NN (k=10, scaled)": (lambda: make_pipeline(StandardScaler(), KNeighborsRegressor(10)), BASE),
 "SVR (RBF, scaled)": (lambda: make_pipeline(StandardScaler(), SVR(C=3, epsilon=0.1)), FULL),
 "MLP (2x32, scaled)": (lambda: make_pipeline(StandardScaler(), MLPRegressor(hidden_layer_sizes=(32, 32), alpha=1e-2, max_iter=800, early_stopping=True, random_state=0)), FULL),
 "Random Forest": (mk_rf, FULL),
 "Extra Trees": (lambda: ExtraTreesRegressor(200, min_samples_leaf=3, n_jobs=1, random_state=0), FULL),
 "HistGradientBoosting": (mk_hgb, FULL),
 "LightGBM": (lambda: lgb.LGBMRegressor(n_estimators=300, learning_rate=0.05, num_leaves=15, min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=0, verbose=-1, n_jobs=1), FULL),
 "XGBoost": (lambda: xgb.XGBRegressor(n_estimators=300, learning_rate=0.05, max_depth=4, subsample=0.8, colsample_bytree=0.8, random_state=0, n_jobs=1), FULL),
}
for k, (fn, ft) in models.items():
    t = time.time(); preds[k] = cv_oof(fn, ft, LS); algos[k] = {"fit_s": round((time.time() - t) / 3, 2)}
tab = {}
for k, p in preds.items():
    per = {s: rmse(df.loc[df.season == s, Y], p[df.season == s]) for s in SEAS}
    ok = p.notna(); y = df.loc[ok, Y]; pp = p[ok]
    tab[k] = dict(rmse=round(rmse(y, pp), 3), mae=round(float(mae(y, pp)), 3), r2=round(float(r2_score(y, pp)), 3),
                  mape=round(float((np.abs(y - pp) / y).mean() * 100), 1), **{f"rmse_{s}": round(v, 3) for s, v in per.items()}, **algos[k])
R["algorithms_loso"] = tab
oof_hgb = preds["HistGradientBoosting"]; oof_base = preds["Crop x region mean (proxy for current method)"]; oof_cm = preds["Crop-mean baseline"]
# simple ensemble
preds["Ensemble: HGB + RF + Ridge"] = (preds["HistGradientBoosting"] + preds["Random Forest"] + preds["Ridge"]) / 3
p = preds["Ensemble: HGB + RF + Ridge"]; R["ensemble_hgb_rf_ridge"] = summarise(p)
R["ensemble_hgb_rf_ridge"]["rmse_by_season"] = {s: round(rmse(df.loc[df.season == s, Y], p[df.season == s]), 3) for s in SEAS}
# feature-set effect for 3 model families
fs = {}
for nm, fn in [("Ridge", models["Ridge"][0]), ("HistGB", mk_hgb), ("Random Forest", mk_rf)]:
    fs[nm] = {"base": summarise(cv_oof(fn, BASE, LS))["rmse"], "full": summarise(cv_oof(fn, FULL, LS))["rmse"]}
R["feature_sets_rmse"] = fs

# ---------------------------------------------------------------- E3 hyper-parameter tuning (nested: outer LOSO, inner GroupKFold by field)
from scipy.stats import loguniform, randint, uniform
spaces = {
 "hgb": (mk_hgb, dict(learning_rate=loguniform(0.01, 0.3), max_leaf_nodes=randint(4, 40), min_samples_leaf=randint(5, 60), l2_regularization=loguniform(1e-3, 10), max_iter=randint(100, 600), max_features=uniform(0.4, 0.6))),
 "rf": (mk_rf, dict(n_estimators=[200, 400], max_depth=[None, 6, 10, 16], min_samples_leaf=randint(1, 15), max_features=uniform(0.2, 0.8))),
}
tune = {}
for nm, (fn, space) in spaces.items():
    out = []; chosen = []; t = time.time()
    for s in SEAS:
        tr = df[df.season != s]; te = df[df.season == s]
        inner = list(GroupKFold(4).split(tr, groups=tr.field_id))
        rs_ = RandomizedSearchCV(fn(), space, n_iter=15, cv=inner, scoring="neg_root_mean_squared_error", random_state=SEED, n_jobs=1, refit=True)
        rs_.fit(tr[FULL], tr[Y]); p = rs_.predict(te[FULL])
        out.append({"season": s, "rmse": round(rmse(te[Y], p), 3), "inner_cv_rmse": round(-rs_.best_score_, 3)}); chosen.append({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rs_.best_params_.items()})
    default = summarise(preds["HistGradientBoosting" if nm == "hgb" else "Random Forest"])["rmse"]
    tuned = round(float(np.sqrt(np.mean([o["rmse"] ** 2 * (df.season == o["season"]).sum() for o in out]) / len(df) * 3 / 3)), 3)
    allt = np.concatenate([[1] * 0]); 
    n_s = {s: int((df.season == s).sum()) for s in SEAS}
    tuned = round(float(np.sqrt(sum(o["rmse"] ** 2 * n_s[o["season"]] for o in out) / len(df))), 3)
    tune[nm] = {"default_rmse": default, "tuned_rmse": tuned, "per_season": out, "chosen": chosen, "seconds": round(time.time() - t, 1)}
R["tuning_nested"] = tune
# sensitivity grid (learning rate x leaves) with inner GroupKFold on all data, 300 iters
grid = {}
lrs = [0.01, 0.03, 0.05, 0.1, 0.2]; leaves = [4, 8, 15, 31]
for lr in lrs:
    for lv in leaves:
        sc = []
        for tr, te in GroupKFold(4).split(df, groups=df.field_id):
            sc.append(rmse(df.loc[df.index[te], Y], mk_hgb(learning_rate=lr, max_leaf_nodes=lv, max_iter=300).fit(df.loc[df.index[tr], FULL], df.loc[df.index[tr], Y]).predict(df.loc[df.index[te], FULL])))
        grid[f"{lr}|{lv}"] = round(float(np.mean(sc)), 3)
R["sensitivity_grid"] = {"lrs": lrs, "leaves": leaves, "rmse": grid}
# iteration curve (early-stopping view): train on 2022+2024, validate 2023 and on random field holdout
curves = {}
for lr in [0.02, 0.05, 0.15]:
    tr = df[df.season != 2023]; te = df[df.season == 2023]
    m = HistGradientBoostingRegressor(max_iter=600, learning_rate=lr, max_leaf_nodes=15, random_state=0).fit(tr[FULL], tr[Y])
    curves[str(lr)] = [round(rmse(te[Y], p), 4) for p in m.staged_predict(te[FULL])][::10]
R["iteration_curves_2023"] = curves
R["iteration_best"] = {k: (int(np.argmin(v)) * 10 + 10, round(min(v), 3)) for k, v in curves.items()}
json.dump(R, open("results_B1.json", "w"), indent=1, default=str)
preds_df = pd.DataFrame(preds); preds_df.to_pickle("preds.pkl")
print(json.dumps({k: R[k] for k in ["data", "split_strategies_rmse", "algorithms_loso", "ensemble_hgb_rf_ridge", "feature_sets_rmse", "tuning_nested", "iteration_best"]}, indent=1, default=str))
