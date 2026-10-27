# ML for Crop Yield Forecasting in Agribusiness

**Data preprocessing (Week 2) and model planning and evaluation (Week 3) for yield forecasting**

This repository holds the Week 2 and Week 3 deliverables of an internship project on *Machine Learning for Crop Yield Forecasting and Input Optimization in Agribusiness*. Week 2 turns messy farm data into a clean, model-ready table. Week 3 plans, compares, tunes and evaluates models on that table, with metrics designed for agribusiness decisions.

> **Data note.** All data are **synthetic** (600 fields, 3 seasons, rice and maize). Results show how well each *method* works, not the accuracy to expect on a real farm.

## The problem

An agribusiness such as a farmer cooperative has to decide how much to buy, store, transport and fertilise long before harvest. A good yield forecast makes those decisions cheaper and less risky. But the data behind a forecast is rarely clean:

- Harvest weights are typed by hand, so a decimal slip turns 3.4 t/ha into 34 t/ha.
- Some farms report area in acres and others in hectares.
- Satellite images are hidden by monsoon cloud, leaving gaps in the vegetation data.
- Sensors fail and report -999 or drift by several degrees.
- Farmers who apply little fertiliser tend to leave the field blank, so the gaps carry information.

A model trained on this data without preparation learns the errors. In the Week 2 experiment, a model trained on the raw table was **worse than predicting the crop average**.

## Week 2: Data preprocessing and feature engineering

1. **Builds a hypothetical dataset** and deliberately corrupts it with realistic defects at known rates, so every cleaning step can be scored.
2. **Cleans the data**: duplicates, spellings, units, physical-range checks, outliers, missing values.
3. **Engineers features** such as water balance, compound heat-and-drought stress and distance from optimal soil pH.
4. **Selects features**, reducing 56 candidates to 10.
5. **Tests the result** with leave-one-season-out validation (train on two seasons, predict the third).

| Pipeline | Model | Error (RMSE, t/ha) |
|---|---|---|
| Crop-average baseline | none | 0.749 |
| Raw, uncleaned data | Gradient boosting | 2.081 |
| Cleaned data | Gradient boosting | 0.412 |
| Cleaned + engineered features | Ridge regression | 0.372 |

- Cleaning cut error by about 80% compared with training on raw data.
- Simple rules (range checks, a median-based outlier score) found every injected error and beat Isolation Forest.
- Advanced imputation helped only where variables were related (NDVI).
- Engineered features helped the linear model but not the tree model.
- Reducing to 10 features cost about 2.7% in accuracy for 82% fewer inputs.

## Week 3: Model planning and evaluation metrics

Week 3 is a 29-page guide, `Week3_ML_Model_Development_Evaluation_Guide.docx`, backed by measured experiments. It also responds to the Week 2 feedback (too generic on unit mix-ups, spellings, spatial mismatch and temporal misalignment) with a dedicated section of worked examples.

**Covered in the guide**

1. Problem framing and success criteria (use cases, forecast origin, thresholds).
2. Data realities that shape the model:
   - Unit mix-ups: conversion tables and a unit-flag-free detection test
   - Inconsistent spellings: normalisation procedure and the limits of fuzzy matching
   - Spatial mismatch: field size against pixel and weather-cell footprints
   - Temporal misalignment: stage-aligned windows, a boundary off-by-one bug, forecast-origin leakage
3. A nine-stage development workflow with exit gates, and five data-splitting schemes.
4. Algorithm selection: weighted decision matrix and a ten-model bake-off.
5. Hyper-parameter tuning with nested cross-validation.
6. Evaluation metrics: regression, classification, domain-specific and cost-based.
7. Validation protocol, tested improvements, roadmap, risks and effort plan.

**Key results (leave-one-season-out, RMSE in t/ha)**

| Model | RMSE |
|---|---|
| Crop x region mean (stand-in for current method) | 0.756 |
| Elastic Net (champion) | 0.370 |
| Ridge | 0.372 |
| LightGBM | 0.406 |
| HistGradientBoosting | 0.415 |
| Random Forest | 0.448 |

- The champion cuts error by **51%** against the baseline (95% bootstrap interval 49.3% to 53.0%); the Week 1 target was 25%.
- Linear models win here because the synthetic yield is almost linear. Expect this to change on real data, so the plan keeps linear and boosted-tree models in competition.
- The same model's error ranged from 0.397 to 0.429 depending on the validation design; nested tuning gained about 1%.
- Acre-as-hectare errors affect 6.0% of records and overstate total production by 8.9%.
- Replacing field-level weather with one value per region-season raised error by 23%.
- Accuracy is misleading for the low-yield alert (a do-nothing rule scores 76.5%); precision, recall and F2 are reported instead.
- Regional production totals were off by 0.63% on average, against 11.5% for the baseline.
- Forecasting at the 25th percentile (assumed 3:1 over/under cost ratio) cut asymmetric cost by 18%.
- Previous-season yield as a feature and a simple three-model average did not help; these negative results are in the report.

## Repository contents

| File | Week | Purpose |
|---|---|---|
| `generate_data.py` | 2 | Creates the clean "truth" dataset and the messy raw version (seed 42) |
| `agri_pipeline.py` | 2 | Audit, cleaning, imputation benchmark, feature engineering, selection, evaluation; saves `results.json` and charts |
| `real_data_starter.py` | 2 | Example code for open data (NASA POWER, Sentinel-2 NDVI). **Untested**: adapt before use |
| `agri_raw_messy.csv` | 2 | Raw, defective dataset (1,845 rows) |
| `agri_clean_truth.csv` | 2 | True values, used for scoring |
| `agri_model_ready.csv` | 2 | Pipeline output: cleaned, imputed, engineered, encoded |
| `Week2_ML_Agribusiness_Data_Preprocessing_Report.docx` | 2 | Week 2 report |
| `Week3_ML_Model_Development_Evaluation_Guide.docx` | 3 | Week 3 report |
| `week3_data_checks.py` | 3 | Unit, spelling and spatial-footprint checks; writes `results_A.json` |
| `week3_experiments_models.py` | 3 | Split strategies, bake-off, nested tuning; writes `results_B1.json`, `preds.pkl` |
| `week3_experiments_evaluation.py` | 3 | Metrics, alerts, cost and intervals, spatial and temporal tests, improvement experiments; writes `results_B2.json` |
| `week3_make_figures.py` | 3 | Generates the Week 3 report figures into `figures/` |

## How to run

```bash
pip install pandas numpy scikit-learn scipy matplotlib lightgbm xgboost rapidfuzz
mkdir -p figures

# Week 2
python generate_data.py                      # optional: regenerate the datasets
python agri_pipeline.py                      # creates results.json and figures/

# Week 3 (run in this order; needs the Week 2 CSVs in the working directory)
python week3_data_checks.py
python week3_experiments_models.py           # slow: about 15-20 min on one CPU core
python week3_experiments_evaluation.py       # needs outputs of the previous script
python week3_make_figures.py
```

Seeds are fixed. The Week 3 scripts take about 30 minutes in total on one core.

## Key concepts

- **NDVI**: satellite vegetation index; higher means denser, healthier crops.
- **Missing at random vs. not at random**: whether a gap depends on other data or on the missing value itself.
- **Data leakage**: information the model would not have at prediction time, which makes results look better than they are.
- **Leave-one-season-out validation**: holding out a whole season in turn, the honest test for yield forecasting.
- **Nested cross-validation**: an outer loop for scoring and an inner loop for choosing settings.
- **Quantile forecast**: forecasting a chosen percentile (for example 25th) to match unequal costs of over- and under-forecasting.
- **F2 score**: balances precision and recall, weighting recall four times as much.

## Limitations

- The data are simulated and the yield relationships are mostly additive, which favours linear models.
- Some Week 2 preprocessing (imputer, outlier limits) was fitted on all rows, and feature selection used the 2023 season, which is also a test fold. Scores are slightly optimistic; production code should fit every step inside each training fold (first item on the Week 3 roadmap).
- Only three seasons exist, so every season contributed to model selection. Reported scores are development estimates, not a locked-test result.
- The imputation benchmark hides values at random, so it cannot measure bias from values missing not at random.
- Assumptions to replace with real figures: crop water demand (rice 900 mm, maize 600 mm), the 3:1 cost ratio, the 20th-percentile alert definition and the decision-matrix scores.
- The Week 3 spatial and temporal simulations show mechanisms, not magnitudes for real farms.
- `real_data_starter.py` was read but not run. Its stage-window slicing includes both endpoints, so boundary days are counted twice (fix described in the Week 3 report).
- References in the Week 3 report were cited from memory; verify details before formal use.

## Next steps

- Rebuild preprocessing inside each cross-validation fold and repeat all scores.
- Repeat the experiments on real or open data (district yields, NASA POWER or CHIRPS weather, MODIS or Sentinel-2 NDVI, SoilGrids) and document the cooperative's current estimation method as the real baseline.
- Add stage-aligned weather features and NDVI time-series features.
- Add spatial block validation with real coordinates, conformal prediction intervals and stacking of comparable models.
- Agree success thresholds and the cost ratio with the cooperative, then plan a shadow season before going live.

## Author

Kanish.
