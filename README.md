# Preparing Noisy Farm Data for Machine Learning

**Data preprocessing and feature engineering for crop yield forecasting in agribusiness**

This project shows how to turn messy, real-world agribusiness data into a clean, model-ready table, and measures how much that preparation improves a yield-forecasting model. It is the Week 2 deliverable of an internship project on *Machine Learning for Crop Yield Forecasting and Input Optimization in Agribusiness*.

## The problem

An agribusiness such as a farmer cooperative has to decide how much to buy, store, transport and fertilise long before harvest. A good yield forecast makes those decisions cheaper and less risky. But the data behind a forecast is rarely clean:

- Harvest weights are typed by hand, so a decimal slip turns 3.4 t/ha into 34 t/ha.
- Some farms report area in acres and others in hectares.
- Satellite images are hidden by monsoon cloud, leaving gaps in the vegetation data.
- Sensors fail and report −999 or drift by several degrees.
- Farmers who apply little fertiliser tend to leave the field blank, so the gaps carry information.

A model trained on this data without preparation learns the errors. In this project's experiment, a model trained on the raw table was **worse than just predicting the crop average**.

## What this project does

1. **Builds a hypothetical dataset** (600 fields, 3 seasons, rice and maize) and deliberately corrupts it with realistic defects at known rates. Because the true values are known, every cleaning step can be scored.
2. **Cleans the data**: removes duplicates, standardises spellings and units, checks physical ranges, detects outliers and fills missing values.
3. **Engineers features** that reflect how crops respond to their environment, such as water balance, compound heat-and-drought stress and distance from optimal soil pH.
4. **Selects features**, reducing 56 candidates to 10.
5. **Tests the result** by training on two seasons and predicting the third (leave-one-season-out), which mimics forecasting a season the model has never seen.

## Key findings

| Pipeline | Model | Error (RMSE, t/ha) |
|---|---|---|
| Crop-average baseline | none | 0.749 |
| Raw, uncleaned data | Gradient boosting | 2.081 |
| Cleaned data | Gradient boosting | 0.412 |
| Cleaned + engineered features | Ridge regression | 0.372 |

- Cleaning cut error by about 80% compared with training on raw data.
- Simple rules (range checks, a median-based outlier score) found every injected error, and beat Isolation Forest.
- Advanced imputation helped only where variables were related (NDVI); for soil pH and nitrogen applied, no method beat the plain mean.
- Engineered features helped the linear model but not the tree model, which learns interactions on its own.
- Reducing to 10 features cost about 2.7% in accuracy for 82% fewer inputs.

> **Caution:** the data are synthetic. These results show how well each *method* works, not the accuracy to expect on a real farm.

## Repository contents

| File | Purpose |
|---|---|
| `generate_data.py` | Creates the clean "truth" dataset and the messy raw version (reproducible, seed 42) |
| `agri_pipeline.py` | Runs the audit, cleaning, imputation benchmark, feature engineering, selection and evaluation; saves `results.json` and charts |
| `real_data_starter.py` | Example code for real open data (NASA POWER weather, Sentinel-2 NDVI). **Untested**: adapt before use |
| `agri_raw_messy.csv` | The raw, defective dataset (1,845 rows) |
| `agri_clean_truth.csv` | The true values, used for scoring and fair test labels |
| `agri_model_ready.csv` | The pipeline output: cleaned, imputed, engineered and encoded |
| `requirements.txt` | Python dependencies |
| `ML_Agribusiness_Week2_Data_Preprocessing_Report.docx` | The full written report with methods, tables and justification |

## How to run

```bash
pip install -r requirements.txt
python generate_data.py     # optional: regenerate the datasets
python agri_pipeline.py     # run the full pipeline; creates results.json and figures/
```

## Key concepts

- **NDVI**: a satellite-based vegetation index; higher means denser, healthier crops.
- **Missing at random vs. not at random**: whether a gap depends on other data or on the missing value itself. This decides how it should be filled.
- **Winsorising**: capping extreme values instead of deleting them.
- **Data leakage**: information from the test data influencing training, which makes results look better than they are.
- **Leave-one-season-out validation**: holding out a whole season in turn, the honest test for yield forecasting.

## Limitations

- The data are simulated, and the yield relationships are mostly additive, which favours the linear model.
- For simplicity, some preprocessing steps (imputer, outlier limits) were fitted on all seasons, and feature selection used the 2023 season, which is also a test fold. This makes the selected-features result slightly optimistic. A production version should fit every step inside each training fold.
- The imputation benchmark hides values at random, so it cannot measure bias from values missing not at random.
- Crop water demand constants (rice 900 mm, maize 600 mm) are planning assumptions.

## Next steps

- Repeat the experiments on real or open data (district yields, NASA POWER or CHIRPS weather, MODIS or Sentinel-2 NDVI, SoilGrids).
- Add NDVI time-series features (smoothing, peak date, cumulative greenness) and growth-stage weather summaries.
- Re-implement the pipeline with leakage-safe cross-validation and region-level hold-out testing.

## Author

Kanish.
