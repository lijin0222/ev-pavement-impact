# Pavement survival analysis under EV traffic (RSF)

This module corresponds to the paper section **“Pavement survival analysis under EV traffic”**.
It trains a Random Survival Forest (RSF) to link traffic/ALS descriptors to time-to-threshold failure events.

## What’s included
- `RSF_IRI_asphalt.py` — RSF training + evaluation + example survival-curve plots (example: asphalt IRI)
- `survival_IRI_asphalt.xlsx` — demo survival dataset
- `README.md` — this guide

## Dependencies (recommended via conda)
```bash
conda install -c conda-forge numpy pandas scikit-learn matplotlib joblib -y
conda install -c conda-forge scikit-survival lifelines optuna -y
```
````markdown id="rsf_demo_md"
## Run (demo)

From this folder:

```bash
python RSF_IRI_asphalt.py
````

## Input data schema (demo)

The script reads:

* `survival_IRI_asphalt.xlsx`

Required columns:

* `SurvTime` — time-to-event or censoring time (years)
* `EventStatus` — event indicator (1=failed, 0=censored)
* `SHRP_ID` — section identifier
* `Exp_type` — optional categorical column used in the demo
* All remaining columns are treated as predictors (traffic/ALS/structure proxies, etc.)

In the demo script, predictors are constructed as:

```python
X = data.drop(columns=["SHRP_ID", "Exp_type", "SurvTime", "EventStatus"])
y = Surv.from_dataframe("EventStatus", "SurvTime", data)
```

## Outputs

* `model_RSF_IRI_asphalt.joblib` — trained RSF model
* `scaler_RSF_IRI_asphalt.joblib` — fitted scaler
* Example figures (e.g., predicted survival curves)

## Extending to other distresses / pavement types

To run RSF for other performance metrics (e.g., fatigue/rutting for asphalt, or IRI for concrete):

1. Create a corresponding survival dataset Excel with the same schema (`SurvTime`, `EventStatus`, predictors).
2. Duplicate `RSF_IRI_asphalt.py` and update:

   * input filename
   * any predictor column exclusions
3. Re-train and export the new model.

## Runtime tips

The demo uses Optuna hyperparameter tuning (`n_trials`). For a quick test, reduce the trial count.

```
```
