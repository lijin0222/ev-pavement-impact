# Pavement IRI modeling under EV traffic (XGBoost)

This module corresponds to the paper section **“Pavement IRI modeling under EV traffic”**.
It trains XGBoost regression models for IRI using LTPP-derived predictors.
The scripts use monotonicity constraints to enforce physically plausible trends.

## What’s included
Asphalt:
- `IRI_xgb_asphalt.py`
- `IRI_asphalt.xlsx` (demo data)

Concrete:
- `IRI_xgb_concrete.py`
- `IRI_concrete.xlsx` (demo data)

## Dependencies
```bash
pip install numpy pandas scipy scikit-learn matplotlib joblib
pip install xgboost scikit-optimize shap seaborn
````markdown id="xgb_demo_md"
## Run (demo)

From this folder:

```bash
python IRI_xgb_asphalt.py
python IRI_xgb_concrete.py
````

## Input data schema (demo)

### Asphalt

* The script reads `IRI_asphalt.xlsx`
* Convention in the script: `X = data.iloc[:, :-1]` and `y = data.iloc[:, -1]`

  * i.e., last column is the target IRI, all previous columns are features.

### Concrete

* The script reads `IRI_concrete.xlsx`
* The first column is treated as categorical (one-hot encoded), and the last column is the target IRI.

## Outputs

### Asphalt

* `model_xgb_asphalt.joblib` — trained model
* a saved scaler (note: update any absolute path to a relative path)
* evaluation plots and `model_train_results_xgb_asphalt.xlsx` (demo export)

### Concrete

* `model_xgb_concrete_bytype.joblib`
* `scaler_concrete_bytype.joblib`
* evaluation plots / metrics

## Notes on monotonicity constraints

Both scripts define `monotone_constraints = "(...)"` and pass it to `xgb.XGBRegressor(...)`.
If you change the feature set or column order, you must update the constraint string to match the new feature order.

## Runtime tips

Hyperparameter search uses Bayesian optimization (`BayesSearchCV`) with `n_iter`. Reduce `n_iter` for a quick smoke test.
