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
