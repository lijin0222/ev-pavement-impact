# ev-pavement-impact

Code (and demo data) for the study: **“Electric vehicles accelerate pavement damage but deliver net climate and cost benefits”**.

This repository is organized to match the paper workflow:
1) **Fitting & shifting axle load spectra (ALS)** under EV scenarios  
2) **Pavement survival analysis** (RSF) under EV traffic  
3) **Pavement roughness (IRI) modeling** (XGBoost) under EV traffic  

Repository structure (three modules):
- `als-fitting-shifting/` — fits parametric ALS models and applies Monte Carlo-based shifting under EV scenarios
- `pavement-survival-analysis/` — Random Survival Forest (RSF) time-to-threshold modeling (example: asphalt IRI)
- `pavement-roughness-modeling/` — XGBoost regression for IRI evolution (asphalt and concrete)

> Each module includes **core scripts** plus **small demo Excel datasets** so you can run examples without the full LTPP extraction pipeline.

---

## Quickstart (demo run)

### 1) Create a Python environment

Recommended: Python 3.9–3.11.

**Option A — conda (recommended for scikit-survival):**
```bash
conda create -n ev_pavement python=3.10 -y
conda activate ev_pavement
conda install -c conda-forge numpy pandas scipy scikit-learn matplotlib joblib -y
conda install -c conda-forge scikit-survival lifelines optuna -y
pip install xgboost scikit-optimize shap seaborn
Run each module (from its folder)

1) ALS fitting & shifting

cd als-fitting-shifting
python als_fitting_example.py
python als_shifting_example_1.py
python als_shifting_example_2.py

2) Survival analysis (RSF)

cd ../pavement-survival-analysis
python RSF_IRI_asphalt.py

3) IRI modeling (XGBoost)

cd ../pavement-roughness-modeling
python IRI_xgb_asphalt.py
python IRI_xgb_concrete.py
Reproducibility notes (important)

Some scripts were exported from notebooks and may require small path edits.

Absolute paths: If a script saves outputs to a user-specific absolute path, replace it with a local relative path (e.g., ./outputs/). See the module READMEs for details.

Runtime: Optuna / Bayesian tuning can be slow. For a quick smoke test, reduce:

Optuna n_trials

BayesSearchCV n_iter
