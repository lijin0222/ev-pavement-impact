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
