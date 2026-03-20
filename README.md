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
```

**Option B — pip only (may be harder for scikit-survival on some OS):**
```bash
pip install numpy pandas scipy scikit-learn matplotlib joblib xgboost scikit-optimize shap seaborn optuna lifelines scikit-survival
```

### 2) Run each module (from its folder)
**ALS fitting & shifting**
```bash
cd als-fitting-shifting
python als_fitting_example.py
python als_shifting_example_1.py
python als_shifting_example_2.py
```

**Survival analysis (RSF)**
```bash
cd ../pavement-survival-analysis
python RSF_IRI_asphalt.py
```

**IRI modeling (XGBoost)**
```bash
cd ../pavement-roughness-modeling
python IRI_xgb_asphalt.py
python IRI_xgb_concrete.py
```

## Notes on reproducibility

- Some scripts were exported from notebooks and may contain plotting/output sections intended for demonstration.
- Paths: If you see absolute local paths (e.g., saving a scaler to a user-specific directory), replace them with relative paths (e.g., `./outputs/`).
- Runtime: hyperparameter search can be compute-intensive (Optuna trials / Bayesian search iterations). For a quick smoke test, reduce:
  - `n_trials` in Optuna
  - `n_iter` in BayesSearchCV

---

## Using your own data

Demo Excel files are included in each module. For your own datasets:

- Keep the same column names expected by the scripts (see each module README for schema).
- Put your input Excel next to the script (or update `pd.read_excel(...)` paths accordingly).

---

## License

MIT License (see `LICENSE`).
