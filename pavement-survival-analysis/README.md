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
