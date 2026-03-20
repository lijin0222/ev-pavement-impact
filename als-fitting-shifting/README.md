# ALS fitting & shifting (EV scenarios)

This module corresponds to the paper section **“Fitting and shifting of ALS”**.
It fits parametric distributions to axle load spectra (ALS) and applies Monte Carlo-based shifting to emulate EV adoption.

## What’s included
Core scripts:
- `als_fitting_example.py` — fit unimodal (single axle) and bimodal (tandem axle) distributions and export fitted parameters
- `als_shifting_single.py` — function `single_als_shift(...)` for shifting single-axle ALS (Gaussian)
- `als_shifting_tandem.py` — function `tandem_als_shift(...)` for shifting tandem-axle ALS (mixture of Gaussians)
- `als_shifting_example_1.py` — example EV scenario (EV50; pen_rate=0.5)
- `als_shifting_example_2.py` — example EV scenario (EV100; pen_rate=1.0)

Demo data:
- `axle load spectra_GPS-1.xlsx` — demo ALS inputs for fitting
- `Fitting_Parameters.xlsx` — demo fitted-parameter table used as shifting input

## Dependencies
```bash
pip install numpy pandas scipy scikit-learn matplotlib
