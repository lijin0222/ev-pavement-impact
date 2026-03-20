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
```

## Step 1 — Fit ALS distributions (demo)

Run in this folder:

```bash
python als_fitting_example.py
```

### Inputs

`axle load spectra_GPS-1.xlsx`

The script groups by `SHRP_ID` and expects fields like:

- `SHRP_ID`
- `VEHICLE_CLASS`, `VEHICLE_CLASS_EXP`
- `AXLE_GROUP`, `AXLE_GROUP_EXP`
- histogram bin/count columns (the demo file matches the script layout)

### Outputs

- `Axle_load_spectra_<SHRP_ID>.png` (one per section)
- `Fitting_Parameters_GPS-1.xlsx` (fitted parameters)

## Step 2 — Shift ALS under EV scenarios (demo)

The example scripts call `single_als_shift(...)` and `tandem_als_shift(...)`.  
If running as standalone scripts, add the imports at the top:

```python
from als_shifting_single import single_als_shift
from als_shifting_tandem import tandem_als_shift
```

## Example: EV50 (50% penetration)

```bash
python als_shifting_example_1.py
````

* Uses `pen_rate=0.5`
* Writes: `fitting_parameters_shifted_scenario_EV50_P1_HiBW_LoWA.xlsx`

## Example: EV100 (100% penetration)

```bash
python als_shifting_example_2.py
```

* Uses `pen_rate=1.0`
* Writes: `fitting_parameters_shifted_scenario_EV100_P1_HiBW_LoWA.xlsx`

## How to define a new EV scenario

Edit (in `als_shifting_example_*.py`):

* `pen_rate` (EV penetration rate, e.g., 0.25/0.5/1.0)
* `x_shift` values for different axle groups (battery mass / load redistribution assumption)
* `output_file` naming

Then run the script to export a scenario-specific shifted parameter Excel.

