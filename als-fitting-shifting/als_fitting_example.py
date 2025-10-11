#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

# Read data
input_file = "axle load spectra_GPS-1.xlsx"
df = pd.read_excel(input_file)

# List to store results
results = []

# Group by each SHRP_ID
grouped = df.groupby('SHRP_ID')

for shrp_id, group in grouped:
    # Calculate subplot rows and columns (3 plots per row, distribute evenly)
    num_plots = 9
    rows = (num_plots // 3) + (1 if num_plots % 3 else 0)
    
    # Create figure and axes
    fig, axs = plt.subplots(rows, 3, figsize=(15, 5 * rows))
    axs = axs.flatten()

    fig.suptitle(f"Axle load spectra-{shrp_id}", fontsize=16)
    ax_idx = 0

    for _, row in group.iterrows():
        # Get current row values
        vehicle_class = row['VEHICLE_CLASS']
        vehicle_class_exp = row['VEHICLE_CLASS_EXP']
        axle_group = row['AXLE_GROUP']
        axle_group_exp = row['AXLE_GROUP_EXP']
        
        # Extract columns 8 to 47 and ensure numeric type
        counts = pd.to_numeric(row.iloc[7:47], errors='coerce').dropna()
        frequencies = counts/counts.sum()
        
        if counts.empty or (vehicle_class not in [5, 6, 8, 9] or axle_group not in [1, 2]):
            continue  # skip rows that don't meet conditions

        # Single-peak (unimodal) fitting
        if vehicle_class in [5, 6, 8, 9] and axle_group == 1:
            interval_midpoints = np.arange(500, 39501, 1000)
            try:
                # Generate approximate data points from frequencies
                data = np.repeat(interval_midpoints, counts)

                # Fit a normal distribution to estimate mean and std
                mean, std_dev = norm.fit(data)

                # Generate normal PDF using the fitted mean and std
                x = np.linspace(min(interval_midpoints), max(interval_midpoints), 1000)
                discrete_pdf = norm.pdf(x, loc=mean, scale=std_dev) * sum(frequencies) * (interval_midpoints[1] - interval_midpoints[0])
                
                # Plot data and fitted curve
                ax = axs[ax_idx]
                ax.scatter(interval_midpoints, frequencies, color='b', label='Data', marker='o')
                ax.plot(x, discrete_pdf, 'r-', label='Single Gaussian Fit')
                ax.set_title(f"{shrp_id}-CLASS {vehicle_class}-{axle_group_exp}")
                ax.legend()
                ax_idx += 1

                # Record parameters for unimodal fit
                results.append({
                    "SHRP_ID": shrp_id,
                    "VEHICLE_CLASS": vehicle_class,
                    "VEHICLE_CLASS_EXP": vehicle_class_exp,
                    "AXLE_GROUP": axle_group,
                    "AXLE_GROUP_EXP": axle_group_exp,
                    "mu": mean,
                    "sigma": std_dev,
                    "mu1": None, "sigma1": None, "weight1": None,
                    "mu2": None, "sigma2": None, "weight2": None
                })

            except RuntimeError:
                print(f"Unimodal fit failed: SHRP_ID {shrp_id}-CLASS {vehicle_class}-{axle_group_exp}")
                ax = axs[ax_idx]
                ax.scatter(interval_midpoints, frequencies, color='b', label='Data', marker='o')
                ax.set_title(f"{shrp_id}-CLASS {vehicle_class}-{axle_group_exp}")
                ax.legend()
                ax_idx += 1
                continue

        # Two-peak (bimodal) fitting
        elif vehicle_class in [6, 8, 9] and axle_group == 2:
            interval_midpoints = np.arange(1000, 79001, 2000)
            try:
                # Generate approximate data points from frequencies
                data = np.repeat(interval_midpoints, counts)

                # Define and fit Gaussian Mixture Model with 2 components (bimodal)
                gmm = GaussianMixture(n_components=2, random_state=0)
                gmm.fit(data.reshape(-1, 1))
                
                # Extract means, std devs, and weights
                means = gmm.means_.flatten()
                std_devs = np.sqrt(gmm.covariances_).flatten()
                weights = gmm.weights_

                # Convert continuous PDF to discrete counts
                x = np.linspace(min(interval_midpoints), max(interval_midpoints), 1000)
                pdf = (weights[0] * norm.pdf(x, means[0], std_devs[0]) +
                       weights[1] * norm.pdf(x, means[1], std_devs[1]))

                # Compute approximate PDF counts for each interval
                discrete_pdf = pdf * sum(frequencies) * (interval_midpoints[1] - interval_midpoints[0]) # multiply by interval width and scale to discrete counts

                # Plot data and fitted curve
                ax = axs[ax_idx]
                ax.scatter(interval_midpoints, frequencies, color='b', label='Data', marker='o')
                ax.plot(x, discrete_pdf, 'g-', label='Double Gaussian Fit')
                ax.set_title(f"{shrp_id}-CLASS {vehicle_class}-{axle_group_exp}")
                ax.legend()
                ax_idx += 1

                # Assign mu1/sigma1/weight1 to the smaller mean component
                if means[0] < means[1]:
                    mu1 = means[0]
                    sigma1 = std_devs[0]
                    weight1 = weights[0]
                    mu2 = means[1]
                    sigma2 = std_devs[1]
                    weight2 = weights[1]
                else:
                    mu1 = means[1]
                    sigma1 = std_devs[1]
                    weight1 = weights[1]
                    mu2 = means[0]
                    sigma2 = std_devs[0]
                    weight2 = weights[0]
                
                # Record parameters for bimodal fit
                results.append({
                    "SHRP_ID": shrp_id,
                    "VEHICLE_CLASS": vehicle_class,
                    "VEHICLE_CLASS_EXP": vehicle_class_exp,
                    "AXLE_GROUP": axle_group,
                    "AXLE_GROUP_EXP": axle_group_exp,
                    "mu": None, "sigma": None,
                    "mu1": mu1, "sigma1": sigma1, "weight1": weight1,
                    "mu2": mu2, "sigma2": sigma2, "weight2": weight2
                })

            except RuntimeError:
                print(f"Bimodal fit failed: SHRP_ID {shrp_id}-CLASS {vehicle_class}-{axle_group_exp}")
                ax = axs[ax_idx]
                ax.scatter(interval_midpoints, frequencies, color='b', label='Data', marker='o')
                ax.set_title(f"{shrp_id}-CLASS {vehicle_class}-{axle_group_exp}")
                ax.legend()
                ax_idx += 1
                continue

    # Remove empty subplots
    for i in range(ax_idx, len(axs)):
        fig.delaxes(axs[i])
    
    # Save figure
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    output_file = f"Axle_load_spectra_{shrp_id}.png"
    plt.savefig(output_file)
    plt.close(fig)
    print(f"Figure for SHRP_ID {shrp_id} saved to {output_file}")

# Save fitting results to Excel file
output_params_file = "Fitting_Parameters_GPS-1.xlsx"
results_df = pd.DataFrame(results)
results_df.to_excel(output_params_file, index=False)
print(f"Fitting parameters saved to {output_params_file}")

