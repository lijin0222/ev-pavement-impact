#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

def tandem_als_shift(mean1, std_dev1, weight1,
                     mean2, std_dev2, weight2,
                     num_samples=1000000,
                     x_shift1=2000, x_shift2=2000,
                     pen_rate=0.5):
    """
    Perform a Monte Carlo simulation for a tandem axle scenario, applying a Gaussian Mixture Model (GMM).

    Parameters:
        mean1 (float): Mean of the first normal distribution.
        std_dev1 (float): Standard deviation of the first normal distribution.
        weight1 (float): Mixing weight of the first normal distribution.
        mean2 (float): Mean of the second normal distribution.
        std_dev2 (float): Standard deviation of the second normal distribution.
        weight2 (float): Mixing weight of the second normal distribution.
        num_samples (int): Number of samples for Monte Carlo simulation (default: 1,000,000).
        x_shift1 (float): Value to add to a portion of samples from distribution 1 (default: 2000 lbs.).
        x_shift2 (float): Value to add to a portion of samples from distribution 2 (default: 2000 lbs.).
        pen_rate (float): Penetration rate, proportion of samples in each distribution to shift (default: 50%).

    Returns:
        tuple: Fitted parameters of the modified GMM: 
            fitted_mean1, fitted_std_dev1, fitted_weight1,
            fitted_mean2, fitted_std_dev2, fitted_weight2.
    """
    # Generate samples from the two normal distributions
    n1 = int(num_samples * weight1)
    n2 = int(num_samples * weight2)
    samples1 = np.random.normal(mean1, std_dev1, n1)
    samples2 = np.random.normal(mean2, std_dev2, n2)

    # Prepare copies for modification
    modified_samples1 = samples1.copy()
    modified_samples2 = samples2.copy()

    # Determine number of points to shift
    k1 = int(n1 * pen_rate)
    k2 = int(n2 * pen_rate)
    
    # Shift distribution 1 if x_shift1 is non-zero
    if x_shift1 != 0:
        indices1 = np.random.choice(n1, k1, replace=False)
        modified_samples1[indices1] += x_shift1

    # Shift distribution 2 if x_shift2 is non-zero
    if x_shift2 != 0:
        indices2 = np.random.choice(n2, k2, replace=False)
        modified_samples2[indices2] += x_shift2

    # Combine original/modified samples1 with original/modified samples2
    original_samples = np.concatenate([samples1, samples2])
    modified_samples = np.concatenate([modified_samples1, modified_samples2])

    # Fit GMM to the original samples
    gmm_original = GaussianMixture(n_components=2)
    gmm_original.fit(original_samples.reshape(-1, 1))

    # Fit GMM to the modified samples
    gmm_modified = GaussianMixture(n_components=2)
    gmm_modified.fit(modified_samples.reshape(-1, 1))

    # Extract parameters from the modified GMM
    fitted_means = gmm_modified.means_.flatten()
    fitted_std_devs = np.sqrt(gmm_modified.covariances_).flatten()
    fitted_weights = gmm_modified.weights_

    # Sort fitted_means and rearrange std_devs and weights accordingly
    sorted_idx = np.argsort(fitted_means)
    fitted_mean1 = fitted_means[sorted_idx[0]]
    fitted_std_dev1 = fitted_std_devs[sorted_idx[0]]
    fitted_weight1 = fitted_weights[sorted_idx[0]]
    fitted_mean2 = fitted_means[sorted_idx[1]]
    fitted_std_dev2 = fitted_std_devs[sorted_idx[1]]
    fitted_weight2 = fitted_weights[sorted_idx[1]]

    # Print original distribution parameters
    print("\nOriginal Distribution Parameters:")
    print("Original Mean1:", mean1)
    print("Original Std Dev1:", std_dev1)
    print("Original Weight1:", weight1)
    print("Original Mean2:", mean2)
    print("Original Std Dev2:", std_dev2)
    print("Original Weight2:", weight2)
    
    # Print modified distribution parameters
    print("\nModified Distribution Parameters:")
    print("Modified Mean1:", fitted_mean1)
    print("Modified Std Dev1:", fitted_std_dev1)
    print("Modified Weight1:", fitted_weight1)
    print("Modified Mean2:", fitted_mean2)
    print("Modified Std Dev2:", fitted_std_dev2)
    print("Modified Weight2:", fitted_weight2)

    # # Visualization
    # interval = 2000
    # x_bins = np.arange(0, 80001, interval)  # Define bins for histograms

    # # Plot the histograms
    # counts_orig, _ = np.histogram(original_samples, bins=x_bins)
    # counts_mod, _ = np.histogram(modified_samples, bins=x_bins)

    # x_mid = (x_bins[:-1] + x_bins[1:]) / 2
    # freq_orig = counts_orig / num_samples
    # freq_mod = counts_mod / num_samples

    # plt.figure(figsize=(10, 6))
    # plt.bar(x_mid, freq_orig, width=interval, alpha=0.6, label='Original Distribution')
    # plt.bar(x_mid, freq_mod, width=interval, alpha=0.6, label='Modified Distribution')

    # x_vals = np.linspace(min(original_samples.min(), modified_samples.min()),
    #                      max(original_samples.max(), modified_samples.max()), 1000)
    # orig_fit = np.exp(gmm_original.score_samples(x_vals.reshape(-1, 1))) * interval
    # mod_fit = np.exp(gmm_modified.score_samples(x_vals.reshape(-1, 1))) * interval

    # plt.plot(x_vals, orig_fit, linestyle='-', linewidth=2, label='Original GMM Fit')
    # plt.plot(x_vals, mod_fit, linestyle='-', linewidth=2, label='Modified GMM Fit')

    # plt.xlabel('X Value')
    # plt.ylabel('Probability Density')
    # plt.legend()
    # plt.title('Original vs Modified GMM Distribution')
    # plt.show()

    return (fitted_mean1, fitted_std_dev1, fitted_weight1,
            fitted_mean2, fitted_std_dev2, fitted_weight2)

# # Example
# tandem_als_shift(
#     mean1=14459.5952777771, std_dev1=3852.29541508114, weight1=0.506238670748942,
#     mean2=32416.3962393595, std_dev2=5723.784053086, weight2=0.493761329251058,
#     num_samples=1000000,
#     x_shift1=-2000, x_shift2=0,
#     pen_rate=0.5
# )

