import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def single_als_shift(mean, std_dev, num_samples=1000000, weight1=0.5, weight2=0.5, x1_shift=2000, x2_shift=2000, pen_rate=0.5):
    """
    Perform a Monte Carlo simulation with specified parameters, 
    shifting a portion of samples and fitting a normal distribution.

    Parameters:
        mean (float): Mean of the original normal distribution.
        std_dev (float): Standard deviation of the original normal distribution.
        num_samples (int): Number of samples for Monte Carlo simulation (default: 1,000,000).
        x1_shift & x2_shift (float): Value to add to a portion of samples (default: 2000 lbs.).
        pen_rate (float): Penetration rate, proportion of samples to shift (default: 50%).

    Returns:
        tuple: Fitted mean and standard deviation of the modified samples.
    """
    # Generate Monte Carlo samples from the original normal distribution
    samples = np.random.normal(mean, std_dev, num_samples)

    # Create a copy of samples to modify
    modified_samples = samples.copy()

    # Determine number of points to shift
    k = int(num_samples * pen_rate)

    # Randomly select a subset of samples based on the penetration rate if x1_shift and x2_shift is non-zero
    if x1_shift != 0 or x2_shift != 0:
        indices = np.random.choice(num_samples, k, replace=False)
    
        half_k = int(k * weight1)
        indices1 = indices[:half_k]
        indices2 = indices[half_k:]
        
        if x1_shift != 0:
            modified_samples[indices1] += x1_shift
    
        if x2_shift != 0:
            modified_samples[indices2] += x2_shift
        
    # Fit a normal distribution to the modified samples
    fitted_mean, fitted_std_dev = norm.fit(modified_samples)
    
    # Print original distribution parameters
    print("\nOriginal Distribution Parameters:")
    print("Original Mean:", mean)
    print("Original Std Dev:", std_dev)

    # Print modified distribution parameters
    print("\nModified Distribution Parameters:")
    print("Fitted Mean:", fitted_mean)
    print("Fitted Std Dev:", fitted_std_dev)

    # Visualization
    interval = 1000
    x_bins = np.arange(0, 40001, interval)  # Define bins for histograms

    # Calculate frequency histograms
    counts_samples, _ = np.histogram(samples, bins=x_bins)
    counts_modified_samples, _ = np.histogram(modified_samples, bins=x_bins)

    # Calculate midpoints for the bins
    x_midpoints = (x_bins[:-1] + x_bins[1:]) / 2

    # Convert counts to frequencies
    freq_samples = counts_samples / num_samples
    freq_modified_samples = counts_modified_samples / num_samples

    # Plot the histograms
    plt.figure(figsize=(10, 6))

    # Original distribution
    plt.bar(x_midpoints, freq_samples, width=interval, alpha=0.6, color='blue', label='Original Distribution')

    # Modified distribution
    plt.bar(x_midpoints, freq_modified_samples, width=interval, alpha=0.6, color='orange', label='Modified Distribution')

    # Plot normal distribution fits
    x_values = np.linspace(min(samples.min(), modified_samples.min()), max(samples.max(), modified_samples.max()), 1000)
    original_fit = norm.pdf(x_values, np.mean(samples), np.std(samples)) * interval
    plt.plot(x_values, original_fit, 'blue', linestyle='-', linewidth=2, label='Original Normal Fit')

    modified_fit = norm.pdf(x_values, fitted_mean, fitted_std_dev) * interval
    plt.plot(x_values, modified_fit, 'orange', linestyle='-', linewidth=2, label='Modified Normal Fit')

    # Configure plot
    plt.xlabel('X Value')
    plt.ylabel('Probability Density')
    plt.legend()
    plt.title('Original vs Modified Distribution')

    # Display plot
    plt.show()

    return fitted_mean, fitted_std_dev

# Example
single_als_shift(mean=9052.97738969874, std_dev=3480.56171928128, num_samples=1000000, x1_shift=2204.6, x2_shift=1000, pen_rate=0.5)
