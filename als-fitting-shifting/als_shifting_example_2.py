#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd

# File paths
input_file = "Fitting_Parameters.xlsx"
output_file = "fitting_parameters_shifted_scenario_EV100_P1_HiBW_LoWA.xlsx"

# Load the input Excel file
df = pd.read_excel(input_file)

# Prepare lists to store results
shifted_single_als_c5 = []
shifted_single_als_c9 = []
shifted_tandem_als = []

# Process each row in the dataframe
for index, row in df.iterrows():
    # Extract SHRP_ID for reference
    shrp_id = row['SHRP_ID']
    
    print(f"\nSection No: {index+1}, SHRP_ID: {shrp_id}")

    # Step 1: Single axle shift for C5-sin
    print(f"\nSingle axle shift for C5-sin of SHRP_ID: {shrp_id}")
    mean_c5, std_dev_c5 = row['C5-sin-mu'], row['C5-sin-sigma']
    fitted_mean_c5, fitted_std_dev_c5 = single_als_shift(mean=mean_c5,
                                                         std_dev=std_dev_c5,
                                                         num_samples=100000,
                                                         x_shift=1000,
                                                         pen_rate=1)
    shifted_single_als_c5.append([shrp_id, fitted_mean_c5, fitted_std_dev_c5])

    # Step 2: Single axle shift for C9-sin
    print(f"\nSingle axle shift for C9-sin of SHRP_ID: {shrp_id}")
    mean_c9, std_dev_c9 = row['C9-sin-mu'], row['C9-sin-sigma']
    fitted_mean_c9, fitted_std_dev_c9 = single_als_shift(mean=mean_c9,
                                                         std_dev=std_dev_c9,
                                                         num_samples=100000,
                                                         x_shift=4409.2,
                                                         pen_rate=1)
    shifted_single_als_c9.append([shrp_id, fitted_mean_c9, fitted_std_dev_c9])

    # Step 3: Tandem axle shift for C9-tan
    print(f"\nTandem axle shift for C9-tan of SHRP_ID: {shrp_id}")
    mean1, std_dev1, weight1 = row['C9-tan-mu1'], row['C9-tan-sigma1'], row['C9-tan-weight1']
    mean2, std_dev2, weight2 = row['C9-tan-mu2'], row['C9-tan-sigma2'], row['C9-tan-weight2']
    (
        fitted_mean1,
        fitted_std_dev1,
        fitted_weight1,
        fitted_mean2,
        fitted_std_dev2,
        fitted_weight2,
    ) = tandem_als_shift(
        mean1=mean1,
        std_dev1=std_dev1,
        weight1=weight1,
        mean2=mean2,
        std_dev2=std_dev2,
        weight2=weight2,
        num_samples=100000,
        x_shift1=-3409.2,
        x_shift2=1000,
        pen_rate=1
    )
    shifted_tandem_als.append([
        shrp_id,
        fitted_mean1,
        fitted_std_dev1,
        fitted_weight1,
        fitted_mean2,
        fitted_std_dev2,
        fitted_weight2,
    ])
    print(f"\nall axle load of SHRP_ID: {shrp_id} shifted!!!")

# Create a new DataFrame for shifted parameters
df_shifted_c5 = pd.DataFrame(
    shifted_single_als_c5,
    columns=['SHRP_ID', 'C5-sin-mu', 'C5-sin-sigma'],
)

df_shifted_c9 = pd.DataFrame(
    shifted_single_als_c9,
    columns=['SHRP_ID', 'C9-sin-mu', 'C9-sin-sigma'],
)

df_shifted_tandem = pd.DataFrame(
    shifted_tandem_als,
    columns=[
        'SHRP_ID',
        'C9-tan-mu1',
        'C9-tan-sigma1',
        'C9-tan-weight1',
        'C9-tan-mu2',
        'C9-tan-sigma2',
        'C9-tan-weight2',
    ],
)

# Merge all shifted data into a single DataFrame
df_shifted = df_shifted_c5.merge(df_shifted_c9, on='SHRP_ID').merge(df_shifted_tandem, on='SHRP_ID')

# Save the shifted parameters to a new Excel file
df_shifted.to_excel(output_file, index=False)

