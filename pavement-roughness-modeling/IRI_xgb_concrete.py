#!/usr/bin/env python
# coding: utf-8



# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import xgboost as xgb
from skopt import BayesSearchCV
import shap
import joblib




# Read dataset from Excel file
data = pd.read_excel("IRI_concrete.xlsx")

# Get the name of the first column (assumed to be categorical)
first_column_name = data.columns[0]

# Perform one-hot encoding on the first column
data_encoded = pd.get_dummies(data, columns=[first_column_name], drop_first=False)

# Reorder the columns to bring the one-hot encoded columns to the front
# Extract the new one-hot encoded column names
encoded_columns = [col for col in data_encoded.columns if first_column_name in col]

# Get the remaining columns (continuous variables)
other_columns = [col for col in data_encoded.columns if col not in encoded_columns]

# Reorder columns: one-hot encoded columns followed by the remaining columns
data = data_encoded[encoded_columns + other_columns]

# Print the result
print(data)




# Separate features and target variable
# Horizontally concatenate original features and encoded features (if any)
X = data.iloc[:, :-1]
y = data.iloc[:, -1]
print(X.columns)
print(y)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[20]:


# Create StandardScaler object
scaler = StandardScaler()

# Fit and standardize the training set (starting from column index 3)
X_train.iloc[:, 3:] = scaler.fit_transform(X_train.iloc[:, 3:])

# Standardize the test set using the scaler fitted on the training set
X_test.iloc[:, 3:] = scaler.transform(X_test.iloc[:, 3:])

# Save the scaler to a local file
joblib.dump(scaler, 'scaler_concrete_bytype.joblib')

# Output the standardized data
print(pd.DataFrame(X_train, columns=X.columns))
print(pd.DataFrame(X_test, columns=X.columns))


# In[21]:


# Define monotonicity constraints
monotone_constraints = "(0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0)"

# Initialize XGBoost regressor
xgb_reg = xgb.XGBRegressor(monotone_constraints=monotone_constraints,
                           random_state=42)

# Define hyperparameter search space
param_space = {
    'n_estimators': (100, 1000),  # number of boosting rounds
    'max_depth': (3, 10),         # maximum tree depth
    'learning_rate': (0.01, 1.0, 'log-uniform'),  # learning rate
    'gamma': (0.01, 1.0, 'log-uniform'),          # minimum loss reduction required to make a further partition on a leaf node
    'min_child_weight': (0, 10),                  # minimum sum of instance weight needed in a child
    'subsample': (0.5, 1.0, 'uniform'),            # subsample ratio of the training instances
    'colsample_bytree': (0.5, 1.0, 'uniform'),    # subsample ratio of columns when constructing each tree
    'reg_alpha': (1e-5, 100, 'log-uniform'),       # L1 regularization term on weights
    'reg_lambda': (1e-5, 100, 'log-uniform')       # L2 regularization term on weights
}


# In[22]:


# Use Bayesian optimization for hyperparameter search
bayes_search = BayesSearchCV(
    xgb_reg,
    param_space,
    n_iter=50,       # number of iterations
    cv=5,            # number of cross-validation folds
    scoring='neg_mean_squared_error',
    random_state=42
)


# In[23]:


# Run the Bayesian optimization search
bayes_search.fit(X_train, y_train)

# Output the best hyperparameter combination
print("Best Parameters:", bayes_search.best_params_)


# In[24]:


# Predict using the model with the best hyperparameters
best_xgb_reg = bayes_search.best_estimator_
y_pred = best_xgb_reg.predict(X_test)

# Save the model to a local file
joblib.dump(best_xgb_reg, 'model_xgb_concrete_bytype.joblib')

# Compute mean squared error (MSE)
train_mse = mean_squared_error(y_train, best_xgb_reg.predict(X_train))
print("Training Mean Squared Error:", train_mse)

# Compute mean squared error (MSE)
test_mse = mean_squared_error(y_test, y_pred)
print("Testing Mean Squared Error:", test_mse)

# Compute mean absolute error (MAE)
train_mae = mean_absolute_error(y_train, best_xgb_reg.predict(X_train))
print("Training Mean Absolute Error:", train_mae)

# Compute mean absolute error (MAE)
test_mae = mean_absolute_error(y_test, y_pred)
print("Testing Mean Absolute Error:", test_mae)

# Compute R^2 score on the training set
train_r2 = r2_score(y_train, best_xgb_reg.predict(X_train))
print("Training R^2 Score:", train_r2)

# Compute R^2 score on the test set
test_r2 = r2_score(y_test, y_pred)
print("Testing R^2 Score:", test_r2)


# In[25]:


# Plot model fit
plt.figure(figsize=(5, 5))
plt.scatter(y_test, y_pred, color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='grey', linestyle='--')  # Plot diagonal line representing perfect fit
plt.title("XGBoost Regression - Model Fit")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.show()

# Compute prediction errors
errors = y_pred - y_test

# Plot histogram of prediction errors and fitted normal distribution curve
plt.figure(figsize=(8, 6))
sns.histplot(errors, bins=30, kde=False, stat='density', color='blue', alpha=0.6)

# Fit normal distribution curve
mean, std = norm.fit(errors)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean, std)
plt.plot(x, p, 'k', linewidth=2)

plt.title("Prediction Error Histogram and Normal Distribution Fit")
plt.xlabel("Prediction Error")
plt.ylabel("Density")
plt.show()

# Compute percentage prediction errors
errors_per = (y_pred - y_test) / y_test * 100

# Plot histogram of percentage prediction errors and fitted normal distribution curve
plt.figure(figsize=(8, 6))
sns.histplot(errors_per, bins=30, kde=False, stat='density', color='blue', alpha=0.6)

# Fit normal distribution curve
mean, std = norm.fit(errors_per)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean, std)
plt.plot(x, p, 'k', linewidth=2)

plt.title("Prediction errors percent Histogram and Normal Distribution Fit")
plt.xlabel("Prediction errors percent")
plt.ylabel("Density")
plt.show()

# Save y_test and y_pred to an Excel file
test_results_df = pd.DataFrame({'True Values': y_test, 'Predicted Values': y_pred})
test_results_df.to_excel('model_test_results_xgb_concrete_bytype.xlsx', index=False)


# In[26]:


# Plot model fit for training set
plt.figure(figsize=(5, 5))
plt.scatter(y_train, best_xgb_reg.predict(X_train), color='blue', alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='grey', linestyle='--')  # Plot diagonal line representing perfect fit
plt.title("XGBoost Regression - Model Fit")
plt.xlabel("True Values")
plt.ylabel("Predicted Values")
plt.show()

# Compute prediction errors on training set
errors = best_xgb_reg.predict(X_train) - y_train

# Plot histogram of prediction errors and fitted normal distribution curve
plt.figure(figsize=(8, 6))
sns.histplot(errors, bins=30, kde=False, stat='density', color='blue', alpha=0.6)

# Fit normal distribution curve
mean, std = norm.fit(errors)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean, std)
plt.plot(x, p, 'k', linewidth=2)

plt.title("Prediction Error Histogram and Normal Distribution Fit")
plt.xlabel("Prediction Error")
plt.ylabel("Density")
plt.show()

# Compute percentage prediction errors on training set
errors_per = (best_xgb_reg.predict(X_train) - y_train) / y_train * 100

# Plot histogram of percentage prediction errors and fitted normal distribution curve
plt.figure(figsize=(8, 6))
sns.histplot(errors_per, bins=30, kde=False, stat='density', color='blue', alpha=0.6)

# Fit normal distribution curve
mean, std = norm.fit(errors_per)
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mean, std)
plt.plot(x, p, 'k', linewidth=2)

plt.title("Prediction errors percent Histogram and Normal Distribution Fit")
plt.xlabel("Prediction errors percent")
plt.ylabel("Density")
plt.show()

# Save y_train and y_pred to an Excel file
train_results_df = pd.DataFrame({'True Values': y_train, 'Predicted Values': best_xgb_reg.predict(X_train)})
train_results_df.to_excel('model_train_results_xgb_concrete_bytype.xlsx', index=False)


# In[27]:


# Create SHAP explainer (best_xgb_reg is the best XGBoost model)
explainer = shap.TreeExplainer(best_xgb_reg)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test)

# Explain prediction for a single sample
sample_idx = 0  # choose one sample
sample_shap_values = shap_values[sample_idx]
sample = X_test.iloc[sample_idx]

# Visualize explanation
shap.initjs()
shap.force_plot(explainer.expected_value, sample_shap_values, sample)

# Set matplotlib parameters
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.dpi'] = 300

# Compute feature importance
fig, ax = plt.subplots(figsize=(20.07 / 2.54, 8.64 / 2.54))
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig('feature_importance_all.png', dpi=300, bbox_inches='tight')

# Output feature importance ranking
shap_summary = pd.DataFrame(shap_values, columns=X_test.columns)
feature_importance = shap_summary.abs().mean().sort_values(ascending=False)
print("Feature Importance:\n", feature_importance)


# In[28]:


# Set matplotlib parameters
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['figure.dpi'] = 300

# Compute feature importance
fig, ax = plt.subplots(figsize=(20.07 / 2.54, 8.64 / 2.54))
shap.summary_plot(shap_values, X_test, show=False)
plt.savefig('feature_importance2_all.png', dpi=300)

