#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sksurv.util import Surv
from sksurv.metrics import concordance_index_censored
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sksurv.ensemble import RandomSurvivalForest
from sksurv.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
import optuna
import joblib



# Read Excel file
data = pd.read_excel("survival_IRI_asphalt.xlsx")

# Preview data format
print(data.head())



# Convert SurvTime and EventStatus columns to a Surv object
y = Surv.from_dataframe("EventStatus", "SurvTime", data)

# Explanatory variables (exclude ID columns)
X = data.drop(columns=["SHRP_ID", "Exp type", "SurvTime", "EventStatus"])

# Use OneHotEncoder to handle categorical variables (if any)
encoder = OneHotEncoder()
X_encoded = encoder.fit_transform(X)

# Split the dataset into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)


# In[5]:

# Create a StandardScaler object
scaler = StandardScaler()

# Fit and standardize the training set
X_train.iloc[:, :] = scaler.fit_transform(X_train.iloc[:, :])

# Standardize the test set using the scaler fitted on the training set
X_test.iloc[:, :] = scaler.transform(X_test.iloc[:, :])

# Save the scaler to a local file
joblib.dump(scaler, 'scaler_RSF_IRI_asphalt.joblib')

# Output the standardized data
print(pd.DataFrame(X_train, columns=X.columns))
print(pd.DataFrame(X_test, columns=X.columns))


# In[6]:

def cross_val_score_rsf(params, X, y, n_splits=5):
    """
    Perform k-fold cross-validation for the given hyperparameters and dataset,
    and return the average concordance index (C-index).

    Parameters:
        params: dict, containing model hyperparameters.
        X: pandas.DataFrame, explanatory variables.
        y: sksurv.util.Surv, response variable (survival data).
        n_splits: int, number of cross-validation folds (default 5).

    Returns:
        mean_c_index: float, mean concordance index.
    """
    # Initialize cross-validation splitter
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    c_indices = []
    for train_index, val_index in kf.split(X):
        # Split training and validation sets
        X_train, X_val = X.iloc[train_index], X.iloc[val_index]
        y_train, y_val = y[train_index], y[val_index]
        
        # Create and train the model
        model = RandomSurvivalForest(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_split=params["min_samples_split"],
            min_samples_leaf=params["min_samples_leaf"],
            random_state=42,
        )
        model.fit(X_train, y_train)
        
        # Compute C-index on the validation set
        c_index = concordance_index_censored(
            y_val["EventStatus"], y_val["SurvTime"], model.predict(X_val)
        )[0]
        c_indices.append(c_index)
    
    # Return mean C-index
    mean_c_index = np.mean(c_indices)
    return mean_c_index


# In[7]:

# Define the objective function for Optuna
def objective(trial):
    # Define the search space
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
        "max_depth": trial.suggest_int("max_depth", 3, 20),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
    }
    
    # Compute the mean C-index using k-fold cross-validation
    mean_c_index = cross_val_score_rsf(params, X_train, y_train, n_splits=5)
    
    # Return negative C-index because Optuna minimizes the objective by default
    return -mean_c_index

# Create an Optuna study and run optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Output best C-index and parameters
print("Best C-index:", study.best_value)
print("Best parameters:", study.best_params)


# In[8]:

# Retrieve best parameters
best_params = study.best_params

# Define the model using the best parameters
rsf = RandomSurvivalForest(
    n_estimators=best_params["n_estimators"],
    max_depth=best_params["max_depth"],
    min_samples_split=best_params["min_samples_split"],
    min_samples_leaf=best_params["min_samples_leaf"],
    random_state=42,
)

# Train the model on the training set
rsf.fit(X_train, y_train)

# Save the model
joblib.dump(rsf, "model_RSF_IRI_asphalt.joblib")


# In[9]:
# Concordance index (C-index)
# The C-index is a common evaluation metric in survival analysis that measures
# the model's ability to rank survival times. Its range is typically 0.5 to 1.0:
# 1.0 indicates perfect prediction.
# 0.5 indicates random guessing.

# Compute C-index on training and test sets
c_index_train = concordance_index_censored(
    y_train["EventStatus"], y_train["SurvTime"], rsf.predict(X_train)
)[0]

c_index_test = concordance_index_censored(
    y_test["EventStatus"], y_test["SurvTime"], rsf.predict(X_test)
)[0]

print(f"C-index on training data: {c_index_train:.4f}")
print(f"C-index on test data: {c_index_test:.4f}")


# In[10]:
# Survival curve fitting quality
# You can visualize actual and predicted survival curves to check how close they are.
# Get actual and predicted survival curves for some samples in the test set
survival_func_pred = rsf.predict_survival_function(X_test)
for i, func in enumerate(survival_func_pred[:5]):  # show only the first 5 samples
    plt.step(func.x, func.y, where="post", label=f"Sample {i+1}")

plt.title("Predicted Survival Curves")
plt.xlabel("Time")
plt.ylabel("Survival Probability")
plt.legend()
plt.show()


# In[11]:
import numpy as np
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
import matplotlib

# Use Arial font
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['axes.facecolor'] = '#f7f7f7'
matplotlib.rcParams['grid.color'] = 'white'  # white grid lines

# Compute Kaplan-Meier survival curve
kmf = KaplanMeierFitter()
kmf.fit(y_test['SurvTime'], event_observed=y_test['EventStatus'])

# Plot
plt.figure(figsize=(4, 4))

# Plot Kaplan-Meier survival curve
plt.step(kmf.survival_function_.index, kmf.survival_function_['KM_estimate'], 
         where="post", label="Kaplan-Meier", color='blue', linestyle='--', linewidth=2)

# Add Kaplan-Meier confidence interval
plt.fill_between(kmf.confidence_interval_.index, 
                 kmf.confidence_interval_['KM_estimate_lower_0.95'], 
                 kmf.confidence_interval_['KM_estimate_upper_0.95'], 
                 color='blue', alpha=0.1, label="KM 95% CI")

# Compute RSF mean survival curve
mean_survival = np.mean([func.y for func in survival_func_pred], axis=0)
plt.step(survival_func_pred[0].x, mean_survival, where="post", label="RSF Mean Prediction", color="green", linestyle='-', linewidth=2)

# Set x and y axis limits
plt.ylim(-0.05, 1.05)
plt.xlim(-2, 52)

# Beautify plot
plt.xlabel("Pavement age (years)", fontsize=12)
plt.ylabel("Survival probability", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Move legend to lower left
plt.legend(frameon=False, fontsize=12, loc='lower left')

plt.grid(True, linestyle='-', linewidth=0.5)  # white grid lines
plt.savefig('survival_IRI_asphalt1.png', dpi=600, bbox_inches='tight')
plt.show()


# In[12]:
# Use Arial font
matplotlib.rcParams['font.family'] = 'Arial'
matplotlib.rcParams['axes.facecolor'] = '#f7f7f7'  # default gray background in R
matplotlib.rcParams['grid.color'] = 'white'  # white grid lines

# Predict survival curves for an individual
survival_func = rsf.predict_survival_function(X_test.iloc[0:-1])  # get survival curves for first test samples

# Plot survival curves
plt.figure(figsize=(4, 4))

for s in survival_func:
    plt.step(s.x, s.y, where="post")
# plt.title("Survival Curve", fontsize=14)
plt.xlabel("Pavement age (years)", fontsize=12)
plt.ylabel("Survival probability", fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# Set x and y axis limits
plt.ylim(-0.05, 1.05)
plt.xlim(-2, 52)

# Adjust legend
plt.legend(frameon=False, fontsize=12)  # remove frame
plt.grid(True, linestyle='-', linewidth=0.5)  # white dashed grid
plt.savefig('survival_IRI_asphalt2.png', dpi=600, bbox_inches='tight')
plt.show()


# In[13]:
import pandas as pd
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt

# Compute feature importance
result = permutation_importance(rsf, X_train, y_train, n_repeats=5, random_state=42)
feature_importance = result.importances_mean
std_importance = result.importances_std

# Create DataFrame
importance_df = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': feature_importance,
    'Standard Deviation': std_importance
})

# Save to Excel file
output_path = "feature_importance_RSF_IRI_asphalt.xlsx"
importance_df.to_excel(output_path, index=False)
print(f"Feature importance saved to {output_path}")

# Print feature importances
print("Feature Importances:")
print(importance_df)

# Plot feature importance
plt.barh(X_train.columns, feature_importance, xerr=std_importance)
plt.xlabel('Importance')
plt.title('Feature Importance')
plt.show()