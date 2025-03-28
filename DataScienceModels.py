# -*- coding: utf-8 -*-

# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV


# Load dataset
file_path = r'C:\Users\iains\Documents\Data Science\Processed Data.xlsx'
df = pd.read_excel(file_path)

# Define the columns on the dataset
df.columns = ["School", "Year", "UG_PG", "FT_PT", "Campus", "Teams", "R_R", "Unitu", "SKYB", "Sessions_Attended", "Engaged"]

# Store label encoders
label_encoders = {}  

# Encode categorical variables
label_cols = ["School", "Year", "UG_PG", "FT_PT", "Campus", "Teams"]
df_encoded = df.copy()
for col in label_cols:
    le = LabelEncoder()
    df_encoded[col] = le.fit_transform(df[col])  # Encode column
    label_encoders[col] = le  # Store the fitted encoder
    
# Define Features (X) and Target (y)
X = df_encoded.drop(columns=["Sessions_Attended", "R_R", "Unitu", "SKYB", "Engaged"])  # Features, ignoring columns "Sessions_Attended", "R_R", "Unitu", "SKYB"
#y = df_encoded["Sessions_Attended"]  # Target variable (0, 1, 2, or 3 training sessions attended)
y = df_encoded["Engaged"]  # Target variable (TRUE or FALSE that training sessions were attended)

# Split into Training and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Data loaded and preprocessed successfully!")

# ------- Decision Tree --------
from sklearn.tree import DecisionTreeClassifier

# Train Decision Tree Model
dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
dt_model.fit(X_train, y_train)

print("Decision Tree model trained successfully!")

from sklearn.tree import plot_tree

# Plot Decision Tree
plt.figure(figsize=(15, 10))
#plot_tree(dt_model, feature_names=X.columns, class_names=["0", "1", "2", "3"], filled=True)
plot_tree(dt_model, feature_names=X.columns, class_names=["TRUE", "FALSE"], filled=True)
plt.title("Decision Tree Visualization")
plt.show()


# -------- Random Forest --------

from sklearn.ensemble import RandomForestClassifier

# Find the best parameters
rf_param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rf_grid_search = GridSearchCV(estimator=RandomForestClassifier(random_state=42),
                              param_grid=rf_param_grid,
                              cv=5,  # 5-fold cross-validation
                              n_jobs=-1,  # Use all processors
                              verbose=2)
rf_grid_search.fit(X_train, y_train)

# Best parameters and score
print("Best Random Forest Parameters:", rf_grid_search.best_params_)
print("Best Random Forest Score:", rf_grid_search.best_score_)

# Use the best model
rf_best_model = rf_grid_search.best_estimator_


# Train Random Forest Model
rf_model = rf_best_model
rf_model.fit(X_train, y_train)

print("Random Forest model trained successfully!")

# Feature Importance for Random Forest
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(10, 5))
plt.bar(range(X.shape[1]), importances[indices], align="center")
plt.xticks(range(X.shape[1]), X.columns[indices], rotation=90)
plt.title("Random Forest Feature Importance")
plt.show()

# ---------- XGBoost ----------
import xgboost as xgb

# Find the best parameters
xgb_param_grid = {
    'n_estimators': [50, 100, 200],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 5, 7],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}


xgb_grid_search = GridSearchCV(estimator=xgb.XGBClassifier(random_state=42),
                               param_grid=xgb_param_grid,
                               cv=5,  # 5-fold cross-validation
                               n_jobs=-1,  # Use all processors
                               verbose=2)
xgb_grid_search.fit(X_train, y_train)

# Best parameters and score
print("Best XGBoost Parameters:", xgb_grid_search.best_params_)
print("Best XGBoost Score:", xgb_grid_search.best_score_)

# Use the best model
xgb_best_model = xgb_grid_search.best_estimator_


# Train XGBoost Model
xgb_model = xgb_best_model
xgb_model.fit(X_train, y_train)

print("XGBoost model trained successfully!")

# Feature Importance for XGBoost
xgb.plot_importance(xgb_model, importance_type="gain", title="XGBoost Feature Importance")
plt.show()

# ------ Testing --------
# Make Predictions
y_pred_dt = dt_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)
y_pred_xgb = xgb_model.predict(X_test)

# Print Accuracy Scores
print("Decision Tree Accuracy:", accuracy_score(y_test, y_pred_dt))
print("Random Forest Accuracy:", accuracy_score(y_test, y_pred_rf))
print("XGBoost Accuracy:", accuracy_score(y_test, y_pred_xgb))

# Print Classification Reports
print("\nDecision Tree Report:\n", classification_report(y_test, y_pred_dt))
print("\nRandom Forest Report:\n", classification_report(y_test, y_pred_rf))
print("\nXGBoost Report:\n", classification_report(y_test, y_pred_xgb))

# Plot confusion matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Decision Tree Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_dt), annot=True, fmt="d", cmap="Blues", ax=axes[0])
axes[0].set_title("Decision Tree")

# Random Forest Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_rf), annot=True, fmt="d", cmap="Blues", ax=axes[1])
axes[1].set_title("Random Forest")

# XGBoost Confusion Matrix
sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, fmt="d", cmap="Blues", ax=axes[2])
axes[2].set_title("XGBoost")

plt.show()

# -------- Predictions ------------
# Predict whether students will attend training (Engaged = TRUE or FALSE)
df_encoded["Predicted_Engagement"] = xgb_model.predict(X)

# Filter students predicted NOT to attend training (Engaged = FALSE)
at_risk_students = df_encoded[df_encoded["Predicted_Engagement"] == 0].copy()

# Convert back categorical labels using stored encoders
for col in label_encoders.keys():
    at_risk_students[col] = label_encoders[col].inverse_transform(at_risk_students[col])

# Show at-risk students
print("Students predicted to NOT attend training:")
print(at_risk_students)

















