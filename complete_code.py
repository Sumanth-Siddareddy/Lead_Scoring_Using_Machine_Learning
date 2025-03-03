import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import subprocess
import shutil
import zipfile

# For pipelines, model selection, and metrics
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, roc_curve, classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_selection import RFE
from sklearn.calibration import calibration_curve

# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

# Optional: Handling imbalanced data
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# 1.1 Load the dataset
df = pd.read_csv("../Leads X Education.csv")

# 1.2 Check the basic information of the dataframe
print("Shape of the dataset:", df.shape)
df.info()

# 2.1 Look at the first few rows
df.head()

# 2.2 Statistical summary of numeric columns
df.describe()

# 2.3 Check the columns
df.columns


cols_to_drop = ["Prospect ID", "Lead Number"]
df.drop(columns=cols_to_drop, inplace=True, errors="ignore")

print("Columns after dropping IDs:", df.columns.tolist())

df.replace("Select", np.nan, inplace=True)
# 5.1 Check Missing Value Counts
df.isnull().sum().sort_values(ascending=False)

missing_counts = df.isnull().sum().sort_values(ascending=False)
print(missing_counts)

# 5.2 Drop
threshold = 0.50
cols_to_drop = [col for col in df.columns 
                if df[col].isnull().mean() > threshold]
df.drop(columns=cols_to_drop, inplace=True)

print("Dropped columns with >50% missing values:", cols_to_drop)

# 5.3 Impute Categorical Columns
# fill missing in a categorical column with "Unknown"
categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    df[col].fillna("Unknown", inplace=True)

# 5.4 Impute Numeric Columns
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Impute Numeric with median
for col in numeric_cols:
    df[col].fillna(df[col].median(), inplace=True)

# Example if they were loaded as strings
bool_cols = ["Do Not Email", "Do Not Call"]  # Add more if needed
for col in bool_cols:
    # Convert "Yes"/"No" or "True"/"False" strings to boolean
    df[col] = df[col].map({"Yes": True, "No": False, "True": True, "False": False})

# Identify remaining categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns

# One-Hot Encoding (Pandas get_dummies)
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

print("Shape after preprocessing:", df_encoded.shape)
df_encoded.head()



X = df_encoded.drop("Converted", axis=1)
y = df_encoded["Converted"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print("Train set size:", X_train.shape, y_train.shape)
print("Test set size:", X_test.shape, y_test.shape)

### 2. Define a common evaluation function

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate model performance and plot metrics."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None

    auc_str = f"{auc:.4f}" if auc is not None else "N/A"

    print(f"Model: {model_name}")
    print(classification_report(y_test, y_pred, zero_division=0))
    print(f"Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1: {f1:.4f}, ROC AUC: {auc_str}")

    # Confusion Matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, annot_kws={"fontsize": 21})
    plt.xlabel("Predicted", fontsize=24)
    plt.ylabel("Actual", fontsize=24)
    plt.title(f"Confusion Matrix: {model_name}", fontsize=24)
    plt.tight_layout()
    plt.savefig(f"Confusion_Matrix_{model_name}.png", dpi=900)
    plt.close()

    # ROC curve plot
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC curve (area = {auc_str})', linewidth=3)
        plt.plot([0, 1], [0, 1], 'k--', linewidth=3)
        plt.xlabel('False Positive Rate', fontsize=24)
        plt.ylabel('True Positive Rate', fontsize=24)
        plt.title(f"ROC Curve: {model_name}", fontsize=24)
        plt.legend(fontsize=21)
        plt.tight_layout()
        plt.savefig(f"ROC_Curve_{model_name}.png", dpi=900)
        plt.close()

    return {"model": model_name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc}


### 3. Define Parameter Grids for Each Model

param_grid_knn = {'clf__n_neighbors': [3, 5, 7, 9]}
param_grid_svm = {'clf__C': [0.1, 1, 10],
                  'clf__gamma': [0.001, 0.01, 0.1],
                  'clf__kernel': ['rbf']}
param_grid_dt = {'clf__max_depth': [None, 5, 10, 20],
                 'clf__min_samples_split': [2, 5, 10]}
param_grid_rf = {'clf__n_estimators': [100, 200],
                 'clf__max_depth': [None, 5, 10],
                 'clf__min_samples_split': [2, 5]}

### 4. Build Pipelines with Explicit PCA (using explicit K values)
# Modify build_pipeline to accept a PCA component number (k)
def build_pipeline(model, k):
    pipeline = ImbPipeline(steps=[
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=k)),
        ('smote', SMOTE(random_state=42)),
        ('clf', model)
    ])
    return pipeline

## PCA which column uses most for which Prinicpal component
# # Assume pca_step is your fitted PCA step from the pipeline
pca_step = best_est.named_steps['pca']
# The PCA components matrix (each row corresponds to a principal component)
loadings = pca_step.components_

# Get the original feature names from your training data
original_features = X_train.columns

# For each principal component, print the top 3 features by absolute loading value
n_top = 3
for i, component in enumerate(loadings):
    # Get indices of top absolute loadings
    top_indices = np.argsort(np.abs(component))[-n_top:][::-1]
    top_features = original_features[top_indices]
    top_loadings = component[top_indices]
    print(f"Principal Component {i+1}:")
    for feat, load in zip(top_features, top_loadings):
        print(f"    {feat}: {load:.4f}")
 
# Hyper parameter tuning
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define models and their parameter grids in a dictionary
models_to_evaluate = {
    "KNN": (KNeighborsClassifier(), param_grid_knn),
    "SVM": (SVC(probability=True, random_state=42), param_grid_svm),
    "DecisionTree": (DecisionTreeClassifier(random_state=42), param_grid_dt),
    "RandomForest": (RandomForestClassifier(random_state=42), param_grid_rf)
}

# List of explicit K values to try
k_values = [3, 6, 9, 15, 20, 25, 30]

# Store results and best estimators for each (model, k) combination
results = []
best_estimators = {}

for k in k_values:
    for model_name, (model, param_grid) in models_to_evaluate.items():
        pipeline = build_pipeline(model, k)
        grid = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train, y_train)
        best_est = grid.best_estimator_
        best_params = grid.best_params_
        
        # Print the PCA column names
        pca_step = best_est.named_steps['pca']
        n_components = pca_step.n_components_
        pc_names = [f"PC{i+1}" for i in range(n_components)]
        print(f"{model_name} with K={k} uses PCA columns: {pc_names}")
        
        # Evaluate the best estimator for this model and PCA setting
        res = evaluate_model(best_est, X_test, y_test, f"{model_name}_K{k}")
        res["k"] = k
        res["model"] = model_name
        res["best_params"] = best_params
        results.append(res)
        best_estimators[(model_name, k)] = best_est
        
        print(f"{model_name} with K={k} done. Best Params: {best_params}")


results_df = pd.DataFrame(results)
results_df.to_csv("PCA_ExplicitK_Results.csv", index=False)
print("Results saved to PCA_ExplicitK_Results.csv")


# Save trained models

import os
os.makedirs("Trained_Models", exist_ok=True)
for (model_name, k), model in best_estimators.items():
    filename = f"Trained_Models/{model_name}_K{k}_model.pkl"
    with open(filename, "wb") as file:
        pickle.dump(model, file)
    print(f"Saved {model_name} with K={k} model to {filename}")

# 7. Plot Accuracy vs. Number of Principal Components for Each Model
plt.figure(figsize=(12, 8))
for model_name in models_to_evaluate.keys():
    subset = results_df[results_df["model"] == model_name]
    plt.plot(subset["k"], subset["accuracy"], marker="o", linewidth=3, label=model_name)
plt.xlabel("Number of Principal Components (K)", fontsize=24)
plt.ylabel("Accuracy", fontsize=24)
plt.title("Accuracy vs. Number of Principal Components", fontsize=28)
plt.legend(fontsize=21)
plt.xticks(fontsize=21)
plt.yticks(fontsize=21)
plt.tight_layout()
plt.savefig("Accuracy_vs_K_Explicit.png", dpi=900)
plt.show()

# 8. Build Ensemble Models: Voting and Stacking (Using the Best Model for Each Type from One Chosen K, e.g., K=15)


# For ensemble, select a representative K value. Here we choose K=15.
selected_k = 15
ensemble_estimators = []
for model_name, (model, param_grid) in models_to_evaluate.items():
    best_model = best_estimators.get((model_name, selected_k))
    if best_model is not None:
        ensemble_estimators.append((model_name, best_model))

# Voting Ensemble (soft voting)
from sklearn.ensemble import VotingClassifier
voting_clf = VotingClassifier(estimators=ensemble_estimators, voting='soft', n_jobs=-1)
voting_clf.fit(X_train, y_train)
voting_results = evaluate_model(voting_clf, X_test, y_test, "VotingEnsemble")

with open("VotingEnsemble_model.pkl", "wb") as f:
    pickle.dump(voting_clf, f)

# Stacking Ensemble (using Logistic Regression as meta-model)
from sklearn.ensemble import StackingClassifier
stacking_clf = StackingClassifier(
    estimators=ensemble_estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=cv, n_jobs=-1
)
stacking_clf.fit(X_train, y_train)
stacking_results = evaluate_model(stacking_clf, X_test, y_test, "StackingEnsemble")

with open("StackingEnsemble_model.pkl", "wb") as f:
    pickle.dump(stacking_clf, f)

# 9. Compare All Models' Performance and Plot a Line Graph

# Define a mapping from model name to the chosen K
best_k_per_model = {
    "KNN": 15,
    "DecisionTree": 15,
    "SVM": 25,
    "RandomForest": 25
}

ensemble_estimators = []
for model_name, (model, param_grid) in models_to_evaluate.items():
    # Fetch the best estimator for the chosen K
    chosen_k = best_k_per_model[model_name]
    best_model = best_estimators.get((model_name, chosen_k))
    if best_model is not None:
        ensemble_estimators.append((model_name, best_model))
        print(f"Using {model_name} with K={chosen_k} for the ensemble.")

# Now build the Voting and Stacking ensembles
from sklearn.ensemble import VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression

# Voting Ensemble (soft voting)
voting_clf = VotingClassifier(
    estimators=ensemble_estimators,
    voting='soft',
    n_jobs=-1
)
voting_clf.fit(X_train, y_train)
voting_results = evaluate_model(voting_clf, X_test, y_test, "VotingEnsemble")

with open("VotingEnsemble_model.pkl", "wb") as f:
    pickle.dump(voting_clf, f)

# Stacking Ensemble (using Logistic Regression as meta-model)
stacking_clf = StackingClassifier(
    estimators=ensemble_estimators,
    final_estimator=LogisticRegression(max_iter=1000),
    cv=cv, 
    n_jobs=-1
)
stacking_clf.fit(X_train, y_train)
stacking_results = evaluate_model(stacking_clf, X_test, y_test, "StackingEnsemble")

with open("StackingEnsemble_model.pkl", "wb") as f:
    pickle.dump(stacking_clf, f)


# 10. Save Performance Metrics and Classification Reports to a Text File

import pandas as pd
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# If you have an in-memory list 'all_results' (results from individual models + ensemble),
# make sure it is defined before this code block.
# For example:
all_results_df = pd.DataFrame(all_results)

# Print the results table
print("Comparison of All Models:")
print(all_results_df)

# Define the metrics to plot
metrics = ['accuracy', 'precision', 'recall', 'f1', 'auc']
results_long = all_results_df.melt(id_vars='model', value_vars=metrics,
                                   var_name='metric', value_name='value')

# Create a line plot with no confidence interval shading
plt.figure(figsize=(12, 8))
sns.lineplot(
    data=results_long,
    x='model',
    y='value',
    hue='metric',
    marker='o',       # Markers on each data point
    linewidth=3,      # Thicker lines
    markersize=8,     # Larger marker size
    ci=None           # No shaded confidence interval
)

plt.title("Model Performance Comparison", fontsize=24)
plt.xlabel("Model", fontsize=24)
plt.ylabel("Metric Value", fontsize=24)
plt.xticks(fontsize=21, rotation=45)
plt.yticks(fontsize=21)
plt.legend(fontsize=21, title="Metric")
plt.tight_layout()
plt.savefig("Model_Comparison_Line_NoShade.png", dpi=900)
plt.show()
plt.close()


# Write all results and detailed classification reports to a text file
with open("model_performance.txt", "w") as f:
    f.write("Comparison of All Models:\n")
    f.write(all_results_df.to_string())
    f.write("\n\n")
    
    # Write detailed classification reports for each individual model.
    # This loops over all best_estimators (which is a dictionary keyed by (model_name, k))
    for (model_name, k), model in best_estimators.items():
        y_pred = model.predict(X_test)
        report = classification_report(y_test, y_pred, zero_division=0)
        f.write(f"Classification Report for {model_name} (K={k}):\n")
        f.write(report)
        f.write("\n" + "="*80 + "\n\n")
    
    # For Voting and Stacking ensembles
    y_pred_voting = voting_clf.predict(X_test)
    report_voting = classification_report(y_test, y_pred_voting, zero_division=0)
    f.write("Classification Report for Voting Ensemble:\n")
    f.write(report_voting)
    f.write("\n" + "="*80 + "\n\n")
    
    y_pred_stacking = stacking_clf.predict(X_test)
    report_stacking = classification_report(y_test, y_pred_stacking, zero_division=0)
    f.write("Classification Report for Stacking Ensemble:\n")
    f.write(report_stacking)
    f.write("\n" + "="*80 + "\n")
    
print("Performance metrics and classification reports saved to 'model_performance.txt'")