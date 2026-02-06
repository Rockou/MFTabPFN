# ----------------------------
# Install and Import Libraries
# ----------------------------
import numpy as np
from sklearn.datasets import fetch_openml, load_diabetes
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.rf_pfn import (
    RandomForestTabPFNClassifier,
    RandomForestTabPFNRegressor,
)

# ----------------------------
# Load and Preprocess Classification Dataset
# ----------------------------
print("\n--- Loading Classification Dataset ---")
df_classification = fetch_openml(
    "electricity",
    version=1,
    as_frame=True,
)
X_class, y_class = df_classification.data, df_classification.target

le = LabelEncoder()
y_class = le.fit_transform(y_class)

for col in X_class.select_dtypes(["category"]).columns:
    X_class.loc[:, col] = X_class.loc[:, col].cat.codes

X_train_class, X_test_class, y_train_class, y_test_class = train_test_split(
    X_class,
    y_class,
    test_size=0.33,
    random_state=42,
)


# ----------------------------
# Classification - Strategy 1: Tree-based Model with Node Subsampling
# ----------------------------
print("\n--- Classification: Strategy 2 (Tree-based Model) ---")
clf_base = TabPFNClassifier(
    ignore_pretraining_limits=True,  # (bool) Allows training on larger datasets.
    inference_config={
        "SUBSAMPLE_SAMPLES": 10000,  # (int) Number of samples to subsample for inference at each node.
    },
)

tabpfn_tree_clf = RandomForestTabPFNClassifier(
    tabpfn=clf_base,  # (TabPFNClassifier) Base TabPFN model to be used within the Random Forest structure.
    verbose=1,  # (int) Controls the verbosity; higher values show more details.
    max_predict_time=60,  # (int) Maximum prediction time allowed in seconds.
)

tabpfn_tree_clf.fit(X_train_class, y_train_class)
prediction_probabilities = tabpfn_tree_clf.predict_proba(X_test_class)
predictions = np.argmax(prediction_probabilities, axis=1)

print(f"ROC AUC: {roc_auc_score(y_test_class, prediction_probabilities[:, 1]):.4f}")
print(f"Accuracy: {accuracy_score(y_test_class, predictions):.4f}")

# ----------------------------
# Classification - Strategy 2: Subsampled Ensemble using TabPFNClassifier
# ----------------------------
print("\n--- Classification: Strategy 1 (Subsampled Ensemble) ---")
tabpfn_subsample_clf = TabPFNClassifier(
    ignore_pretraining_limits=True,  # (bool) Allows the use of datasets larger than pretraining limits.
    n_estimators=32,  # (int) Number of estimators for ensembling; improves accuracy with higher values.
    inference_config={
        "SUBSAMPLE_SAMPLES": 10000,  # (int) Maximum number of samples per inference step to manage memory usage.
    },
)

tabpfn_subsample_clf.fit(X_train_class, y_train_class)
prediction_probabilities = tabpfn_subsample_clf.predict_proba(X_test_class)
predictions = np.argmax(prediction_probabilities, axis=1)

print(f"ROC AUC: {roc_auc_score(y_test_class, prediction_probabilities[:, 1]):.4f}")
print(f"Accuracy: {accuracy_score(y_test_class, predictions):.4f}")

# ----------------------------
# Load and Preprocess Regression Dataset
# ----------------------------
print("\n--- Loading Regression Dataset ---")
regression_data = load_diabetes()
X_reg, y_reg = regression_data.data, regression_data.target

X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_reg,
    y_reg,
    test_size=0.33,
    random_state=42,
)

# ----------------------------
# Regression - Strategy 1: Tree-based Model with Node Subsampling
# ----------------------------
print("\n--- Regression: Strategy 2 (Tree-based Model) ---")
reg_base = TabPFNRegressor(
    ignore_pretraining_limits=True,  # (bool) Allows use with datasets larger than those used during pretraining.
    inference_config={
        "SUBSAMPLE_SAMPLES": 10000,  # (int) Number of samples to subsample per node to manage memory usage.
    },
)

tabpfn_tree_reg = RandomForestTabPFNRegressor(
    tabpfn=reg_base,  # (TabPFNRegressor) Base regressor used in the Random Forest architecture.
    verbose=1,  # (int) Level of verbosity for detailed logs.
    max_predict_time=60,  # (int) Upper time limit for prediction in seconds.
)

tabpfn_tree_reg.fit(X_train_reg, y_train_reg)
reg_tree_predictions = tabpfn_tree_reg.predict(X_test_reg)

print(
    f"Mean Squared Error (MSE): {mean_squared_error(y_test_reg, reg_tree_predictions):.4f}",
)
print(
    f"Mean Absolute Error (MAE): {mean_absolute_error(y_test_reg, reg_tree_predictions):.4f}",
)
print(f"R^2 Score: {r2_score(y_test_reg, reg_tree_predictions):.4f}")


# ----------------------------
# Regression - Strategy 2: Subsampled Ensemble using TabPFNRegressor
# ----------------------------
print("\n--- Regression: Strategy 1 (Subsampled Ensemble) ---")
tabpfn_subsample_reg = TabPFNRegressor(
    ignore_pretraining_limits=True,  # (bool) Enables handling datasets beyond pretraining constraints.
    n_estimators=32,  # (int) Number of estimators in the ensemble for robustness.
    inference_config={
        "SUBSAMPLE_SAMPLES": 10000,  # (int) Controls sample subsampling per inference to avoid OOM errors.
    },
)

tabpfn_subsample_reg.fit(X_train_reg, y_train_reg)
reg_predictions = tabpfn_subsample_reg.predict(X_test_reg)

print(
    f"Mean Squared Error (MSE): {mean_squared_error(y_test_reg, reg_predictions):.4f}",
)
print(
    f"Mean Absolute Error (MAE): {mean_absolute_error(y_test_reg, reg_predictions):.4f}",
)
print(f"R^2 Score: {r2_score(y_test_reg, reg_predictions):.4f}")
