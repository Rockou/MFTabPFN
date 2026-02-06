"""WARNING: This example may run slowly on CPU-only systems.
For better performance, we recommend running with GPU acceleration.
Feature selection involves training multiple TabPFN models, which is computationally intensive.
"""

from sklearn.datasets import load_breast_cancer

from tabpfn_extensions import TabPFNClassifier, interpretability

# Load data
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# Initialize model
clf = TabPFNClassifier(n_estimators=3)

# Feature selection
sfs = interpretability.feature_selection.feature_selection(
    estimator=clf,
    X=X,
    y=y,
    n_features_to_select=5,  # How many features to select
    feature_names=feature_names,
)

# Print selected features
selected_features = [
    feature_names[i] for i in range(len(feature_names)) if sfs.get_support()[i]
]
print("\nSelected features:")
for feature in selected_features:
    print(f"- {feature}")
