"""This example demonstrates how to use the SHAP-IQ library to explain a TabPFN model.

WARNING: This example may run slowly on CPU-only systems.
For better performance, we recommend running with GPU acceleration.
SHAP value computation involves multiple TabPFN model evaluations, which is computationally intensive.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from tabpfn_extensions import TabPFNClassifier

# Import tabpfn adapters from interpretability module
from tabpfn_extensions.interpretability import shapiq as tabpfn_shapiq

# Load example dataset
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names

# Set the number of model evaluations for the explanation, Note this number is very low for
# demonstration and testing purposes and should be increased for a real use cases
n_model_evals = 100

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=200,
    random_state=42,
)
x_explain = X_test[0]

# Initialize and train model
clf = TabPFNClassifier()
clf.fit(X_train, y_train)

# Get a TabPFNExplainer
explainer = tabpfn_shapiq.get_tabpfn_explainer(
    model=clf,
    data=X_train,
    labels=y_train,
    index="SV",  # SV: Shapley Value (like in shap)
    verbose=True,  # show a progress bar during explanation
)

# Get shap values
print("Calculating SHAP values...")
shapley_values = explainer.explain(x=x_explain, budget=n_model_evals)

# plot the force plot
shapley_values.plot_force(feature_names=feature_names)

# Get an Shapley Interaction Explainer (here we use the Faithful Shapley Interaction Index)
explainer = tabpfn_shapiq.get_tabpfn_explainer(
    model=clf,
    data=X_train,
    labels=y_train,
    index="FSII",  # SV: Shapley Value, FSII: Faithful Shapley Interaction Index
    max_order=2,  # maximum order of the Shapley interactions (2 for pairwise interactions)
    verbose=True,  # show a progress bar during explanation
)

# Get shapley interaction values
print("Calculating Shapley interaction values...")
shapley_interaction_values = explainer.explain(x=x_explain, budget=n_model_evals)

# Plot the upset plot for visualizing the interactions
shapley_interaction_values.plot_upset(feature_names=feature_names)
