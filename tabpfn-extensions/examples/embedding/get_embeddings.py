"""TabPFN Embedding Example

This example demonstrates how to extract embeddings from TabPFN models and use them
for classification and regression tasks.

NOTE: This example requires the full TabPFN implementation (pip install tabpfn).
It will not work with the TabPFN client (pip install tabpfn-client) because
the embedding functionality is not available in the client version.
"""

from sklearn.datasets import load_breast_cancer, load_diabetes
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

# Note: You need to install the full TabPFN package for this example
# pip install tabpfn
from tabpfn_extensions import TabPFNClassifier, TabPFNRegressor
from tabpfn_extensions.embedding import TabPFNEmbedding

# Load and evaluate classification dataset
print("Loading classification dataset (kc1)...")
df = load_breast_cancer(return_X_y=False)
X, y = df["data"], df["target"]
attribute_names = df["feature_names"]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.5,
    random_state=42,
)


# Train and evaluate vanilla logistic regression
model = LogisticRegression()
model.fit(X_train, y_train)
print(
    f"Baseline Logistic Regression Accuracy: {accuracy_score(y_test, model.predict(X_test)):.4f}",
)

# Train and evaluate TabPFN embeddings (vanilla)
clf = TabPFNClassifier(n_estimators=1, random_state=42)
embedding_extractor = TabPFNEmbedding(tabpfn_clf=clf, n_fold=0)
train_embeddings = embedding_extractor.get_embeddings(
    X_train,
    y_train,
    X_test,
    data_source="train",
)
test_embeddings = embedding_extractor.get_embeddings(
    X_train,
    y_train,
    X_test,
    data_source="test",
)

model = LogisticRegression()
model.fit(train_embeddings[0], y_train)
y_pred = model.predict(test_embeddings[0])
print(
    f"Logistic Regression with TabPFN (Vanilla) Accuracy: {accuracy_score(y_test, y_pred):.4f}",
)


# Train and evaluate TabPFN embeddings (K-fold cross-validation)
clf = TabPFNClassifier(n_estimators=1, random_state=42)
embedding_extractor = TabPFNEmbedding(tabpfn_clf=clf, n_fold=10)
train_embeddings = embedding_extractor.get_embeddings(
    X_train,
    y_train,
    X_test,
    data_source="train",
)
test_embeddings = embedding_extractor.get_embeddings(
    X_train,
    y_train,
    X_test,
    data_source="test",
)

model = LogisticRegression()
model.fit(train_embeddings[0], y_train)
y_pred = model.predict(test_embeddings[0])
print(
    f"Logistic Regression with TabPFN (K-Fold CV) Accuracy: {accuracy_score(y_test, y_pred):.4f}",
)

# Load and evaluate regression dataset
X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.33,
    random_state=42,
)

# Train and evaluate vanilla linear regression
model = LinearRegression()
model.fit(X_train, y_train)
print(
    f"Baseline Linear Regression R2 Score: {r2_score(y_test, model.predict(X_test)):.4f}",
)

# Train and evaluate TabPFN embeddings (vanilla)
reg = TabPFNRegressor(n_estimators=1, random_state=42)
embedding_extractor = TabPFNEmbedding(tabpfn_reg=reg, n_fold=0)
train_embeddings = embedding_extractor.get_embeddings(
    X_train,
    y_train,
    X_test,
    data_source="train",
)
test_embeddings = embedding_extractor.get_embeddings(
    X_train,
    y_train,
    X_test,
    data_source="test",
)

model = LinearRegression()
model.fit(train_embeddings[0], y_train)
y_pred = model.predict(test_embeddings[0])
print(
    f"Linear Regression with TabPFN (Vanilla) R2 Score: {r2_score(y_test, y_pred):.4f}",
)


# Train and evaluate TabPFN embeddings (K-fold cross-validation)
reg = TabPFNRegressor(n_estimators=1, random_state=42)
embedding_extractor = TabPFNEmbedding(tabpfn_reg=reg, n_fold=10)
train_embeddings = embedding_extractor.get_embeddings(
    X_train,
    y_train,
    X_test,
    data_source="train",
)
test_embeddings = embedding_extractor.get_embeddings(
    X_train,
    y_train,
    X_test,
    data_source="test",
)

model = LinearRegression()
model.fit(train_embeddings[0], y_train)
y_pred = model.predict(test_embeddings[0])

print(
    f"Linear Regression with TabPFN (K-Fold CV) R2 Score: {r2_score(y_test, y_pred):.4f}",
)
