"""Train five classifiers and save model artifacts plus a held-out test CSV."""
import os
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from joblib import dump

os.makedirs("model", exist_ok=True)
dataset = load_breast_cancer(as_frame=True)
data = dataset.frame.rename(columns={"target": "diagnosis"})
data["diagnosis"] = data["diagnosis"].map({0: "malignant", 1: "benign"})
X = data.drop(columns=["diagnosis"])
y = data["diagnosis"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
test = X_test.copy()
test["diagnosis"] = y_test
test.to_csv("test_data.csv", index=False)
data.to_csv("full_dataset.csv", index=False)

models = {
    "logistic_regression": Pipeline([("scaler", StandardScaler()), ("classifier", LogisticRegression(max_iter=5000, random_state=42))]),
    "decision_tree": DecisionTreeClassifier(random_state=42, max_depth=4),
    "knn": Pipeline([("scaler", StandardScaler()), ("classifier", KNeighborsClassifier(n_neighbors=5))]),
    "gaussian_naive_bayes": GaussianNB(),
    "support_vector_machine": Pipeline([("scaler", StandardScaler()), ("classifier", SVC(kernel="rbf", probability=True, random_state=42))]),
}
rows = []
for filename, estimator in models.items():
    estimator.fit(X_train, y_train)
    dump(estimator, f"model/{filename}.joblib")
    prediction = estimator.predict(X_test)
    probability = estimator.predict_proba(X_test)[:, list(estimator.classes_).index("benign")]
    rows.append({
        "Model": filename,
        "Accuracy": accuracy_score(y_test, prediction),
        "AUC": roc_auc_score((y_test == "benign").astype(int), probability),
        "Precision": precision_score(y_test, prediction, pos_label="benign", zero_division=0),
        "Recall": recall_score(y_test, prediction, pos_label="benign", zero_division=0),
        "F1": f1_score(y_test, prediction, pos_label="benign", zero_division=0),
        "MCC": matthews_corrcoef(y_test, prediction),
    })
pd.DataFrame(rows).to_csv("model_comparison.csv", index=False)
print(pd.DataFrame(rows).to_string(index=False))
