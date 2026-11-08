import streamlit as st
import pandas as pd
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="AIML Classifier Lab", page_icon="🧪", layout="wide")
st.title("🧪 Breast Cancer Classification Model Explorer")
st.caption("BITS Pilani M.Tech AIML — Assignment 2 | Upload the held-out test CSV to evaluate saved models.")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
TARGET = "diagnosis"

@st.cache_resource
def load_model(path):
    return joblib.load(path)

uploaded = st.file_uploader("Upload test data (CSV; must include all 30 input features and the 'diagnosis' target column)", type=["csv"])
selected = st.selectbox("Choose a classifier", list(MODEL_FILES.keys()))

if uploaded is None:
    st.info("Upload test_data.csv from the GitHub repository to view evaluation results. The app does not train or retrain models.")
    st.markdown("**Target classes:** malignant and benign.")
    st.stop()

try:
    data = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read the CSV: {exc}")
    st.stop()

model_path = os.path.join("model", MODEL_FILES[selected])
if not os.path.exists(model_path):
    st.error(f"Saved model not found: {model_path}")
    st.stop()

model = load_model(model_path)
expected = list(getattr(model, "feature_names_in_", []))
if not expected:
    # Pipelines retain feature_names_in_ when fitted with a DataFrame.
    expected = list(data.columns.drop(TARGET)) if TARGET in data.columns else []
missing = [c for c in expected if c not in data.columns]
if TARGET not in data.columns:
    st.error(f"Target column '{TARGET}' is missing.")
    st.stop()
if missing:
    st.error("Missing required feature columns: " + ", ".join(missing))
    st.stop()

X_eval = data[expected]
y_true = data[TARGET].astype(str)
if not set(y_true.unique()).issubset({"malignant", "benign"}):
    st.error("Target values must be 'malignant' or 'benign'.")
    st.stop()

try:
    y_pred = model.predict(X_eval)
    probabilities = model.predict_proba(X_eval)
    class_order = list(model.classes_)
    benign_probability = probabilities[:, class_order.index("benign")]
    y_binary = (y_true == "benign").astype(int)
    metrics = {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_binary, benign_probability),
        "Precision": precision_score(y_true, y_pred, pos_label="benign", zero_division=0),
        "Recall": recall_score(y_true, y_pred, pos_label="benign", zero_division=0),
        "F1 Score": f1_score(y_true, y_pred, pos_label="benign", zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }
except Exception as exc:
    st.error(f"Evaluation failed. Check that the CSV contains numeric feature values. Details: {exc}")
    st.stop()

st.subheader(f"Results — {selected}")
cols = st.columns(6)
for col, (label, value) in zip(cols, metrics.items()):
    col.metric(label, f"{value:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("#### Confusion matrix")
    cm = confusion_matrix(y_true, y_pred, labels=["malignant", "benign"])
    fig, ax = plt.subplots()
    ax.imshow(cm)
    ax.set_xticks([0, 1], labels=["malignant", "benign"])
    ax.set_yticks([0, 1], labels=["malignant", "benign"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    for (i, j), val in __import__("numpy").ndenumerate(cm):
        ax.text(j, i, str(val), ha="center", va="center")
    st.pyplot(fig)
    plt.close(fig)
with right:
    st.markdown("#### Classification report")
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(report).transpose().round(4), use_container_width=True)

with st.expander("Preview uploaded test data"):
    st.dataframe(data.head(20), use_container_width=True)
st.caption("Evaluation is performed only on the uploaded test rows. The saved model is loaded; no training occurs in this app.")
