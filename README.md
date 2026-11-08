# AIML Assignment 2 — Classification Model Comparison

## a. Problem statement
Build and compare five supervised classification models on one public binary-classification dataset. Evaluate the models using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC), then provide a Streamlit interface that evaluates saved models on uploaded test data without retraining.

## b. Dataset description
**Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC), available through the UCI Machine Learning Repository and distributed in scikit-learn's public dataset collection.

- Instances: 569
- Input features: 30 numeric measurements computed from digitised images of fine needle aspirates of breast masses.
- Target: `diagnosis` with two labels: `malignant` and `benign`.
- Classification type: Binary classification.
- Train/test split: 80% training and 20% held-out testing, stratified by target, with `random_state=42`.
- Test file: `test_data.csv` (contains the held-out rows and target column).
- Positive class for Precision, Recall and F1: `benign`.
- AUC: calculated using the predicted probability of the `benign` class.

The dataset has more than the required 12 features and 500 instances.

## c. Github Repository Link
**Repository URL:** [ADD YOUR PUBLIC GITHUB REPOSITORY URL HERE]

## d. Models used

The following models were trained on the same training split. StandardScaler is included in the Logistic Regression, KNN and SVM pipelines. The Decision Tree and Gaussian Naive Bayes models use the original numeric features. All metrics below are calculated on the same held-out test split.

### Evaluation metrics

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| Logistic Regression | 0.9649 | 0.9960 | 0.9595 | 0.9861 | 0.9726 | 0.9245 |
| Decision Tree | 0.9123 | 0.8770 | 0.8974 | 0.9722 | 0.9333 | 0.8113 |
| KNN | 0.9561 | 0.9823 | 0.9467 | 0.9861 | 0.9660 | 0.9058 |
| Gaussian Naive Bayes | 0.9386 | 0.9934 | 0.9114 | 1.0000 | 0.9536 | 0.8715 |
| Support Vector Machine | 0.9737 | 0.9947 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | A linear decision boundary provides a strong, interpretable baseline. Standardisation is applied before fitting because the feature scales differ. |
| Decision Tree | The tree captures non-linear feature interactions and is easy to interpret. Its depth is limited to reduce overfitting; performance can vary with small changes in the data. |
| KNN | KNN classifies a sample using nearby training observations. Standardisation is essential because distance calculations are scale-sensitive; prediction can be slower as data grows. |
| Gaussian Naive Bayes | The model is computationally efficient and provides probabilistic predictions. Its conditional-independence and Gaussian-feature assumptions may limit performance when features are correlated. |
| Support Vector Machine | The RBF-kernel SVM can model non-linear boundaries. Standardisation and probability estimation are used; it can perform well but is less interpretable and may be more computationally expensive. |
| **Overall winner** | **Support Vector Machine** achieved the strongest F1 score in this experiment. The final choice should also consider AUC, MCC, the cost of false positives/false negatives and interpretability. |

## Streamlit application
The app provides:
1. CSV upload for held-out test data.
2. A dropdown to select one of the five saved classifiers.
3. Accuracy, AUC, Precision, Recall, F1 and MCC metrics.
4. Confusion matrix and classification report.
5. Validation for the target and required feature columns.

The app loads `.joblib` models from `model/` and evaluates uploaded rows only. It does not retrain models.

**Live Streamlit URL:** [ADD YOUR DEPLOYED STREAMLIT COMMUNITY CLOUD URL HERE]

## Repository structure
```text
aiml_assignment2/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── full_dataset.csv
├── model_comparison.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── gaussian_naive_bayes.joblib
    └── support_vector_machine.joblib
```

## Run locally
```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
streamlit run app.py
```
Upload `test_data.csv` when the app opens.

## Reproducibility and notes
- Split seed: `42`.
- Models are trained once by the training workflow and saved under `model/`.
- The test CSV is separate from training data.
- Do not upload a screenshot placeholder as proof. Run the project on BITS Virtual Lab and capture your own execution screenshot.
- Replace both link placeholders after publishing the repository and deploying the app.
- Review the code, run it yourself, and be prepared to explain the preprocessing, model assumptions and metric interpretation.
