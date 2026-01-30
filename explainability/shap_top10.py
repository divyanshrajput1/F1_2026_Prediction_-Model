import pandas as pd
import shap
from joblib import load
import matplotlib.pyplot as plt

model = load("models/top10_model.pkl")

X = pd.read_csv("data_processed/X_train.csv")
y_dnf = pd.read_csv("data_processed/y_dnf.csv").values.ravel()

X = X[y_dnf == 0]

X_sample = X.sample(500, random_state=42)

# SHAP explainer 
explainer = shap.Explainer(
    model.predict_proba,
    X_sample
)

shap_values = explainer(X_sample)

shap.summary_plot(
    shap_values[..., 1],
    X_sample,
    show=True
)
