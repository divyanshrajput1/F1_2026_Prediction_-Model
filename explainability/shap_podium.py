import pandas as pd
import shap
from joblib import load
import matplotlib.pyplot as plt

model = load("models/podium_model.pkl")

X = pd.read_csv("data_processed/X_train.csv")
y_dnf = pd.read_csv("data_processed/y_dnf.csv").values.ravel()
y_top10 = pd.read_csv("data_processed/y_top10.csv").values.ravel()

mask = (y_dnf == 0) & (y_top10 == 1)
X = X[mask]

X_sample = X.sample(400, random_state=42)

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
