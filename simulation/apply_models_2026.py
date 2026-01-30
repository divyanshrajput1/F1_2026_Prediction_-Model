import pandas as pd
from joblib import load

df = pd.read_csv("data_processed/race_features.csv")

output = df[["driver_number"]].copy()

X_train = pd.read_csv("data_processed/X_train.csv")
feature_cols = X_train.columns.tolist()

X_pred = df.copy()

X_pred = X_pred.select_dtypes(include=["number", "bool"])

X_pred = X_pred.reindex(columns=feature_cols, fill_value=0)

dnf_model = load("models/dnf_model.pkl")
top10_model = load("models/top10_model.pkl")
podium_model = load("models/podium_model.pkl")

# Predict probabilities

output["p_dnf"] = dnf_model.predict_proba(X_pred)[:, 1]
output["p_top10_given_finish"] = top10_model.predict_proba(X_pred)[:, 1]
output["p_podium_given_top10"] = podium_model.predict_proba(X_pred)[:, 1]

# Chain probabilities

output["p_top10"] = (1 - output["p_dnf"]) * output["p_top10_given_finish"]
output["p_podium"] = (
    (1 - output["p_dnf"])
    * output["p_top10_given_finish"]
    * output["p_podium_given_top10"]
)

output = output.sort_values(
    ["p_podium", "p_top10"],
    ascending=False
)

output.to_csv(
    "simulation/australian_gp_2026_predictions.csv",
    index=False
)

print("Australian GP 2026 predictions generated")
print(output)
