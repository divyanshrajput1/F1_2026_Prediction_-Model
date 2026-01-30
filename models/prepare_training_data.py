import pandas as pd

# ---------------------------
# Load data
# ---------------------------
features = pd.read_csv("data_processed/race_features.csv")
results = pd.read_csv("data_raw/results.csv")

# Use only past seasons for training
train_results = results[results["year"] <= 2025]

# Merge labels
df = train_results.merge(
    features,
    on="driver_number",
    how="inner"
)

# ---------------------------
# Create labels
# ---------------------------
df["dnf_label"] = df["dnf"]
df["top10_label"] = ((df["finish_position"] <= 10) & (df["dnf"] == 0)).astype(int)
df["podium_label"] = ((df["finish_position"] <= 3) & (df["dnf"] == 0)).astype(int)

# ---------------------------
# Drop leakage columns
# ---------------------------
DROP_COLS = [
    "finish_position",
    "points",
    "status",
    "dnf",
    "year",
    "round"
]

df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])

# ---------------------------
# KEEP ONLY NUMERIC FEATURES
# (this removes constructor names automatically)
# ---------------------------
X = df.select_dtypes(include=["number", "bool"])

# Targets
y_dnf = df["dnf_label"]
y_top10 = df["top10_label"]
y_podium = df["podium_label"]

# Remove labels from X
X = X.drop(columns=["dnf_label", "top10_label", "podium_label"])

# ---------------------------
# Save training matrices
# ---------------------------
X.to_csv("data_processed/X_train.csv", index=False)
y_dnf.to_csv("data_processed/y_dnf.csv", index=False)
y_top10.to_csv("data_processed/y_top10.csv", index=False)
y_podium.to_csv("data_processed/y_podium.csv", index=False)

print("Training datasets prepared")
print(f"X shape: {X.shape}")
print("Remaining columns:", list(X.columns))
