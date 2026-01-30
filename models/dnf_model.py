import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from joblib import dump

# ---------------------------
# Load training data
# ---------------------------
X = pd.read_csv("data_processed/X_train.csv")
y = pd.read_csv("data_processed/y_dnf.csv").values.ravel()

# ---------------------------
# Train / validation split
# ---------------------------
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# ---------------------------
# Handle class imbalance
# ---------------------------
pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

# ---------------------------
# TUNED DNF MODEL
# ---------------------------
model = XGBClassifier(
    n_estimators=600,
    max_depth=3,                 # shallower trees
    learning_rate=0.03,          # slower learning
    subsample=0.7,
    colsample_bytree=0.7,
    min_child_weight=5,          # reduce noise
    gamma=0.2,                   # require stronger splits
    reg_alpha=0.5,               # L1 regularization
    reg_lambda=1.5,              # L2 regularization
    scale_pos_weight=pos_weight, # imbalance fix
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42
)

# ---------------------------
# Train
# ---------------------------
model.fit(X_train, y_train)

# ---------------------------
# Evaluate
# ---------------------------
val_probs = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, val_probs)

print(f"DNF model trained (tuned)")
print(f"DNF ROC-AUC: {auc:.3f}")

# ---------------------------
# Save
# ---------------------------
dump(model, "models/dnf_model.pkl")
print("dnf_model.pkl saved")
