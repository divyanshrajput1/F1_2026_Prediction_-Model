import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from joblib import dump

X = pd.read_csv("data_processed/X_train.csv")
y_podium = pd.read_csv("data_processed/y_podium.csv").values.ravel()
y_top10 = pd.read_csv("data_processed/y_top10.csv").values.ravel()
y_dnf = pd.read_csv("data_processed/y_dnf.csv").values.ravel()

mask = (y_dnf == 0) & (y_top10 == 1)
X = X[mask]
y_podium = y_podium[mask]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y_podium,
    test_size=0.25,
    random_state=42,
    stratify=y_podium
)

pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

model = XGBClassifier(
    n_estimators=600,
    max_depth=4,
    learning_rate=0.04,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.15,
    reg_alpha=0.4,
    reg_lambda=1.5,
    scale_pos_weight=pos_weight,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42
)

model.fit(X_train, y_train)

val_probs = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, val_probs)

print(f"Podium model trained")
print(f"Podium ROC-AUC: {auc:.3f}")

dump(model, "models/podium_model.pkl")
print("podium_model.pkl saved")
