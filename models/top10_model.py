import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from joblib import dump

X = pd.read_csv("data_processed/X_train.csv")
y_top10 = pd.read_csv("data_processed/y_top10.csv").values.ravel()
y_dnf = pd.read_csv("data_processed/y_dnf.csv").values.ravel()

mask = y_dnf == 0
X = X[mask]
y_top10 = y_top10[mask]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y_top10,
    test_size=0.25,
    random_state=42,
    stratify=y_top10
)

model = XGBClassifier(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.3,
    reg_lambda=1.2,
    objective="binary:logistic",
    eval_metric="auc",
    random_state=42
)

model.fit(X_train, y_train)

val_probs = model.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, val_probs)

print(f"Top-10 model trained")
print(f"Top-10 ROC-AUC: {auc:.3f}")

dump(model, "models/top10_model.pkl")
print("top10_model.pkl saved")
