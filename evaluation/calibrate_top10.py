import pandas as pd
import matplotlib.pyplot as plt
from joblib import load
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

X = pd.read_csv("data_processed/X_train.csv")
y_dnf = pd.read_csv("data_processed/y_dnf.csv").values.ravel()
y_top10 = pd.read_csv("data_processed/y_top10.csv").values.ravel()

mask = y_dnf == 0
X = X[mask]
y = y_top10[mask]

model = load("models/top10_model.pkl")

p = model.predict_proba(X)[:, 1]

# Calibration curve
fraction_of_positives, mean_predicted_value = calibration_curve(
    y, p, n_bins=10, strategy="quantile"
)

plt.figure()
plt.plot(mean_predicted_value, fraction_of_positives, marker="o", label="Top-10 model")
plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
plt.xlabel("Predicted probability")
plt.ylabel("Observed frequency")
plt.title("Top-10 Probability Calibration")
plt.legend()
plt.show()

# Brier score
brier = brier_score_loss(y, p)
print(f"Top-10 Brier score: {brier:.4f}")
