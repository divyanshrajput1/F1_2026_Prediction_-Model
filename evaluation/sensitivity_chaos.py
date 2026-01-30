import pandas as pd
import numpy as np

CHAOS_LEVELS = [0.0, 0.1, 0.2, 0.3]   # +0% → +30% DNF risk

df = pd.read_csv("simulation/australian_gp_2026_predictions.csv")

rows = []

for chaos in CHAOS_LEVELS:
    p_dnf_adj = np.clip(df.p_dnf * (1 + chaos), 0, 1)

    exp_finishers = len(df) * (1 - p_dnf_adj.mean())
    avg_podium = (1 - p_dnf_adj).mean() * df.p_podium.mean()

    rows.append({
        "chaos_increase": chaos,
        "expected_finishers": exp_finishers,
        "avg_podium_probability": avg_podium
    })
    
    df["robust_score"] = (
    (1 - df.p_dnf) *
    df.p_top10
)

df = df.sort_values("robust_score", ascending=False)

print(df[["driver_number", "robust_score"]].head(10))


out = pd.DataFrame(rows)
print(out)
