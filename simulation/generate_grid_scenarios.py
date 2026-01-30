import pandas as pd
import numpy as np

df = pd.read_csv("simulation/australian_gp_2026_predictions.csv")
features = pd.read_csv("data_processed/race_features.csv")

df = df.merge(
    features[["driver_number", "avg_quali_position"]],
    on="driver_number",
    how="left"
)

def generate_grid(df, noise_std):
    noise = np.random.normal(0, noise_std, size=len(df))
    grid_score = df["avg_quali_position"] + noise
    order = np.argsort(grid_score)
    grid = pd.Series(index=df.index, dtype=int)
    for pos, idx in enumerate(order, start=1):
        grid[idx] = pos
    return grid

df["grid_baseline"] = generate_grid(df, noise_std=0.3)
df["grid_good"] = generate_grid(df, noise_std=0.1)
df["grid_bad"] = generate_grid(df, noise_std=0.6)

df.to_csv("simulation/grid_scenarios_2026.csv", index=False)

print("Grid scenarios generated")
