import pandas as pd
import numpy as np

TARGET_DRIVER = 44    # change driver number here
GRID_SHIFTS = [-5, -3, -1, 0, +1, +3, +5]

preds = pd.read_csv("simulation/australian_gp_2026_predictions.csv")
grid_df = pd.read_csv("simulation/grid_scenarios_2026.csv")

df = preds.merge(
    grid_df[["driver_number", "grid_baseline"]],
    on="driver_number",
    how="left"
)

base_grid = df.loc[df.driver_number == TARGET_DRIVER, "grid_baseline"].values[0]

results = []

for shift in GRID_SHIFTS:
    new_grid = max(1, base_grid + shift)

    grid_effect = -0.03 * (new_grid - 1)

    row = df[df.driver_number == TARGET_DRIVER].iloc[0]

    performance_score = (
        0.6 * row.p_top10 +
        1.0 * row.p_podium -
        0.5 * row.p_dnf +
        grid_effect
    )

    results.append({
        "grid_position": new_grid,
        "performance_score": performance_score
    })

out = pd.DataFrame(results)
print(out)
