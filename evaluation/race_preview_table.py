import pandas as pd

sim = pd.read_csv("simulation/australian_gp_2026_simulation_results.csv")
pred = pd.read_csv("simulation/australian_gp_2026_predictions.csv")
grid = pd.read_csv("simulation/grid_scenarios_2026.csv")

df = sim.merge(pred, on="driver_number")
df = df.merge(grid[["driver_number", "grid_baseline"]], on="driver_number")

df["best_case"] = df["avg_finish_position"] - 2
df["worst_case"] = df["avg_finish_position"] + 4

df = df[[
    "driver_number",
    "grid_baseline",
    "avg_finish_position",
    "best_case",
    "worst_case",
    "p_podium",
    "p_top10",
    "dnf_prob",
    "expected_points"
]]

df = df.sort_values("expected_points", ascending=False)

df.to_csv(
    "evaluation/australian_gp_2026_race_preview.csv",
    index=False
)

print("Race preview table generated")
print(df.head(10))
