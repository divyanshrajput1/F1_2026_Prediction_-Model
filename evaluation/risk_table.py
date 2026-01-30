import pandas as pd

df = pd.read_csv("evaluation/australian_gp_2026_race_preview.csv")

risk = df.sort_values("dnf_prob", ascending=False).head(8)

print("\n High-Risk Drivers (DNF)\n")
print(risk[[
    "driver_number",
    "grid_baseline",
    "dnf_prob",
    "expected_points"
]])
