import pandas as pd

df = pd.read_csv("evaluation/australian_gp_2026_race_preview.csv")

podium = df.sort_values("p_podium", ascending=False).head(6)

print("\nAustralian GP 2026 — Podium Contenders\n")
print(podium[[
    "driver_number",
    "grid_baseline",
    "p_podium",
    "p_top10",
    "dnf_prob"
]])
