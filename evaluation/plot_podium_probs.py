import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("simulation/australian_gp_2026_simulation_results.csv")

df = df.sort_values("podium_prob", ascending=False).head(10)

plt.figure()
plt.bar(df["driver_number"].astype(str), df["podium_prob"])
plt.xlabel("Driver")
plt.ylabel("Podium Probability")
plt.title("Australian GP 2026 — Podium Probabilities")

plt.show()
