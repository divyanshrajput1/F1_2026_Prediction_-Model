import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("simulation/australian_gp_2026_simulation_results.csv")

df = df.sort_values("expected_points", ascending=False).head(10)

plt.figure()
plt.bar(df["driver_number"].astype(str), df["expected_points"])
plt.xlabel("Driver")
plt.ylabel("Expected Points")
plt.title("Australian GP 2026 — Expected Points")

plt.show()
