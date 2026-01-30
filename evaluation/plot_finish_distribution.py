import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("simulation/australian_gp_2026_simulation_results.csv")

df = df.sort_values("avg_finish_position").head(8)

plt.figure()
plt.bar(df["driver_number"].astype(str), df["avg_finish_position"])
plt.gca().invert_yaxis()
plt.xlabel("Driver")
plt.ylabel("Average Finish Position")
plt.title("Australian GP 2026 — Average Finish Position")

plt.show()
