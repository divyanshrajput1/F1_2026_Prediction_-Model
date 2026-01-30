import numpy as np
import pandas as pd
from configs.load_race_config import load_race_config

cfg = load_race_config()

N_SIM        = cfg["simulation"]["n_simulations"]
NOISE_STD    = cfg["chaos"]["noise_std"]
DNF_MULT     = cfg["chaos"]["dnf_multiplier"]
GRID_WEIGHT  = cfg["qualifying"]["grid_effect_weight"]
GRID_SOURCE  = cfg["qualifying"]["grid_source"]
POINTS_TABLE = cfg["scoring"]["points_table"]

np.random.seed(cfg["simulation"]["random_seed"])

df = pd.read_csv("simulation/australian_gp_2026_predictions.csv")
grid_df = pd.read_csv("simulation/grid_scenarios_2026.csv")

df = df.merge(grid_df, on="driver_number", how="left")

def find_probability_column(df, keywords, logical_name):
    candidates = [
        c for c in df.columns
        if any(k in c.lower() for k in keywords)
        and df[c].dtype != object
    ]

    if not candidates:
        raise KeyError(
            f"{logical_name} column not found.\n"
            f"Tried keywords: {keywords}\n"
            f"Available columns: {list(df.columns)}"
        )

    return df[candidates[0]].values, candidates[0]

p_dnf, dnf_col = find_probability_column(
    df,
    keywords=["dnf"],
    logical_name="DNF probability"
)

p_top10, top10_col = find_probability_column(
    df,
    keywords=["top10", "top_10"],
    logical_name="Top-10 probability"
)

p_podium, podium_col = find_probability_column(
    df,
    keywords=["podium"],
    logical_name="Podium probability"
)

print(f"✔ Using DNF column: {dnf_col}")
print(f"✔ Using Top-10 column: {top10_col}")
print(f"✔ Using Podium column: {podium_col}")

p_dnf = np.clip(p_dnf * DNF_MULT, 0, 1)

grid_map = {
    "baseline": "grid_baseline",
    "good": "grid_good",
    "bad": "grid_bad"
}

grid_col = grid_map[GRID_SOURCE]
grid = df[grid_col].values

drivers = df["driver_number"].values
n_drivers = len(drivers)

# Monte-Carlo Simulation
finish_positions = {d: [] for d in drivers}
dnf_counts = {d: 0 for d in drivers}
points_sum = {d: 0 for d in drivers}

for _ in range(N_SIM):

    dnfs = np.random.rand(n_drivers) < p_dnf
    noise = np.random.normal(0, NOISE_STD, size=n_drivers)
    grid_effect = GRID_WEIGHT * (grid - 1)

    performance = (
        0.6 * p_top10 +
        1.0 * p_podium -
        0.5 * p_dnf +
        grid_effect +
        noise
    )

    performance[dnfs] = -999
    order = np.argsort(performance)[::-1]

    pos = 1
    for idx in order:
        d = drivers[idx]
        if dnfs[idx]:
            dnf_counts[d] += 1
            finish_positions[d].append(None)
        else:
            finish_positions[d].append(pos)
            points_sum[d] += POINTS_TABLE.get(pos, 0)
            pos += 1

rows = []
for d in drivers:
    finishes = [f for f in finish_positions[d] if f is not None]
    rows.append({
        "driver_number": d,
        "avg_finish_position": np.mean(finishes) if finishes else np.nan,
        "podium_prob": sum(f <= 3 for f in finishes) / N_SIM,
        "top10_prob": sum(f <= 10 for f in finishes) / N_SIM,
        "dnf_prob": dnf_counts[d] / N_SIM,
        "expected_points": points_sum[d] / N_SIM
    })

results = pd.DataFrame(rows).sort_values("expected_points", ascending=False)

output_path = f"simulation/{cfg['outputs']['output_prefix']}_simulation_results.csv"
results.to_csv(output_path, index=False)

print("Monte-Carlo simulation completed successfully")
print(results.head(10))
