import os
import pandas as pd

from features.driver import compute_driver_features
from features.constructor import compute_constructor_features
from features.track import melbourne_features


def build_race_features():

    results = pd.read_csv("data_raw/results.csv")
    qualifying = pd.read_csv("data_raw/qualifying.csv")
    lap_times = pd.read_csv("data_raw/lap_times.csv")
    drivers = pd.read_csv("data_raw/drivers.csv")

    TARGET_YEAR = 2026
    TARGET_ROUND = 1  # Australian GP

    driver_race_counts_2025 = (
        results[results["year"] == 2025]
        .groupby("driver_number")
        .size()
    )

    expected_drivers = driver_race_counts_2025[
        driver_race_counts_2025 >= 5
    ].index.values

    driver_df = compute_driver_features(
        results,
        qualifying,
        lap_times,
        drivers,
        TARGET_YEAR,
        TARGET_ROUND
    )

    driver_df = driver_df[
        driver_df["driver_number"].isin(expected_drivers)
    ]

    constructor_df = compute_constructor_features(
        results,
        TARGET_YEAR,
        TARGET_ROUND
    )

    latest_constructor = (
        results[
            (results["year"] < TARGET_YEAR) |
            ((results["year"] == TARGET_YEAR) & (results["round"] < TARGET_ROUND))
        ]
        .sort_values(["year", "round"])
        .groupby("driver_number")
        .tail(1)[["driver_number", "constructor"]]
    )

    df = driver_df.merge(
        latest_constructor,
        on="driver_number",
        how="left"
    )

    df = df.merge(
        constructor_df,
        on="constructor",
        how="left"
    )

    track_feats = melbourne_features()
    for k, v in track_feats.items():
        df[k] = v

    df["dnf_risk_index"] = (
        df["dnf_rate"] *
        df["constructor_dnf_rate"] *
        df["chaos_factor"]
    )

    os.makedirs("data_processed", exist_ok=True)
    df.to_csv("data_processed/race_features.csv", index=False)

    print("race_features.csv generated successfully")
    print(f"Rows: {df.shape[0]} | Unique drivers: {df['driver_number'].nunique()}")


if __name__ == "__main__":
    build_race_features()
