import pandas as pd
import numpy as np


def compute_driver_features(
    results,
    qualifying,
    lap_times,
    drivers,
    target_year,
    target_round
):
    features = []

    for driver in results["driver_number"].unique():
        past_results = results[
            (results["driver_number"] == driver) &
            (
                (results["year"] < target_year) |
                ((results["year"] == target_year) & (results["round"] < target_round))
            )
        ].sort_values(["year", "round"], ascending=False)

        if len(past_results) < 5:
            continue

        last_5 = past_results.head(5)
        last_10 = past_results.head(10)
        recent_3 = past_results.head(3)

        dnf_rate = last_10["dnf"].mean()
        recent_dnf_rate = recent_3["dnf"].mean()

        avg_finish_5 = last_5["finish_position"].mean()
        avg_finish_10 = last_10["finish_position"].mean()

        # Qualifying performance
        quali_hist = qualifying[
            qualifying["driver_number"] == driver
        ]
        avg_quali = quali_hist["quali_position"].mean()

        # ---------------------------
        # Lap-time based features
        # ---------------------------
        pace_hist = lap_times[
            lap_times["driver_number"] == driver
        ]

        # Average race pace (from aggregated lap times)
        avg_pace = pace_hist["avg_lap_time_sec"].mean()

        # Consistency (DEFENSIVE: column-safe)
        if "stint_consistency" in pace_hist.columns:
            pace_std = pace_hist["stint_consistency"].mean()
        else:
            pace_std = np.nan

        # Experience
        first_year = drivers[
            drivers["driver_number"] == driver
        ]["first_seen_year"].min()

        experience = target_year - first_year if pd.notna(first_year) else 0

        features.append({
            "driver_number": driver,
            "driver_form_5": avg_finish_5,
            "driver_form_10": avg_finish_10,
            "dnf_rate": dnf_rate,
            "recent_dnf_rate": recent_dnf_rate,
            "avg_quali_position": avg_quali,
            "avg_race_pace": avg_pace,
            "pace_consistency": pace_std,  # standardized name
            "experience_years": experience
        })

    return pd.DataFrame(features)
