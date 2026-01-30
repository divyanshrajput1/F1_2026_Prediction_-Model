import pandas as pd
import numpy as np


def compute_constructor_features(results, target_year, target_round):
    features = []

    for constructor in results["constructor"].unique():

        hist = results[
            (results["constructor"] == constructor) &
            (
                (results["year"] < target_year) |
                ((results["year"] == target_year) & (results["round"] < target_round))
            )
        ].sort_values(["year", "round"])

        # Skip constructors with insufficient history
        if len(hist) < 10:
            continue

        constructor_dnf_rate = hist["dnf"].mean()
        avg_finish = hist["finish_position"].mean()
        avg_points = hist["points"].mean()

        # ---------------------------
        # Recent reliability (last 5 races)
        # ---------------------------
        recent_5 = hist.tail(5)
        recent_constructor_dnf_rate = recent_5["dnf"].mean()
        recent_finish = recent_5["finish_position"].mean()

        # ---------------------------
        # Reliability trend (VERY important)
        # ---------------------------
        dnf_shift = hist["dnf"].diff()
        constructor_dnf_trend = dnf_shift.mean()

        # ---------------------------
        # Volatility (mechanical instability proxy)
        # ---------------------------
        finish_std = hist["finish_position"].std()

        features.append({
            "constructor": constructor,
            "constructor_dnf_rate": constructor_dnf_rate,
            "recent_constructor_dnf_rate": recent_constructor_dnf_rate,
            "constructor_dnf_trend": constructor_dnf_trend,
            "avg_constructor_finish": avg_finish,
            "recent_avg_finish": recent_finish,
            "constructor_finish_std": finish_std,
            "avg_points": avg_points
        })

    return pd.DataFrame(features)
