"""Feature engineering for the Smart Airbnb project.

Adds the engineered features from the notebook: review velocity,
availability rate, minimum-nights category, host activity score,
price category, and distance to Manhattan. Also defines the model
feature lists and the demand proxy label.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from paths import PROCESSED_DIR

# Shared with dashboard/app.py and the trainers.
FEATURE_COLS = [
    "neighbourhood_group_cleansed", "room_type", "accommodates",
    "bedrooms", "bathrooms", "beds", "minimum_nights",
    "number_of_reviews", "review_scores_rating", "availability_365",
    "host_listings_count", "host_is_superhost",
    "review_velocity", "host_activity_score",
    "distance_to_manhattan",
]

CLF_FEATURE_COLS = [
    "neighbourhood_group_cleansed", "room_type", "accommodates",
    "bedrooms", "bathrooms", "beds", "minimum_nights",
    "price", "availability_365", "host_listings_count",
    "host_is_superhost", "distance_to_manhattan",
]

CATEGORICAL_COLS = [
    "neighbourhood_group_cleansed", "room_type", "host_is_superhost",
]

MANHATTAN_LAT, MANHATTAN_LON = 40.7831, -73.9712


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features. Expects the output of preprocessing.clean_listings."""
    df = df.copy()

    df["neighbourhood_group"] = df["neighbourhood_group_cleansed"]
    df["review_velocity"] = df["reviews_per_month"]
    df["availability_rate"] = df["availability_365"] / 365

    df["minimum_nights_category"] = pd.cut(
        df["minimum_nights"],
        bins=[0, 3, 7, 30, float("inf")],
        labels=["short", "weekly", "monthly", "long_term"],
    )

    df["host_activity_score"] = (
        df["host_listings_count"] * 0.5 + df["number_of_reviews"] * 0.5
    )

    # Analysis-only segmentation. Do not use as a model input: it is derived
    # from price itself, so feeding it back would be target leakage.
    df["price_category"] = pd.cut(
        df["price"],
        bins=[0, 100, 250, 500, float("inf")],
        labels=["budget", "mid", "premium", "luxury"],
    )

    df["distance_to_manhattan"] = np.sqrt(
        (df["latitude"] - MANHATTAN_LAT) ** 2
        + (df["longitude"] - MANHATTAN_LON) ** 2
    )

    return df


def add_demand_label(df: pd.DataFrame) -> pd.DataFrame:
    """Demand proxy: a listing counts as High demand if it had any estimated
    booked nights in the last 365 days (occupancy > 0).

    Caveat: the occupancy estimate itself is derived from review activity, so
    the label is a proxy, not ground truth. Keep every occupancy-derived
    column out of the classifier's inputs.
    """
    df = df.copy()
    df["demand_label"] = (df["estimated_occupancy_l365d"] > 0).astype(int)
    return df


def cap_outliers(df: pd.DataFrame, quantile: float = 0.99) -> tuple[pd.DataFrame, float]:
    """Trim the extreme luxury tail above the given price quantile."""
    cap = df["price"].quantile(quantile)
    return df[df["price"] <= cap].copy(), float(cap)


def main() -> None:
    df = pd.read_csv(PROCESSED_DIR / "listings_clean.csv")
    df = engineer_features(df)
    df.to_csv(PROCESSED_DIR / "listings_features.csv", index=False)
    print(f"Saved listings_features.csv: {df.shape}")


if __name__ == "__main__":
    main()
