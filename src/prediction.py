"""Single-listing inference (used by the Streamlit dashboard).

Loads the saved models once and turns raw user inputs into encoded
feature rows for both the price and demand models.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

import features as F
from paths import MODELS_DIR

# Borough centroids (lat, lon) used to derive a sensible
# distance_to_manhattan value from the selected neighbourhood group.
BOROUGH_CENTROIDS = {
    "Manhattan": (40.7831, -73.9712),
    "Brooklyn": (40.6782, -73.9442),
    "Queens": (40.7282, -73.7949),
    "Bronx": (40.8448, -73.8648),
    "Staten Island": (40.5795, -74.1502),
}


MODEL_FILES = {
    "price_model": (MODELS_DIR / "price_model.pkl", MODELS_DIR / "price_model_xgboost.pkl"),
    "demand_model": (MODELS_DIR / "demand_model.pkl", MODELS_DIR / "demand_model_xgboost.pkl"),
}


def _load_any(key: str):
    """Load the pipeline artifact, falling back to the legacy notebook pickle."""
    for path in MODEL_FILES[key]:
        try:
            return joblib.load(path)
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        f"No saved model found for '{key}'. Run: python src/run_pipeline.py"
    )


def load_models():
    """Load price model, demand model, encoders, and feature config."""
    config = {
        "feature_cols": F.FEATURE_COLS,
        "clf_feature_cols": F.CLF_FEATURE_COLS,
        "categorical_cols": F.CATEGORICAL_COLS,
    }
    try:
        config = joblib.load(MODELS_DIR / "feature_config.pkl")
    except FileNotFoundError:
        pass
    return (
        _load_any("price_model"),
        _load_any("demand_model"),
        joblib.load(MODELS_DIR / "label_encoders.pkl"),
        config,
    )


def build_input_row(details: dict, label_encoders: dict) -> pd.DataFrame:
    """Create an encoded single-row frame covering both feature sets."""
    borough = details["neighbourhood_group_cleansed"]
    lat, lon = BOROUGH_CENTROIDS.get(borough, (F.MANHATTAN_LAT, F.MANHATTAN_LON))
    distance = np.sqrt((lat - F.MANHATTAN_LAT) ** 2 + (lon - F.MANHATTAN_LON) ** 2)

    row = {
        "accommodates": details["accommodates"],
        "bedrooms": details["bedrooms"],
        "bathrooms": details["bathrooms"],
        "beds": details["beds"],
        "minimum_nights": details["minimum_nights"],
        "number_of_reviews": details["number_of_reviews"],
        "review_scores_rating": details["review_scores_rating"],
        "availability_365": details["availability_365"],
        "host_listings_count": details["host_listings_count"],
        "host_is_superhost": details["host_is_superhost"],
        "review_velocity": details["reviews_per_month"],
        "host_activity_score": (
            details["host_listings_count"] * 0.5 + details["number_of_reviews"] * 0.5
        ),
        "distance_to_manhattan": distance,
    }

    data = {"neighbourhood_group_cleansed": [borough], "room_type": [details["room_type"]]}
    data.update({k: [v] for k, v in row.items()})
    input_df = pd.DataFrame(data)

    for col in F.CATEGORICAL_COLS:
        le = label_encoders[col]
        if details_value := input_df.at[0, col]:
            if details_value not in le.classes_:
                raise ValueError(
                    f"Unknown value '{details_value}' for {col}. "
                    f"Valid options: {list(le.classes_)}"
                )
        input_df[col] = le.transform(input_df[col])
    return input_df


def predict(price_model, demand_model, input_df: pd.DataFrame) -> dict:
    """Run both models on one encoded row; returns price in dollars and demand."""
    predicted_log_price = price_model.predict(input_df[F.FEATURE_COLS])[0]
    predicted_price = float(np.expm1(predicted_log_price))

    demand_input = input_df[[c for c in F.CLF_FEATURE_COLS if c != "price"]].copy()
    demand_input["price"] = predicted_price
    demand_input = demand_input[F.CLF_FEATURE_COLS]
    demand_prob = float(demand_model.predict_proba(demand_input)[0, 1])

    return {
        "price": predicted_price,
        "demand": "High" if demand_prob >= 0.5 else "Low",
        "demand_probability": demand_prob,
    }
