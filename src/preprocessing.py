"""Data preprocessing for the Smart Airbnb project.

Replicates the cleaning steps from notebooks/01_data_exploration.ipynb:
drop sparse/unusable columns, parse the price strings, impute review/room
counts, and drop rows with critical missing values.
"""
from __future__ import annotations

import pandas as pd

from paths import PROCESSED_DIR, RAW_DIR

RAW_LISTINGS = RAW_DIR / "listings.csv.gz"


def load_raw_listings(path=RAW_LISTINGS) -> pd.DataFrame:
    return pd.read_csv(path, compression="gzip")


def clean_listings(listings: pd.DataFrame) -> pd.DataFrame:
    """Apply the full cleaning pipeline and return a model-ready dataframe."""
    df = listings.copy()

    # 1. Drop sparse or unusable columns (100% missing, URLs, free text...)
    cols_to_drop = [
        "neighborhood_overview", "host_since", "host_response_time",
        "host_response_rate", "host_acceptance_rate", "host_thumbnail_url",
        "host_verifications", "neighbourhood", "host_total_listings_count",
        "host_neighbourhood", "calendar_updated", "instant_bookable",
        "license", "price_quote_checkin_date", "price_quote_checkout_date",
        "price_quote_raw", "host_about",
    ]
    df = df.drop(columns=cols_to_drop)

    # 2. Target rows: keep only listings with a published price
    df = df.dropna(subset=["price"])

    # 3. Parse price strings like "$1,234.00" into floats
    df["price"] = (
        df["price"]
        .str.replace("$", "", regex=False)
        .str.replace(",", "", regex=False)
        .astype(float)
    )

    # 4. Impute room counts with the median
    for col in ["bedrooms", "bathrooms", "beds"]:
        df[col] = df[col].fillna(df[col].median())

    # 5. Review scores / velocity: missing means "no reviews yet"
    review_cols = [
        "review_scores_rating", "review_scores_accuracy",
        "review_scores_cleanliness", "review_scores_checkin",
        "review_scores_communication", "review_scores_location",
        "review_scores_value", "reviews_per_month",
    ]
    df[review_cols] = df[review_cols].fillna(0)

    # 6. Drop rows missing critical fields, then columns that stay sparse
    df = df.dropna(subset=[
        "host_name", "host_is_superhost", "bathrooms_text",
        "minimum_nights", "name", "has_availability",
    ])
    df = df.drop(columns=[
        "last_review", "first_review", "host_location",
        "description", "host_profile_url",
    ])
    df = df.dropna(subset=["price_quote_price_per_night", "price_quote_total_price"])

    return df


def main() -> None:
    df = clean_listings(load_raw_listings())
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "listings_clean.csv", index=False)
    print(f"Saved listings_clean.csv: {df.shape}, missing values: {df.isnull().sum().sum()}")


if __name__ == "__main__":
    main()
