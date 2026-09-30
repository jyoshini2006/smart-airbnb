"""Run the full pipeline end to end: clean -> features -> models -> SHAP.

Usage:
    python src/run_pipeline.py          # standard run
    python src/run_pipeline.py --tune   # also grid-search the price model
"""
from __future__ import annotations

import argparse

import explain
import features as F
import preprocessing
import train_demand
import train_price
from paths import PROCESSED_DIR


def main(tune: bool = False) -> None:
    print("=== 1/5 Preprocessing ===")
    df = preprocessing.clean_listings(preprocessing.load_raw_listings())
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DIR / "listings_clean.csv", index=False)
    print(f"Clean shape: {df.shape}")

    print("\n=== 2/5 Feature engineering ===")
    df = F.engineer_features(df)
    df.to_csv(PROCESSED_DIR / "listings_features.csv", index=False)
    print(f"Feature shape: {df.shape}")

    print("\n=== 3/5 Price model ===")
    train_price.main(tune=tune)

    print("\n=== 4/5 Demand model ===")
    train_demand.main()

    print("\n=== 5/5 Explainability (SHAP) ===")
    explain.explain_all(df)

    print("\nPipeline complete. Artifacts: models/*.pkl, outputs/explainability/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Smart Airbnb full pipeline")
    parser.add_argument("--tune", action="store_true", help="grid-search the best price model")
    args = parser.parse_args()
    main(tune=args.tune)
