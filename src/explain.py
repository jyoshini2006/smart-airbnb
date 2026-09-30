"""Model explainability (diagram stage 8).

Uses SHAP to explain both models: global feature importance (bar +
beeswarm plots) and per-prediction waterfalls, for the price regressor
and the demand classifier.
"""
from __future__ import annotations

import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from features import (CLF_FEATURE_COLS, CATEGORICAL_COLS, FEATURE_COLS,
                      add_demand_label, cap_outliers)
from paths import EXPLAIN_DIR, MODELS_DIR, PROCESSED_DIR

ARTIFACT_DIR = EXPLAIN_DIR


def _load_models():
    import prediction

    price_model = prediction._load_any("price_model")
    demand_model = prediction._load_any("demand_model")
    label_encoders = joblib.load(MODELS_DIR / "label_encoders.pkl")
    return price_model, demand_model, label_encoders


def _prepare_background(df: pd.DataFrame, label_encoders: dict) -> pd.DataFrame:
    """Encoded feature frame used to fit the SHAP explainers."""
    df = add_demand_label(df)
    df, _ = cap_outliers(df)

    X_price = df[FEATURE_COLS].copy()
    for col in CATEGORICAL_COLS:
        X_price[col] = label_encoders[col].transform(X_price[col])

    X_clf = df[CLF_FEATURE_COLS].copy()
    for col in CATEGORICAL_COLS:
        X_clf[col] = label_encoders[col].transform(X_clf[col])

    return X_price, X_clf


def explain_all(df: pd.DataFrame, background_size: int = 500) -> pd.DataFrame:
    """Generate all SHAP artifacts; returns the global importance table."""
    os.makedirs(ARTIFACT_DIR, exist_ok=True)
    price_model, demand_model, label_encoders = _load_models()
    X_price, X_clf = _prepare_background(df, label_encoders)

    # --- Price regressor -------------------------------------------------
    price_explainer = shap.TreeExplainer(price_model)
    price_bg = X_price.sample(min(background_size, len(X_price)), random_state=42)
    price_sv = price_explainer.shap_values(price_bg)
    price_importance = (
        pd.DataFrame({"feature": X_price.columns,
                      "mean_abs_shap": np.abs(price_sv).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
    )

    plt.figure()
    shap.summary_plot(price_sv, price_bg, show=False)
    plt.tight_layout()
    plt.savefig(f"{ARTIFACT_DIR}/price_shap_summary.png", dpi=150)
    plt.close()

    # --- Demand classifier -----------------------------------------------
    demand_explainer = shap.TreeExplainer(demand_model)
    demand_bg = X_clf.sample(min(background_size, len(X_clf)), random_state=42)
    demand_sv = demand_explainer.shap_values(demand_bg)
    if isinstance(demand_sv, list):  # older shap: [class0, class1]
        demand_sv = demand_sv[1]
    elif getattr(demand_sv, "ndim", 2) == 3:  # newer shap: (rows, features, classes)
        demand_sv = demand_sv[:, :, 1]
    demand_importance = (
        pd.DataFrame({"feature": X_clf.columns,
                      "mean_abs_shap": np.abs(demand_sv).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
    )

    plt.figure()
    shap.summary_plot(demand_sv, demand_bg, show=False)
    plt.tight_layout()
    plt.savefig(f"{ARTIFACT_DIR}/demand_shap_summary.png", dpi=150)
    plt.close()

    importance = price_importance.merge(
        demand_importance, on="feature", how="outer", suffixes=("_price", "_demand")
    ).fillna(0)
    importance.to_csv(f"{ARTIFACT_DIR}/feature_importance.csv", index=False)
    print(f"Saved SHAP artifacts to {ARTIFACT_DIR}/")
    return importance


def explain_prediction(price_model, input_df: pd.DataFrame):
    """Per-prediction SHAP waterfall for one encoded price row."""
    explainer = shap.TreeExplainer(price_model)
    sv = explainer(input_df)
    fig = plt.figure()
    shap.plots.waterfall(sv[0], show=False)
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    explain_all(pd.read_csv(PROCESSED_DIR / "listings_features.csv"))
