"""Train and compare price regression models (diagram stage 6).

Trains the five candidate regressors on log(price), evaluates MAE / RMSE /
R2 on the original dollar scale, optionally grid-searches the best model,
and saves the winner plus the fitted label encoders.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from features import (
    CLF_FEATURE_COLS,
    CATEGORICAL_COLS,
    FEATURE_COLS,
    cap_outliers,
)
from paths import MODELS_DIR, PROCESSED_DIR


def build_models() -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(random_state=42),
        "Random Forest": RandomForestRegressor(random_state=42, n_estimators=100),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "XGBoost": XGBRegressor(random_state=42),
    }


def prepare_xy(df: pd.DataFrame, label_encoders: dict | None = None):
    """Encode categoricals, cap the luxury tail, and split train/test."""
    df, price_cap = cap_outliers(df)

    X = df[FEATURE_COLS].copy()
    y = df["price"]

    if label_encoders is None:
        from sklearn.preprocessing import LabelEncoder

        label_encoders = {}
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col])
            label_encoders[col] = le
    else:
        for col in CATEGORICAL_COLS:
            X[col] = label_encoders[col].transform(X[col])

    y_log = np.log1p(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_log, test_size=0.2, random_state=42
    )
    return X_train, X_test, y_train, y_test, label_encoders, price_cap


def evaluate(models: dict, X_train, X_test, y_train, y_test) -> pd.DataFrame:
    """Fit each model and score it on the original dollar scale."""
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = np.expm1(model.predict(X_test))
        y_actual = np.expm1(y_test)

        results.append({
            "Model": name,
            "MAE": mean_absolute_error(y_actual, y_pred),
            "RMSE": np.sqrt(mean_squared_error(y_actual, y_pred)),
            "R2": r2_score(y_actual, y_pred),
        })
    return pd.DataFrame(results).sort_values("R2", ascending=False).reset_index(drop=True)


def tune_best(model_name: str, X_train, y_train) -> dict:
    """Small grid search around the notebook's winning configuration."""
    grids = {
        "XGBoost": {
            "n_estimators": [100, 200, 300],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.05, 0.1, 0.2],
        },
        "Random Forest": {
            "n_estimators": [100, 200],
            "max_depth": [None, 20],
        },
    }
    if model_name not in grids:
        return {}

    search = GridSearchCV(
        build_models()[model_name],
        grids[model_name],
        scoring="r2",
        cv=3,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    print(f"Best params: {search.best_params_} | CV R2: {search.best_score_:.4f}")
    return search.best_params_


def main(tune: bool = False) -> None:
    df = pd.read_csv(PROCESSED_DIR / "listings_features.csv")

    (X_train, X_test, y_train, y_test,
     label_encoders, price_cap) = prepare_xy(df)
    print(f"Price cap (99th pct): ${price_cap:,.2f} | input rows: {len(df):,}")
    print(f"Train: {X_train.shape} | Test: {X_test.shape}\n")

    models = build_models()
    results = evaluate(models, X_train, X_test, y_train, y_test)
    print(results.to_string(index=False), "\n")

    best_name = results.iloc[0]["Model"]
    best_params = {}
    if tune:
        best_params = tune_best(best_name, X_train, y_train)
        models = build_models()
        models[best_name].set_params(**best_params)
        results = evaluate(models, X_train, X_test, y_train, y_test)
        print("After tuning:")
        print(results.to_string(index=False), "\n")

    best_model = models[best_name]
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_model, MODELS_DIR / "price_model.pkl")
    joblib.dump(label_encoders, MODELS_DIR / "label_encoders.pkl")
    joblib.dump(
        {"feature_cols": FEATURE_COLS, "clf_feature_cols": CLF_FEATURE_COLS,
         "categorical_cols": CATEGORICAL_COLS},
        MODELS_DIR / "feature_config.pkl",
    )
    print(f"Saved {MODELS_DIR / 'price_model.pkl'} ({best_name}, MAE "
          f"${results.iloc[0]['MAE']:.2f}, R2 {results.iloc[0]['R2']:.3f})")


if __name__ == "__main__":
    main()
