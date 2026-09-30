"""Train and compare demand classification models (diagram stage 7).

Builds the High/Low demand proxy label, trains four candidate classifiers,
evaluates accuracy / precision / recall / F1 plus the confusion matrix, and
saves the best model.
"""
from __future__ import annotations

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from features import CLF_FEATURE_COLS, CATEGORICAL_COLS, add_demand_label
from paths import MODELS_DIR, PROCESSED_DIR


def build_models() -> dict:
    return {
        # LR needs scaled features to converge; other models are scale-invariant.
        "Logistic Regression": make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=2000, random_state=42)
        ),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100),
        "XGBoost": XGBClassifier(random_state=42),
    }


def prepare_xy(df: pd.DataFrame, label_encoders: dict):
    X = df[CLF_FEATURE_COLS].copy()
    y = df["demand_label"]

    for col in CATEGORICAL_COLS:
        X[col] = label_encoders[col].transform(X[col])

    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def evaluate(models: dict, splits) -> pd.DataFrame:
    X_train, X_test, y_train, y_test = splits
    results = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        results.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1": f1_score(y_test, y_pred),
        })
    return pd.DataFrame(results).sort_values("F1", ascending=False).reset_index(drop=True)


def main() -> None:
    df = pd.read_csv(PROCESSED_DIR / "listings_features.csv")
    df = add_demand_label(df)

    splits = prepare_xy(df, joblib.load(MODELS_DIR / "label_encoders.pkl"))
    print(f"Train: {splits[0].shape} | Test: {splits[1].shape}")

    models = build_models()
    results = evaluate(models, splits)
    print(results.to_string(index=False), "\n")

    best_name = results.iloc[0]["Model"]
    best_model = models[best_name]

    y_test = splits[3]
    y_pred = best_model.predict(splits[1])
    cm = confusion_matrix(y_test, y_pred)
    print(f"Confusion matrix [{best_name}] (rows=actual Low,High / cols=pred Low,High):")
    print(cm, "\n")

    joblib.dump(best_model, MODELS_DIR / "demand_model.pkl")
    print(f"Saved {MODELS_DIR / 'demand_model.pkl'} ({best_name}, "
          f"F1 {results.iloc[0]['F1']:.3f}, Accuracy {results.iloc[0]['Accuracy']:.3f})")


if __name__ == "__main__":
    main()
