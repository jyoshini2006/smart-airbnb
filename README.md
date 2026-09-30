# 🏠 Smart Airbnb Price & Demand Prediction

End-to-end machine learning project for NYC Airbnb listings (Inside Airbnb data):

1. **Price prediction (regression)** — expected nightly price, e.g. `$185/night`
2. **Demand prediction (classification)** — High vs Low demand, e.g. `High Demand`

Plus a **SHAP explainability layer** and an interactive **Streamlit dashboard**.

```
Data Collection → Cleaning → EDA → Feature Engineering → Price Model + Demand Model
     → Explainability (SHAP) → Streamlit Dashboard → Business Insights
```

## Project structure

```
smart-airbnb/
├── data/
│   ├── raw/                  # Inside Airbnb dumps (listings, calendar, reviews, neighbourhoods)
│   └── processed/            # listings_clean.csv, listings_features.csv
├── notebooks/
│   └── 01_data_exploration.ipynb   # EDA + exploratory modeling (executed end-to-end)
├── src/
│   ├── preprocessing.py      # cleaning: missing values, dtypes, drops
│   ├── features.py           # engineered features + demand label + feature lists
│   ├── train_price.py        # regression model comparison + tuning
│   ├── train_demand.py       # classification model comparison + confusion matrix
│   ├── explain.py            # SHAP global + per-prediction explanations
│   ├── prediction.py         # single-listing inference used by the dashboard
│   └── run_pipeline.py       # runs everything in order
├── models/                   # saved .pkl artifacts (recreated by the pipeline)
├── outputs/explainability/   # SHAP summary plots + importance table
├── dashboard/app.py          # Streamlit application
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows  (source venv/bin/activate on macOS/Linux)
pip install -r requirements.txt
```

Download the NYC listings data from [insideairbnb.com/get-the-data](https://insideairbnb.com/get-the-data/)
and place `listings.csv.gz`, `calendar.csv.gz`, `reviews.csv.gz`, `neighbourhoods.csv` in `data/raw/`.

## Run the full pipeline

```bash
python src/run_pipeline.py            # clean → features → both models → SHAP
python src/run_pipeline.py --tune     # also grid-search the winning price model
streamlit run dashboard/app.py        # launch the dashboard
```

## Results (NYC, 20.9k listings after cleaning)

| Price model | MAE | RMSE | R² |
|---|---|---|---|
| XGBoost (log-target, tuned) | ~$69 | ~$134 | ~0.63 |

| Demand model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Random Forest | 0.805 | 0.823 | 0.724 | 0.771 |

Top price drivers (mean |SHAP|): `minimum_nights`, `accommodates`, `room_type`, `distance_to_manhattan`.
Top demand drivers: `minimum_nights`, `host_is_superhost`, `price`.

All numbers above are reproduced by `python src/run_pipeline.py`, which can be
run from any working directory (all paths resolve from the project root).

## Design notes

- **Demand label** — there is no raw "demand" column, so a listing is *High demand*
  when its estimated booked nights in the last 365 days is > 0. This is a **proxy**
  (the estimate itself derives from review activity), so no occupancy-derived column
  is ever fed into the classifier.
- **Target safety** — `price_category` is derived from price and is used for analysis
  only, never as a model input (target leakage).
- **Outliers** — prices above the 99th percentile are trimmed before training; the
  target is modeled as `log1p(price)` and predictions are inverted with `expm1`.
- **Encoding** — LabelEncoder + tree models handle the categoricals; Logistic
  Regression is wrapped in a `StandardScaler` pipeline so it converges cleanly.

## Business insights

- Manhattan commands the highest median nightly price; room type (Entire home vs
  Private room) is the biggest single lever a host controls.
- Long minimum-night requirements correlate with both higher prices and the
  high-demand label (monthly-stay inventory dominates NYC supply).
- Very high availability often flags weak booking activity — a useful signal for
  hosts reconsidering pricing.

## Future enhancements (not implemented)

Real-time data ingestion · dynamic pricing recommendations · FastAPI backend ·
PostgreSQL storage · MLflow experiment tracking · Docker/cloud deployment.
