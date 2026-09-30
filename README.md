# 🏠 Smart Airbnb Price & Demand Prediction

An end-to-end machine learning project for predicting **nightly Airbnb prices** and **listing demand** using NYC Airbnb listing data from Inside Airbnb.

The project combines:

* 💰 **Price Prediction** — predicts the expected nightly listing price
* 📈 **Demand Prediction** — classifies a listing as High or Low demand
* 🔍 **SHAP Explainability** — explains which features influence model predictions
* 📊 **Interactive Streamlit Dashboard** — allows users to explore the data and generate predictions

---

## 🎯 Project Objective

Airbnb hosts need to understand how property characteristics, location, availability, reviews, and host activity influence listing performance.

This project builds an ML-powered decision-support system that helps analyze:

1. What price a listing could be offered at.
2. Whether a listing is likely to have high or low demand.
3. Which factors influence the model's predictions.
4. How these insights can be explored through an interactive dashboard.

---

## 🔄 Project Workflow

```text
NYC Airbnb Data
       ↓
Data Cleaning & Preprocessing
       ↓
Exploratory Data Analysis
       ↓
Feature Engineering
       ↓
 ┌───────────────────────┐
 │                       │
 ↓                       ↓
Price Prediction     Demand Prediction
Regression           Classification
 │                       │
 ↓                       ↓
Model Evaluation     Model Evaluation
 │                       │
 └───────────┬───────────┘
             ↓
       SHAP Explainability
             ↓
    Streamlit Dashboard
             ↓
       Business Insights
```

---

## ✨ Key Features

### 1. Price Prediction

The system predicts the expected nightly price of an Airbnb listing using features such as:

* Neighbourhood
* Room type
* Number of guests accommodated
* Bedrooms
* Bathrooms
* Beds
* Minimum nights
* Number of reviews
* Review rating
* Availability
* Host listing count
* Superhost status
* Review velocity
* Host activity
* Distance from Manhattan

The target is transformed using:

```python
log1p(price)
```

and converted back to the original dollar scale using:

```python
expm1(prediction)
```

---

### 2. Demand Prediction

The project creates a **High/Low demand proxy label** because the dataset does not contain a direct demand column.

A listing is classified as High demand when its estimated booked nights over the previous 365 days are greater than zero.

```text
estimated_occupancy_l365d > 0
        ↓
     High Demand
```

Otherwise:

```text
estimated_occupancy_l365d = 0
        ↓
      Low Demand
```

This is explicitly treated as a **proxy label**, not ground-truth demand.

---

### 3. SHAP Explainability

SHAP is used to understand the contribution of individual features to model predictions.

The project generates:

* Price model SHAP summary
* Demand model SHAP summary
* Feature importance table

This makes the ML system more interpretable instead of treating the models as black boxes.

---

### 4. Interactive Dashboard

The project includes a Streamlit dashboard with:

* Dataset overview
* Price distribution
* Reviews vs. price analysis
* Availability distribution
* Geographic listing map
* Price model feature importance
* Demand model feature importance
* SHAP explanations
* Individual listing prediction

Run the dashboard with:

```bash
streamlit run dashboard/app.py
```

---

# 📂 Project Structure

```text
smart-airbnb/
│
├── data/
│   ├── raw/
│   │   ├── listings.csv.gz
│   │   ├── calendar.csv.gz
│   │   ├── reviews.csv.gz
│   │   └── neighbourhoods.csv
│   │
│   └── processed/
│       ├── listings_clean.csv
│       └── listings_features.csv
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── features.py
│   ├── train_price.py
│   ├── train_demand.py
│   ├── explain.py
│   ├── prediction.py
│   ├── paths.py
│   └── run_pipeline.py
│
├── models/
│   └── Generated model artifacts
│
├── outputs/
│   └── explainability/
│       ├── price_shap_summary.png
│       ├── demand_shap_summary.png
│       └── feature_importance.csv
│
├── dashboard/
│   └── app.py
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

# 🧹 Data Preprocessing

The preprocessing stage prepares the raw Airbnb data for machine learning.

The pipeline handles:

* Missing values
* Data type conversion
* Unnecessary columns
* Invalid records
* Price cleaning
* Listing-level feature preparation

The processed datasets are stored in:

```text
data/processed/
```

---

# ⚙️ Feature Engineering

Several additional features are created to improve the models.

### Review Velocity

```text
review_velocity = reviews_per_month
```

Represents the rate at which reviews are received.

### Availability Rate

```text
availability_rate = availability_365 / 365
```

Represents the proportion of the year for which a listing is available.

### Minimum Nights Category

Listings are grouped into:

```text
short
weekly
monthly
long_term
```

### Host Activity Score

```text
host_activity_score =
    0.5 × host_listings_count
    +
    0.5 × number_of_reviews
```

### Distance to Manhattan

The project calculates geographical distance from the Manhattan reference point using latitude and longitude.

### Price Category

Price is also segmented into:

```text
budget
mid
premium
luxury
```

This is used for **analysis only** and is not used as a model input because it is directly derived from the target variable.

---

# 🤖 Machine Learning Models

## Price Prediction

Candidate regression models are evaluated before selecting the best-performing model.

The training pipeline supports:

* XGBoost
* Random Forest
* Additional regression configurations used during experimentation

The target variable is modeled on the logarithmic scale.

### Price Model Result

| Metric | Result |
| ------ | -----: |
| MAE    |   ~$69 |
| RMSE   |  ~$134 |
| R²     |  ~0.63 |

The final saved price model is generated by the training pipeline.

---

## Demand Prediction

Four classification models are compared:

1. Logistic Regression
2. Decision Tree
3. Random Forest
4. XGBoost

The models are evaluated using:

* Accuracy
* Precision
* Recall
* F1-score

### Demand Model Result

The Random Forest model produced the strongest F1-score in the tested comparison.

| Model               |  Accuracy | Precision | Recall |        F1 |
| ------------------- | --------: | --------: | -----: | --------: |
| Random Forest       | **0.805** | **0.823** |  0.724 | **0.771** |
| XGBoost             |     0.795 |     0.805 |  0.721 |     0.761 |
| Logistic Regression |     0.762 |     0.777 |  0.663 |     0.716 |
| Decision Tree       |     0.739 |     0.710 |  0.712 |     0.711 |

---

## 📊 Confusion Matrix

For the Random Forest demand model:

```text
                 Predicted
              Low       High

Actual Low    2017       296
Actual High    525      1379
```

This corresponds to:

```text
[[2017, 296],
 [ 525, 1379]]
```

---

# 🔍 Explainability

The project uses SHAP to provide global model explanations.

Generated artifacts include:

```text
outputs/explainability/
├── price_shap_summary.png
├── demand_shap_summary.png
└── feature_importance.csv
```

The dashboard displays these explanations under the **Model Insights** section.

Important features identified by the project's explainability analysis include factors such as:

* Minimum nights
* Accommodates
* Room type
* Distance from Manhattan
* Superhost status
* Price

---

# 🖥️ Dashboard

The Streamlit application provides an interactive interface for exploring the Smart Airbnb system.

### Dashboard capabilities

**Data Exploration**

* Price distribution
* Reviews vs. price
* Availability distribution
* Geographic listing map

**Model Insights**

* Price model feature importance
* Demand model feature importance
* SHAP summary plots
* Feature importance table

**Prediction**

Users can provide listing characteristics and obtain:

```text
Predicted Price
        +
Demand Classification
        +
Demand Probability
```

---

# 📸 Dashboard Screenshots

Add the final screenshots to the repository, for example:

```text
docs/
└── screenshots/
    ├── dashboard-overview.png
    ├── prediction.png
    └── model-insights.png
```

Then reference them here:

### Dashboard Overview

![Smart Airbnb Dashboard](docs/screenshots/dashboard-overview.png)

### Prediction

![Smart Airbnb Prediction](docs/screenshots/prediction.png)

### Model Insights

![Smart Airbnb Model Insights](docs/screenshots/model-insights.png)

---

# 🚀 Installation

Clone the repository:

```bash
git clone https://github.com/jyoshini2006/smart-airbnb.git
cd smart-airbnb
```

Create a virtual environment:

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux

```bash
python -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# 📥 Dataset

This project uses NYC Airbnb data from **Inside Airbnb**.

Download the required files from:

https://insideairbnb.com/get-the-data/

Place the following files inside:

```text
data/raw/
```

```text
listings.csv.gz
calendar.csv.gz
reviews.csv.gz
neighbourhoods.csv
```

The raw datasets are intentionally excluded from Git because of their size.

---

# ▶️ Running the Project

## Run the complete ML pipeline

From the project root:

```bash
python src/run_pipeline.py
```

The pipeline performs the major stages in sequence:

```text
Preprocessing
    ↓
Feature Engineering
    ↓
Price Model Training
    ↓
Demand Model Training
    ↓
SHAP Explainability
```

For price-model hyperparameter tuning:

```bash
python src/run_pipeline.py --tune
```

---

## Launch the Dashboard

```bash
streamlit run dashboard/app.py
```

The application will open locally in your browser.

---

# 🛡️ Important ML Design Considerations

### Demand is a proxy

The project does not claim to have a direct ground-truth demand variable.

The demand label is derived from estimated occupancy:

```python
estimated_occupancy_l365d > 0
```

Therefore, demand predictions should be interpreted as predictions of this **proxy definition**.

### Target leakage prevention

`price_category` is derived directly from price.

Therefore, it is used for analysis only and is not included in the prediction feature set.

Occupancy-derived variables are also kept out of the demand classifier's input features.

### Price outlier handling

The extreme luxury tail is trimmed above the 99th percentile before price-model training.

### Log transformation

The price target is transformed using:

```python
np.log1p(price)
```

Predictions are converted back using:

```python
np.expm1(prediction)
```

---

# 💡 Business Insights

The analysis is designed to help Airbnb hosts understand how factors such as:

* Location
* Room type
* Minimum-night requirements
* Availability
* Reviews
* Host activity
* Superhost status
* Property capacity

relate to listing price and the project's demand proxy.

These insights can support future development of pricing and listing-management tools.

---

# 🧰 Technology Stack

| Category          | Technology            |
| ----------------- | --------------------- |
| Programming       | Python                |
| Data Processing   | Pandas, NumPy         |
| Visualization     | Matplotlib, Streamlit |
| Machine Learning  | Scikit-learn          |
| Gradient Boosting | XGBoost               |
| Explainability    | SHAP                  |
| Model Persistence | Joblib                |
| Notebook          | Jupyter               |
| Version Control   | Git / GitHub          |

---

# 🔮 Future Enhancements

Potential future extensions include:

* Real-time Airbnb data ingestion
* Dynamic pricing recommendations
* More advanced demand forecasting
* FastAPI backend
* PostgreSQL database
* MLflow experiment tracking
* Docker containerization
* Cloud deployment
* Automated model retraining
* Personalized host recommendations
* Time-series demand forecasting

---

# 👩‍💻 Author

**Jyoshini N T**

Artificial Intelligence and Data Science
Rajalakshmi Engineering College

---

## 📌 Project Status

**Core machine learning pipeline:** ✅ Completed

**Explainability layer:** ✅ Completed

**Interactive dashboard:** ✅ Completed

**GitHub repository:** ✅ Published

**Final documentation:** 🔄 Being finalized

**Cloud deployment:** ⏳ Planned
