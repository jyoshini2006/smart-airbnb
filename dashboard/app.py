"""Smart Airbnb Intelligence - Streamlit dashboard (diagram stage 9).

Tabs: Overview | Price & Demand Prediction | Market Analysis | Model Insights.
Run from the project root with:  streamlit run dashboard/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

import prediction as P  # noqa: E402
import features as F  # noqa: E402

st.set_page_config(page_title="Smart Airbnb Intelligence", page_icon="🏠", layout="wide")

EXPLAIN_DIR = ROOT / "outputs" / "explainability"


@st.cache_resource(show_spinner="Loading models...")
def load_all():
    price_model, demand_model, encoders, config = P.load_models()
    listings = pd.read_csv(ROOT / "data" / "processed" / "listings_features.csv")
    return price_model, demand_model, encoders, config, listings


@st.cache_data(show_spinner=False)
def borough_stats(listings: pd.DataFrame) -> pd.DataFrame:
    return (
        listings.groupby("neighbourhood_group_cleansed")["price"]
        .agg(median_price="median", avg_price="mean", listings="count")
        .sort_values("median_price", ascending=False)
        .round(2)
    )


try:
    price_model, demand_model, label_encoders, config, listings = load_all()
except FileNotFoundError:
    st.error("Models not found. Train them first:\n\n```\npython src/run_pipeline.py\n```")
    st.stop()

st.title("🏠 Smart Airbnb Intelligence")
st.markdown("Predict nightly **price** and **demand** for NYC Airbnb listings - with explainability.")

tab_overview, tab_predict, tab_market, tab_insights = st.tabs(
    ["📊 Overview", "🔮 Predict", "🗺️ Market Analysis", "🧠 Model Insights"]
)

# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Listings analyzed", f"{len(listings):,}")
    c2.metric("Median price", f"${listings['price'].median():,.0f} / night")
    c3.metric("Avg review score", f"{listings['review_scores_rating'].mean():.2f} / 5")
    high_share = (listings["estimated_occupancy_l365d"] > 0).mean() * 100
    c4.metric("High-demand share", f"{high_share:.0f}%")

    left, right = st.columns(2)
    with left:
        st.subheader("Median price by borough")
        st.bar_chart(borough_stats(listings)["median_price"])
    with right:
        st.subheader("Listings by room type")
        st.bar_chart(listings["room_type"].value_counts())

    st.subheader("Price distribution (listings up to $500)")
    visible = listings[listings["price"] <= 500]
    hist = pd.cut(visible["price"], bins=40).value_counts().sort_index()
    hist.index = hist.index.astype(str)  # interval index -> labels for charting
    st.bar_chart(hist, x_label="Price bin ($)", y_label="Listings")

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
with tab_predict:
    st.sidebar.header("Enter Listing Details")

    neighbourhood = st.sidebar.selectbox(
        "Neighbourhood Group", sorted(label_encoders["neighbourhood_group_cleansed"].classes_)
    )
    room_type = st.sidebar.selectbox("Room Type", sorted(label_encoders["room_type"].classes_))

    accommodates = st.sidebar.number_input("Accommodates", 1, 16, 2)
    bedrooms = st.sidebar.number_input("Bedrooms", 0, 10, 1)
    bathrooms = st.sidebar.number_input("Bathrooms", 0.0, 10.0, 1.0, step=0.5)
    beds = st.sidebar.number_input("Beds", 0, 16, 1)
    minimum_nights = st.sidebar.number_input("Minimum Nights", 1, 365, 2)
    number_of_reviews = st.sidebar.slider("Number of Reviews", 0, 1000, 45)
    review_scores_rating = st.sidebar.slider("Review Score", 0.0, 5.0, 4.5, 0.01)
    reviews_per_month = st.sidebar.slider("Reviews per Month", 0.0, 20.0, 1.0, 0.1)
    availability_365 = st.sidebar.slider("Availability (days/year)", 0, 365, 180)
    host_listings_count = st.sidebar.number_input("Host Listings Count", 1, 500, 1)
    host_is_superhost = "t" if st.sidebar.checkbox("Host is Superhost") else "f"

    if st.sidebar.button("🔮 Predict", type="primary", use_container_width=True):
        details = {
            "neighbourhood_group_cleansed": neighbourhood,
            "room_type": room_type,
            "accommodates": int(accommodates),
            "bedrooms": int(bedrooms),
            "bathrooms": float(bathrooms),
            "beds": int(beds),
            "minimum_nights": int(minimum_nights),
            "number_of_reviews": int(number_of_reviews),
            "review_scores_rating": float(review_scores_rating),
            "reviews_per_month": float(reviews_per_month),
            "availability_365": int(availability_365),
            "host_listings_count": int(host_listings_count),
            "host_is_superhost": host_is_superhost,
        }

        input_row = P.build_input_row(details, label_encoders)
        result = P.predict(price_model, demand_model, input_row)

        col1, col2 = st.columns(2)
        col1.metric("💰 Predicted Price", f"${result['price']:,.0f} / night")
        col2.metric("📈 Demand Level", result["demand"],
                    f"{result['demand_probability']:.0%} chance of High")

        st.progress(result["demand_probability"], text="Demand probability")

        map_col, feat_col = st.columns(2)

        with map_col:
            st.subheader("📍 Similar listings in this borough")
            borough = listings[listings["neighbourhood_group_cleansed"] == neighbourhood]
            if len(borough):
                st.map(
                    borough[["latitude", "longitude", "price"]]
                    .sample(min(300, len(borough)), random_state=42),
                    latitude="latitude", longitude="longitude",
                    size=15, zoom=10,
                )
            else:
                st.info("No listings available for this selection.")

        with feat_col:
            st.subheader("🔑 Top features driving this price")
            importances = pd.DataFrame({
                "Feature": config["feature_cols"],
                "Importance": price_model.feature_importances_,
            }).sort_values("Importance", ascending=False).head(8)
            st.bar_chart(importances.set_index("Feature"))

            with st.expander("Explain this prediction with SHAP"):
                try:
                    from explain import explain_prediction

                    fig = explain_prediction(price_model, input_row[F.FEATURE_COLS])
                    st.pyplot(fig)
                except Exception as err:  # noqa: BLE001
                    st.caption(f"SHAP waterfall unavailable: {err}")

# ---------------------------------------------------------------------------
# Market Analysis
# ---------------------------------------------------------------------------
with tab_market:
    st.subheader("Borough comparison")
    st.dataframe(borough_stats(listings), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.subheader("Median price by room type")
        room_stats = listings.groupby("room_type")["price"].median().sort_values(ascending=False)
        st.bar_chart(room_stats)
    with right:
        st.subheader("Reviews vs price")
        sample = listings[listings["price"] <= 500].sample(min(3000, len(listings)), random_state=42)
        st.scatter_chart(
            sample, x="price", y="number_of_reviews",
            x_label="Price ($)", y_label="Number of reviews", height=350,
        )

    left, right = st.columns(2)
    with left:
        st.subheader("Availability distribution")
        avail = pd.cut(listings["availability_365"], bins=[0, 30, 90, 180, 270, 365])
        avail_counts = avail.value_counts().sort_index()
        avail_counts.index = avail_counts.index.astype(str)
        st.bar_chart(avail_counts, x_label="Available days / year", y_label="Listings")
    with right:
        st.subheader("Geographic price map (listings up to $500)")
        geo = listings[listings["price"] <= 500].sample(min(2000, len(listings)), random_state=42)
        st.map(geo, latitude="latitude", longitude="longitude", size=8)

# ---------------------------------------------------------------------------
# Model Insights
# ---------------------------------------------------------------------------
with tab_insights:
    st.subheader("Feature importance - price model")
    price_imp = (
        pd.DataFrame({"Feature": config["feature_cols"],
                      "Importance": price_model.feature_importances_})
        .sort_values("Importance", ascending=False)
    )
    st.bar_chart(price_imp.set_index("Feature"))

    st.subheader("Feature importance - demand model")
    demand_imp = (
        pd.DataFrame({"Feature": config["clf_feature_cols"],
                      "Importance": demand_model.feature_importances_})
        .sort_values("Importance", ascending=False)
    )
    st.bar_chart(demand_imp.set_index("Feature"))

    st.subheader("Global SHAP explanations")
    price_png = EXPLAIN_DIR / "price_shap_summary.png"
    demand_png = EXPLAIN_DIR / "demand_shap_summary.png"
    if price_png.exists() and demand_png.exists():
        left, right = st.columns(2)
        left.image(str(price_png), caption="Price model - SHAP summary", use_container_width=True)
        right.image(str(demand_png), caption="Demand model - SHAP summary", use_container_width=True)

        importance_csv = EXPLAIN_DIR / "feature_importance.csv"
        if importance_csv.exists():
            st.dataframe(pd.read_csv(importance_csv), use_container_width=True)
    else:
        st.info("SHAP artifacts not generated yet. Run:\n\n```\npython src/explain.py\n```")
