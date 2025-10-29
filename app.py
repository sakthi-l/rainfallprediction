import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(
    page_title="Rainfall Forecast & Flood/Drought Prediction",
    page_icon="🌦️",
    layout="wide"
)

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("Navigation")
mode = st.sidebar.radio("Select Mode", ["Home", "Rainfall Forecasting", "Flood/Drought Prediction"])

st.sidebar.markdown("---")
st.sidebar.info("This minimal app provides text-only predictions. No charts or analytics.")

# ============================================================
# HOME PAGE
# ============================================================
if mode == "Home":
    st.title("🌦️ Rainfall & 🌊 Flood/Drought Prediction App")
    st.markdown("""
    This app provides **rainfall forecasts** using LASSO regularization  
    and **flood/drought classification** using XGBoost.
    
    📁 Expected files in your repository:
    - `Rainfall_Data_LL.csv` → for Rainfall Forecasting  
    - `rainfallpred.csv` → for Flood/Drought Prediction  
    """)

# ============================================================
# 🌦️ RAINFALL FORECASTING (LASSO)
# ============================================================
elif mode == "Rainfall Forecasting":
    st.title("🌦️ Rainfall Forecasting (LASSO Model)")

    try:
        df = pd.read_csv("Rainfall_Data_LL.csv")
        st.success("✅ Dataset loaded successfully: Rainfall_Data_LL.csv")
    except Exception as e:
        st.error(f"⚠️ Unable to load dataset: {e}")
        st.stop()

    # Use exact known columns
    expected_cols = {"SUBDIVISION", "YEAR", "ANNUAL"}
    if not expected_cols.issubset(df.columns):
        st.error("❌ Dataset must contain columns: SUBDIVISION, YEAR, and ANNUAL.")
        st.stop()

    df = df.dropna(subset=["SUBDIVISION", "YEAR", "ANNUAL"])
    regions = sorted(df["SUBDIVISION"].unique())

    region = st.selectbox("Select Region", regions)
    year_to_predict = st.number_input(
        "Enter Year to predict",
        min_value=int(df["YEAR"].min()),
        max_value=int(df["YEAR"].max()) + 20,
        value=int(df["YEAR"].max()) + 1
    )

    if st.button("Predict Rainfall"):
        df_region = df[df["SUBDIVISION"] == region].dropna(subset=["ANNUAL"])
        X = np.array(df_region["YEAR"]).reshape(-1, 1)
        y = df_region["ANNUAL"].values

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        tscv = TimeSeriesSplit(n_splits=5)
        model = LassoCV(cv=tscv)
        model.fit(X_scaled, y)

        year_scaled = scaler.transform([[year_to_predict]])
        y_pred = model.predict(year_scaled)[0]

        st.success("✅ Prediction completed.")
        st.write(f"**Region:** {region}")
        st.write(f"**Year:** {year_to_predict}")
        st.write(f"**Predicted Annual Rainfall:** {y_pred:.2f} mm")

        if year_to_predict in df_region["YEAR"].values:
            actual = df_region[df_region["YEAR"] == year_to_predict]["ANNUAL"].values[0]
            st.write(f"**Actual Annual Rainfall (available):** {actual:.2f} mm")

# ============================================================
# 🌊 FLOOD/DROUGHT PREDICTION (XGBoost)
# ============================================================
elif mode == "Flood/Drought Prediction":
    st.title("🌊 Flood/Drought Prediction (XGBoost Model)")

    try:
        df = pd.read_csv("rainfallpred.csv")
        st.success("✅ Dataset loaded successfully: rainfallpred.csv")
    except Exception as e:
        st.error(f"⚠️ Unable to load dataset: {e}")
        st.stop()

    # Check required columns
    if not {"SUBDIVISION", "YEAR", "ANNUAL"}.issubset(df.columns):
        st.error("❌ Dataset must contain columns: SUBDIVISION, YEAR, and ANNUAL.")
        st.stop()

    df = df.dropna(subset=["SUBDIVISION", "ANNUAL"])
    # Automatically create a label column
    avg_rain = df["ANNUAL"].mean()
    df["LABEL"] = df["ANNUAL"].apply(lambda x: "Flood" if x > avg_rain * 1.1 else ("Drought" if x < avg_rain * 0.9 else "Normal"))

    st.info("💡 Labels auto-generated based on rainfall intensity thresholds.")

    X = df[["ANNUAL"]]
    y = df["LABEL"]

    clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    clf.fit(X, y)

    rainfall_input = st.number_input("Enter rainfall (mm) to classify:", 0, 3000, 1200)
    pred = clf.predict([[rainfall_input]])[0]

    st.success("✅ Prediction completed.")
    st.write(f"**Input Rainfall:** {rainfall_input:.2f} mm")
    st.write(f"**Predicted Condition:** {pred}")
