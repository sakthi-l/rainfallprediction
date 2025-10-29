# ============================================================
# 🌦️ RAINFALL FORECASTING (LASSO) + 🌊 FLOOD/DROUGHT PREDICTION (XGBoost)
# Auto-Detect Columns | Minimal Text-Only UI
# ============================================================

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
# PAGE CONFIGURATION
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
    
    Upload your datasets in your GitHub repo:
    - `Rainfall_Data_LL.csv` → for Rainfall Forecasting  
    - `rainfallpred.csv` → for Flood/Drought Classification  
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

    df.columns = df.columns.str.strip().str.upper()

    # --- Auto-detect columns ---
    region_col, year_col, rain_col = None, None, None
    for col in df.columns:
        if any(x in col for x in ["REGION", "STATE", "PLACE", "AREA"]):
            region_col = col
        elif any(x in col for x in ["YEAR"]):
            year_col = col
        elif any(x in col for x in ["ANNUAL", "RAIN", "PRECIP"]):
            rain_col = col

    if not all([region_col, year_col, rain_col]):
        st.error("❌ Could not detect region, year, or rainfall column names.")
        st.info("Ensure your dataset includes columns related to region, year, and annual rainfall.")
        st.stop()

    # Clean dataframe
    df = df.dropna(subset=[region_col, year_col, rain_col])
    regions = sorted(df[region_col].unique())
    region = st.selectbox("Select Region", regions)
    year_to_predict = st.number_input(
        "Enter Year to predict",
        min_value=int(df[year_col].min()),
        max_value=int(df[year_col].max()) + 20,
        value=int(df[year_col].max()) + 1
    )

    if st.button("Predict Rainfall"):
        df_region = df[df[region_col] == region].dropna(subset=[rain_col])
        X = np.array(df_region[year_col]).reshape(-1, 1)
        y = df_region[rain_col].values

        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        tscv = TimeSeriesSplit(n_splits=5)
        model = LassoCV(cv=tscv)
        model.fit(X_scaled, y)

        # Predict rainfall for selected year
        year_scaled = scaler.transform([[year_to_predict]])
        y_pred = model.predict(year_scaled)[0]

        st.success("✅ Prediction completed.")
        st.write(f"**Region:** {region}")
        st.write(f"**Year:** {year_to_predict}")
        st.write(f"**Predicted Annual Rainfall:** {y_pred:.2f} mm")

        # Show actual value if available
        if year_to_predict in df_region[year_col].values:
            actual = df_region[df_region[year_col] == year_to_predict][rain_col].values[0]
            st.write(f"**Actual Annual Rainfall (available):** {actual:.2f} mm")

# ============================================================
# 🌊 FLOOD/DROUGHT CLASSIFICATION (XGBoost)
# ============================================================

elif mode == "Flood/Drought Prediction":
    st.title("🌊 Flood/Drought Prediction (XGBoost Model)")

    try:
        df = pd.read_csv("rainfallpred.csv")
        st.success("✅ Dataset loaded successfully: rainfallpred.csv")
    except Exception as e:
        st.error(f"⚠️ Unable to load dataset: {e}")
        st.stop()

    df.columns = df.columns.str.strip().str.upper()

    # --- Auto-detect rainfall & label columns ---
    rainfall_col, label_col = None, None
    for col in df.columns:
        if any(x in col for x in ["RAIN", "PRECIP", "TOTAL"]):
            rainfall_col = col
        if any(x in col for x in ["LABEL", "CLASS", "STATUS", "TYPE", "CATEGORY"]):
            label_col = col

    if rainfall_col is None or label_col is None:
        st.error("❌ Could not automatically detect rainfall and label columns.")
        st.info("Please ensure dataset contains columns with 'RAIN' and 'LABEL' or 'CLASS'.")
        st.stop()

    st.write(f"🌧 Using rainfall column: `{rainfall_col}`")
    st.write(f"🏷 Using label column: `{label_col}`")

    df = df.dropna(subset=[rainfall_col, label_col]).reset_index(drop=True)
    X = df[[rainfall_col]]
    y = df[label_col]

    clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    clf.fit(X, y)

    rainfall_input = st.number_input("Enter rainfall (mm) to classify:", 0, 3000, 1200)
    pred = clf.predict([[rainfall_input]])[0]

    st.success("✅ Prediction completed.")
    st.write(f"**Input Rainfall:** {rainfall_input:.2f} mm")
    st.write(f"**Predicted Condition:** {pred}")
