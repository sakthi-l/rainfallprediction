# ============================================================
# 🌦️ RAINFALL FORECASTING (LASSO) + FLOOD/DROUGHT CLASSIFICATION
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
# PAGE SETTINGS
# ============================================================

st.set_page_config(
    page_title="Rainfall Forecast & Flood/Drought Classification",
    page_icon="🌦️",
    layout="centered"
)

st.title("🌦️ Rainfall Forecasting & 🌊 Flood/Drought Classification")

# ============================================================
# SIDEBAR MENU
# ============================================================

st.sidebar.title("📂 Select Task")
task = st.sidebar.radio("Choose an option:", [
    "🌦 Rainfall Forecasting (LASSO)",
    "🌊 Flood/Drought Classification"
])

# ============================================================
# 🌦 RAINFALL FORECASTING (LASSO)
# ============================================================

if task == "🌦 Rainfall Forecasting (LASSO)":
    st.subheader("🌦 Rainfall Forecasting using LASSO Regularization")

    try:
        df = pd.read_csv("Rainfall_Data_LL.csv")  # 📂 ensure this file is in the repo
        st.success("✅ Dataset loaded successfully: Rainfall_Data_LL.csv")
    except Exception as e:
        st.error(f"⚠️ Error loading dataset: {e}")
        st.stop()

    # Ensure columns
    if "YEAR" not in df.columns or "ANNUAL" not in df.columns:
        st.error("❌ Dataset must contain columns: YEAR, ANNUAL")
        st.stop()

    df = df.dropna().reset_index(drop=True)

    X = np.array(df.index).reshape(-1, 1)
    y = df["ANNUAL"].values

    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    tscv = TimeSeriesSplit(n_splits=5)
    model = LassoCV(cv=tscv)
    model.fit(X_scaled, y)

    # Forecast next 10 years
    future_years = np.arange(df["YEAR"].max() + 1, df["YEAR"].max() + 11)
    X_future = np.arange(len(df), len(df) + 10).reshape(-1, 1)
    X_future_scaled = scaler.transform(X_future)
    y_pred = model.predict(X_future_scaled)

    forecast_df = pd.DataFrame({
        "YEAR": future_years,
        "Predicted_Rainfall(mm)": np.round(y_pred, 2)
    })

    st.dataframe(forecast_df, width=600)
    st.success("✅ Forecast completed successfully.")

# ============================================================
# 🌊 FLOOD/DROUGHT CLASSIFICATION
# ============================================================

elif task == "🌊 Flood/Drought Classification":
    st.subheader("🌊 Flood/Drought Classification using XGBoost")

    try:
        df = pd.read_csv("rainfallpred.csv")  # 📂 ensure this file is in the repo
        st.success("✅ Dataset loaded successfully: rainfallpred.csv")
    except Exception as e:
        st.error(f"⚠️ Error loading dataset: {e}")
        st.stop()

    # Ensure columns
    if "RAINFALL" not in df.columns or "LABEL" not in df.columns:
        st.error("❌ Dataset must contain columns: RAINFALL, LABEL")
        st.stop()

    df = df.dropna().reset_index(drop=True)
    X = df[["RAINFALL"]]
    y = df["LABEL"]

    clf = XGBClassifier()
    clf.fit(X, y)

    rainfall_input = st.number_input("Enter rainfall (mm) to classify:", 0, 3000, 1200)
    pred = clf.predict([[rainfall_input]])[0]

    if pred == 1:
        st.success("🌧️ Prediction: FLOOD LIKELY")
    else:
        st.warning("☀️ Prediction: DROUGHT LIKELY")

    st.info("Model trained on your dataset (rainfallpred.csv)")
