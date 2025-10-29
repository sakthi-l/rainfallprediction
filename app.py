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

# ==============================
# 🌊 Flood/Drought Classification
# ==============================
st.header("🌊 Flood/Drought Prediction (XGBoost Model)")

try:
    df = pd.read_csv("rainfallpred.csv")
    st.success("✅ Dataset loaded successfully: rainfallpred.csv")

    # Select numeric columns only
    df = df.select_dtypes(include=[np.number])

    # Handle missing values
    df = df.fillna(df.mean())

    # Ensure ANNUAL exists
    if 'ANNUAL' not in df.columns:
        st.error("❌ 'ANNUAL' column not found.")
    else:
        # Auto-generate labels if missing
        if 'LABEL' not in df.columns:
            st.info("💡 Labels auto-generated based on rainfall intensity thresholds.")
            mean_rain = df['ANNUAL'].mean()
            std_rain = df['ANNUAL'].std()
            conditions = [
                (df['ANNUAL'] < mean_rain - std_rain),
                (df['ANNUAL'] > mean_rain + std_rain)
            ]
            choices = [0, 2]  # 0 = drought, 2 = flood
            df['LABEL'] = np.select(conditions, choices, default=1)  # 1 = normal

        # Split features and labels
        X = df.drop(columns=['LABEL'])
        y = df['LABEL'].astype(int)

        # Train model safely
        from xgboost import XGBClassifier
        clf = XGBClassifier(eval_metric='mlogloss', use_label_encoder=False)
        clf.fit(X, y)

        # Prediction UI
        st.subheader("🔮 Predict Flood/Drought Status")
        region = st.text_input("Enter Region Name:")
        year = st.number_input("Enter Year:", min_value=int(df['YEAR'].min()), max_value=int(df['YEAR'].max()))

        if st.button("Predict"):
            # Match row for region/year if exists
            row = df[(df['YEAR'] == year)]
            if not row.empty:
                X_input = row.drop(columns=['LABEL'])
                pred = clf.predict(X_input)[0]
                label_map = {0: "🌵 Drought", 1: "🌤 Normal", 2: "🌊 Flood"}
                st.success(f"✅ Predicted Condition for {region or 'Region'} ({year}): {label_map[pred]}")
            else:
                st.warning("⚠️ No matching year found in dataset.")

except Exception as e:
    st.error(f"⚠️ Error: {e}")
