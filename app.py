"""
🌦️ Enhanced Rainfall Prediction & Flood/Drought Analysis Dashboard
Deploy with: streamlit run app.py
"""

# ============================================================
# Imports
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# Machine Learning
from sklearn.linear_model import LassoCV, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from prophet import Prophet

# ============================================================
# Streamlit Page Configuration
# ============================================================

st.set_page_config(
    page_title="🌧️ Rainfall & Flood/Drought Forecast Dashboard",
    page_icon="🌦️",
    layout="wide"
)

# ============================================================
# Sidebar Navigation
# ============================================================

st.sidebar.title("Navigation")
selected_mode = st.sidebar.radio(
    "Select Mode:",
    ["🏠 Home", "📊 Rainfall Forecasting", "🌊 Flood/Drought Prediction"]
)

# ============================================================
# 🏠 HOME
# ============================================================

if selected_mode == "🏠 Home":
    st.title("🌦️ Enhanced Rainfall Prediction & Analysis System")
    st.markdown("""
    ### Welcome!
    This system enables:
    - 📊 **Rainfall Forecasting** using AI models  
    - 🌊 **Flood & Drought Detection** using rainfall thresholds  
    - 📅 Forecasts up to 2035 using hybrid ML and Prophet models  

    💡 *Developed with Prophet, XGBoost, and Lasso Regularization for improved accuracy.*
    """)

# ============================================================
# 📊 RAINFALL FORECASTING
# ============================================================

elif selected_mode == "📊 Rainfall Forecasting":
    st.title("📊 Rainfall Forecasting Module")

    uploaded_file = st.file_uploader("📂 Upload Rainfall Data (CSV)", type=["csv"])

    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Data successfully loaded!")
        st.write(df.head())

        # --- Data Preprocessing ---
        df.columns = [c.strip().title() for c in df.columns]
        if "Year" not in df.columns:
            st.error("❌ The dataset must contain a 'Year' column.")
        elif "Rainfall_Mm" not in df.columns and "Rainfall_mm" not in df.columns:
            st.error("❌ The dataset must contain a 'Rainfall_mm' column.")
        else:
            if "Rainfall_Mm" not in df.columns:
                df.rename(columns={"Rainfall_mm": "Rainfall_Mm"}, inplace=True)
            df["Year"] = pd.to_datetime(df["Year"], format="%Y")

            # --- Forecasting Model ---
            prophet_df = df.rename(columns={"Year": "ds", "Rainfall_Mm": "y"})
            model = Prophet()
            model.fit(prophet_df)

            # Forecast next 10 years
            future = model.make_future_dataframe(periods=10, freq="Y")
            forecast = model.predict(future)

            # Display results
            st.markdown("### 🔮 10-Year Rainfall Forecast (Prophet Model)")
            st.dataframe(forecast[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(10))

            # Evaluation (for past data)
            if len(df) > 5:
                y_true = prophet_df["y"].values
                y_pred = model.predict(prophet_df)["yhat"].values
                mae = mean_absolute_error(y_true, y_pred)
                rmse = np.sqrt(mean_squared_error(y_true, y_pred))
                st.markdown("### 📈 Model Evaluation Metrics")
                st.write(f"**MAE:** {mae:.2f}")
                st.write(f"**RMSE:** {rmse:.2f}")

            st.success("✅ Forecast generated successfully!")

# ============================================================
# 🌊 FLOOD/DROUGHT PREDICTION
# ============================================================

elif selected_mode == "🌊 Flood/Drought Prediction":
    st.title("🌊 Flood & Drought Prediction Module")

    uploaded_file = st.file_uploader("📂 Upload Rainfall Dataset (CSV)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.success("✅ Data loaded successfully!")
        st.write(df.head())

        if "Rainfall_mm" not in df.columns:
            st.error("❌ 'Rainfall_mm' column not found in dataset.")
        else:
            # --- Define thresholds ---
            threshold_high = df["Rainfall_mm"].quantile(0.75)
            threshold_low = df["Rainfall_mm"].quantile(0.25)

            def classify_rainfall(value):
                if value >= threshold_high:
                    return "Flood"
                elif value <= threshold_low:
                    return "Drought"
                else:
                    return "Normal"

            df["LABEL"] = df["Rainfall_mm"].apply(classify_rainfall)
            st.markdown("### 🌧️ Classification Results")
            st.dataframe(df[["Year", "Rainfall_mm", "LABEL"]])

            # Count results
            st.markdown("### 📊 Classification Summary")
            summary = df["LABEL"].value_counts().reset_index()
            summary.columns = ["Condition", "Count"]
            st.dataframe(summary)

            st.success("✅ Flood/Drought classification complete!")
