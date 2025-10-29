"""
🌦️ Rainfall Prediction & Flood/Drought Analysis Dashboard
Optimized with caching for faster loading
"""

# ============================================================
# Imports
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error

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
app_mode = st.sidebar.radio(
    "Select Mode:",
    ["🏠 Home", "📊 Rainfall Forecasting", "🌊 Flood/Drought Prediction"]
)

# ============================================================
# 🧠 Caching Functions
# ============================================================

@st.cache_data
def load_data_from_github(url: str):
    """Load dataset from GitHub (cached)."""
    df = pd.read_csv(url)
    df.columns = [c.strip().title() for c in df.columns]
    return df

@st.cache_resource
def train_prophet(df: pd.DataFrame):
    """Train Prophet model (cached)."""
    prophet_df = df.rename(columns={"Year": "ds", "Rainfall_Mm": "y"})
    model = Prophet()
    model.fit(prophet_df)
    forecast = model.predict(prophet_df)
    df["Predicted_Rainfall"] = forecast["yhat"]
    return df, model

# ============================================================
# 🏠 HOME
# ============================================================

if app_mode == "🏠 Home":
    st.title("🌦️ Rainfall Prediction & Flood/Drought Analysis System")
    st.markdown("""
    ### Welcome!
    This app enables:
    - 📊 **Rainfall Analysis** using Prophet  
    - 🌊 **Flood & Drought Classification**  
    - ⚙️ **Fast Performance** using caching  

    💡 Improvements: Multi-model ensemble, enhanced features, better validation.
    """)

# ============================================================
# 📊 RAINFALL FORECASTING
# ============================================================

elif app_mode == "📊 Rainfall Forecasting":
    st.title("📊 Rainfall Forecasting (Historical Data Only)")

    github_url = st.text_input(
        "🌐 Enter GitHub Raw CSV URL:",
        "https://raw.githubusercontent.com/USERNAME/REPO/main/your_rainfall_data.csv"
    )

    if github_url:
        try:
            df = load_data_from_github(github_url)
            st.success("✅ Dataset loaded successfully from GitHub!")
            st.dataframe(df.head())

            # --- Validate Columns ---
            if "Year" not in df.columns:
                st.error("❌ 'Year' column missing.")
            elif "Rainfall_Mm" not in df.columns and "Rainfall_mm" not in df.columns:
                st.error("❌ 'Rainfall_mm' column missing.")
            else:
                if "Rainfall_Mm" not in df.columns:
                    df.rename(columns={"Rainfall_mm": "Rainfall_Mm"}, inplace=True)
                df["Year"] = pd.to_datetime(df["Year"], format="%Y")

                # --- Train Prophet (cached) ---
                df_with_preds, _ = train_prophet(df)

                st.markdown("### 📋 Observed vs Predicted Rainfall")
                st.dataframe(df_with_preds[["Year", "Rainfall_Mm", "Predicted_Rainfall"]])

                # --- Metrics ---
                mae = mean_absolute_error(df_with_preds["Rainfall_Mm"], df_with_preds["Predicted_Rainfall"])
                rmse = np.sqrt(mean_squared_error(df_with_preds["Rainfall_Mm"], df_with_preds["Predicted_Rainfall"]))

                st.markdown("### 📈 Model Performance")
                st.write(f"**MAE:** {mae:.2f}")
                st.write(f"**RMSE:** {rmse:.2f}")

                st.success("✅ Historical rainfall analysis complete (no future forecast).")

        except Exception as e:
            st.error(f"❌ Failed to load dataset: {e}")

# ============================================================
# 🌊 FLOOD/DROUGHT PREDICTION
# ============================================================

elif app_mode == "🌊 Flood/Drought Prediction":
    st.title("🌊 Flood & Drought Risk Assessment")

    github_url = st.text_input(
        "🌐 Enter GitHub Raw CSV URL:",
        "https://raw.githubusercontent.com/USERNAME/REPO/main/your_rainfall_data.csv"
    )

    if github_url:
        try:
            df = load_data_from_github(github_url)
            st.success("✅ Dataset loaded successfully!")
            st.dataframe(df.head())

            if "Rainfall_mm" not in df.columns:
                st.error("❌ 'Rainfall_mm' column not found.")
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

                summary = df["LABEL"].value_counts().reset_index()
                summary.columns = ["Condition", "Count"]
                st.markdown("### 📊 Summary")
                st.dataframe(summary)

                st.success("✅ Flood/Drought classification complete!")

        except Exception as e:
            st.error(f"❌ Error loading dataset: {e}")

# ============================================================
# END OF APP
# ============================================================

st.markdown("---")
st.caption("🌧️ Developed by Sakthi | Optimized Rainfall & Disaster Forecasting System")
