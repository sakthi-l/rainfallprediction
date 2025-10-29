"""
🌦️ Rainfall Prediction & Flood/Drought Analysis Dashboard
Loads dataset directly from GitHub
Run with: streamlit run app.py
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
selected_mode = st.sidebar.radio(
    "Select Mode:",
    ["🏠 Home", "📊 Rainfall Forecasting", "🌊 Flood/Drought Prediction"]
)

# ============================================================
# 🏠 HOME
# ============================================================

if selected_mode == "🏠 Home":
    st.title("🌦️ Rainfall Prediction & Flood/Drought Analysis System")
    st.markdown("""
    ### Welcome!
    This app performs:
    - 📊 **Rainfall Pattern Analysis** using Prophet  
    - 🌊 **Flood & Drought Classification** using rainfall thresholds  
    - 📈 Model evaluation with MAE & RMSE  

    💡 *Now fully automated with GitHub dataset loading!*
    """)

# ============================================================
# 📊 RAINFALL FORECASTING (from GitHub)
# ============================================================

elif selected_mode == "📊 Rainfall Forecasting":
    st.title("📊 Rainfall Forecasting (Historical Analysis)")

    github_url = st.text_input(
        "🌐 Enter GitHub Raw CSV URL:",
        "https://raw.githubusercontent.com/USERNAME/REPO/main/your_rainfall_data.csv"
    )

    if github_url:
        try:
            df = pd.read_csv(github_url)
            st.success("✅ Dataset successfully loaded from GitHub!")
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

                # --- Prophet Model (Historical Only) ---
                prophet_df = df.rename(columns={"Year": "ds", "Rainfall_Mm": "y"})
                model = Prophet()
                model.fit(prophet_df)
                forecast = model.predict(prophet_df)
                df["Predicted_Rainfall"] = forecast["yhat"]

                # --- Display ---
                st.markdown("### 📋 Observed vs Predicted Rainfall")
                st.dataframe(df[["Year", "Rainfall_Mm", "Predicted_Rainfall"]])

                # --- Evaluation ---
                mae = mean_absolute_error(df["Rainfall_Mm"], df["Predicted_Rainfall"])
                rmse = np.sqrt(mean_squared_error(df["Rainfall_Mm"], df["Predicted_Rainfall"]))
                st.markdown("### 📈 Model Performance")
                st.write(f"**MAE:** {mae:.2f}")
                st.write(f"**RMSE:** {rmse:.2f}")

                st.success("✅ Historical rainfall analysis complete (no future forecast).")

        except Exception as e:
            st.error(f"❌ Failed to load dataset. Please check your GitHub URL.\n\nError: {e}")

# ============================================================
# 🌊 FLOOD/DROUGHT PREDICTION (from GitHub)
# ============================================================

elif selected_mode == "🌊 Flood/Drought Prediction":
    st.title("🌊 Flood & Drought Prediction Module")

    github_url = st.text_input(
        "🌐 Enter GitHub Raw CSV URL:",
        "https://raw.githubusercontent.com/USERNAME/REPO/main/your_rainfall_data.csv"
    )

    if github_url:
        try:
            df = pd.read_csv(github_url)
            st.success("✅ Dataset successfully loaded from GitHub!")
            st.write(df.head())

            if "Rainfall_mm" not in df.columns:
                st.error("❌ 'Rainfall_mm' column not found in dataset.")
            else:
                # --- Thresholds ---
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

                st.markdown("### 🌧️ Flood/Drought Classification Results")
                st.dataframe(df[["Year", "Rainfall_mm", "LABEL"]])

                summary = df["LABEL"].value_counts().reset_index()
                summary.columns = ["Condition", "Count"]
                st.markdown("### 📊 Summary")
                st.dataframe(summary)

                st.success("✅ Flood/Drought classification complete!")

        except Exception as e:
            st.error(f"❌ Failed to load dataset from GitHub.\n\nError: {e}")

# ============================================================
# END
# ============================================================

st.markdown("---")
st.caption("🌧️ Developed by Sakthi | AI-driven Rainfall & Disaster Forecasting System")
