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

# ML models
from sklearn.linear_model import LassoCV, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
from xgboost import XGBRegressor
from prophet import Prophet

# Plotly
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# App Configuration
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
    ### Welcome to the Rainfall Prediction Dashboard  
    This system enables:
    - 📊 **Rainfall Forecasting** using hybrid AI models  
    - 🌊 **Flood & Drought Detection** based on rainfall patterns  
    - 📅 Forecasts up to 2035 using advanced machine learning models  
    
    💡 *Developed using Prophet, XGBoost, and Lasso regularization models.*
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

        # Basic preprocessing
        df.columns = [c.strip().title() for c in df.columns]
        if "Year" not in df.columns:
            st.error("❌ The dataset must contain a 'Year' column.")
        elif "Rainfall_Mm" not in df.columns:
            st.error("❌ The dataset must contain a 'Rainfall_mm' column.")
        else:
            df["Year"] = pd.to_datetime(df["Year"], format="%Y")

            st.markdown("### 🔍 Rainfall Trend Overview")
            fig = px.line(df, x="Year", y="Rainfall_Mm", title="Rainfall Trend Over Time")
            fig.update_xaxes(title="Year")
            fig.update_yaxes(title="Rainfall (mm)")
            st.plotly_chart(fig, use_container_width=True)

            # Data scaling
            scaler = RobustScaler()
            y = df["Rainfall_Mm"].values.reshape(-1, 1)
            y_scaled = scaler.fit_transform(y).flatten()

            # Prophet forecast
            prophet_df = df.rename(columns={"Year": "ds", "Rainfall_Mm": "y"})
            model = Prophet()
            model.fit(prophet_df)

            future = model.make_future_dataframe(periods=10, freq="Y")
            forecast = model.predict(future)

            st.markdown("### 🔮 10-Year Rainfall Forecast (Prophet Model)")
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(x=prophet_df["ds"], y=prophet_df["y"], mode="lines", name="Observed"))
            fig2.add_trace(go.Scatter(x=forecast["ds"], y=forecast["yhat"], mode="lines", name="Forecast"))
            fig2.update_layout(title="Rainfall Forecast (Prophet)", xaxis_title="Year", yaxis_title="Rainfall (mm)")
            st.plotly_chart(fig2, use_container_width=True)

            st.markdown("✅ Forecast generated successfully!")

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
            st.error("❌ 'Rainfall_mm' column not found in the dataset.")
        else:
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
            st.markdown("### 🌧️ Classification Summary")
            st.dataframe(df[["Year", "Rainfall_mm", "LABEL"]])

            # Plot results
            fig = px.bar(
                df,
                x="Year",
                y="Rainfall_mm",
                color="LABEL",
                title="Flood/Drought Classification by Year"
            )
            fig.update_xaxes(title="Year", tickangle=45)
            fig.update_yaxes(title="Rainfall (mm)")
            st.plotly_chart(fig, use_container_width=True)

            st.success("✅ Flood/Drought classification complete!")

