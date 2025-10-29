import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from prophet import Prophet
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# ============================================================
# APP HEADER
# ============================================================
st.set_page_config(page_title="Rainfall & Flood/Drought Predictor", layout="centered")

st.title("🌦 Rainfall Forecasting & Flood/Drought Classification Dashboard")
st.markdown("### Upload your dataset and choose a task")

# ============================================================
# FILE UPLOAD
# ============================================================
uploaded_file = st.file_uploader("📂 Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Data successfully uploaded!")
    st.write("### Preview of Dataset:")
    st.dataframe(df.head())
else:
    st.warning("⚠️ Please upload a dataset to continue.")
    st.stop()

# ============================================================
# SELECT TASK
# ============================================================
task = st.radio(
    "Choose Task:",
    ["🌧 Flood/Drought Classification", "🌦 Rainfall Forecasting (Hybrid)"]
)

# ============================================================
# 🌧 FLOOD/DROUGHT CLASSIFICATION
# ============================================================
if task == "🌧 Flood/Drought Classification":
    if 'LABEL' not in df.columns:
        st.error("Dataset must contain a 'LABEL' column for classification.")
        st.stop()

    label_encoder = LabelEncoder()
    df['LABEL'] = label_encoder.fit_transform(df['LABEL'])

    X = df.drop(columns=['LABEL'], errors='ignore')
    y = df['LABEL']

    # Handle categorical columns
    X = pd.get_dummies(X, drop_first=True)

    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)

    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42
    )

    model = XGBClassifier(random_state=42, n_estimators=200, learning_rate=0.1)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    st.subheader("✅ Classification Completed")
    st.write("**Predicted Classes (sample):**")
    st.write(pd.DataFrame({'Actual': y_test[:10].values, 'Predicted': preds[:10]}))

    report = classification_report(y_test, preds, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    st.write("**Classification Summary:**")
    st.dataframe(report_df.round(3))

# ============================================================
# 🌦 RAINFALL FORECASTING (HYBRID)
# ============================================================
elif task == "🌦 Rainfall Forecasting (Hybrid)":
    if 'YEAR' not in df.columns or 'ANNUAL' not in df.columns:
        st.error("Dataset must contain 'YEAR' and 'ANNUAL' columns for rainfall forecasting.")
        st.stop()

    data = df[['YEAR', 'ANNUAL']].dropna()
    data = data.sort_values(by='YEAR')

    # Prophet expects columns: ds, y
    prophet_df = pd.DataFrame({
        'ds': pd.to_datetime(data['YEAR'], format='%Y'),
        'y': data['ANNUAL']
    })

    model_prophet = Prophet(yearly_seasonality=True, daily_seasonality=False)
    model_prophet.fit(prophet_df)

    # Create future dataframe till 2035
    future = model_prophet.make_future_dataframe(periods=(2035 - data['YEAR'].max()), freq='Y')
    forecast_prophet = model_prophet.predict(future)

    # Hybrid: Combine Prophet + XGBoost
    forecast = forecast_prophet[['ds', 'yhat']].copy()
    forecast.rename(columns={'ds': 'YEAR'}, inplace=True)
    forecast['YEAR'] = forecast['YEAR'].dt.year

    merged = pd.merge(data, forecast, on='YEAR', how='right')
    merged['FINAL_PRED'] = merged['yhat']

    # XGBoost fine-tuning
    xgb = XGBRegressor(random_state=42, n_estimators=300, learning_rate=0.05)
    scaler = RobustScaler()

    valid_data = merged.dropna()
    X = scaler.fit_transform(valid_data[['YEAR']])
    y = valid_data['FINAL_PRED']

    xgb.fit(X, y)
    future_years = np.arange(data['YEAR'].max() + 1, 2036).reshape(-1, 1)
    future_scaled = scaler.transform(future_years)
    future_preds = xgb.predict(future_scaled)

    forecast_table = pd.DataFrame({
        'YEAR': future_years.flatten(),
        'Predicted_Rainfall_mm': np.round(future_preds, 2)
    })

    st.subheader("✅ Rainfall Forecast (Hybrid Prophet + XGBoost)")
    st.write("**Forecasted Rainfall up to 2035:**")
    st.dataframe(forecast_table)

    st.download_button(
        label="⬇️ Download Forecast Data",
        data=forecast_table.to_csv(index=False),
        file_name="Rainfall_Forecast_2035.csv",
        mime="text/csv"
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.caption("Developed for Rainfall & Flood/Drought Prediction — Clean No-Plot Version")
