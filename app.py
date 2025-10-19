# -*- coding: utf-8 -*-
"""🌧️ Rainfall & Flood/Drought Prediction Dashboard"""

import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LassoCV
import matplotlib.pyplot as plt

# ============================================================
# APP HEADER
# ============================================================
st.set_page_config(page_title="Rainfall Forecasting & Flood/Drought Analysis", layout="wide")
st.sidebar.title("🌦️ Climate Prediction Dashboard")

task = st.sidebar.radio(
    "Choose Analysis Mode:",
    ["🌊 Flood/Drought Prediction", "🌦️ Rainfall Forecasting (Hybrid)"]
)

# ============================================================
# 🌊 FLOOD/DROUGHT CLASSIFICATION SECTION
# ============================================================
if task == "🌊 Flood/Drought Prediction":
    st.title("🌊 Flood / Drought Prediction using Rainfall Data")

    uploaded_file = st.file_uploader("Upload rainfall dataset (CSV format)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        st.write("✅ Data successfully loaded!")
        st.dataframe(df.head())

        # Encode categorical columns
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = LabelEncoder().fit_transform(df[col].astype(str))

        # Handle class imbalance with SMOTE
        X = df.drop("Event", axis=1, errors="ignore")
        y = df["Event"] if "Event" in df.columns else None

        if y is not None:
            sm = SMOTE(random_state=42)
            X_res, y_res = sm.fit_resample(X, y)

            X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.25, random_state=42)
            model = XGBClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
            model.fit(X_train, y_train)

            preds = model.predict(X_test)
            st.subheader("📊 Classification Results")
            st.text(classification_report(y_test, preds))
            st.write("Confusion Matrix:")
            st.write(confusion_matrix(y_test, preds))
        else:
            st.warning("⚠️ 'Event' column not found. Please include it for flood/drought classification.")


# ============================================================
# 🌦️ RAINFALL FORECASTING (Hybrid Ensemble + Visualization)
# ============================================================
elif task == "🌦️ Rainfall Forecasting (Hybrid)":
    st.title("🌦️ Rainfall Forecasting (Hybrid Ensemble Model)")

    uploaded_file = st.file_uploader("Upload rainfall dataset (CSV format)", type=["csv"])
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        df.columns = df.columns.str.strip().str.upper()

        region = st.selectbox("Select Region/Subdivision", sorted(df["SUBDIVISION"].unique()))
        year_input = st.number_input("Enter Year to Forecast", min_value=1901, max_value=2100, value=2025, step=1)

        if st.button("🔮 Forecast Rainfall"):
            region_data = df[df["SUBDIVISION"].str.contains(region, case=False, na=False)]
            if region_data.empty:
                st.error("❌ Region not found in dataset!")
            else:
                data = region_data.groupby("YEAR")["ANNUAL"].mean().reset_index().dropna()
                data = data.sort_values("YEAR").reset_index(drop=True)

                # ---------------- Feature Engineering ----------------
                def create_features(data, target_year=None):
                    df = data.copy()
                    for i in range(1, 8):
                        df[f'Lag{i}'] = df["ANNUAL"].shift(i)
                    for window in [2, 3, 5, 7, 10]:
                        df[f'MA{window}'] = df["ANNUAL"].rolling(window).mean()
                        df[f'STD{window}'] = df["ANNUAL"].rolling(window).std()
                    for span in [2, 3, 5, 7]:
                        df[f'EMA{span}'] = df["ANNUAL"].ewm(span=span, adjust=False).mean()
                    df['Year_Norm'] = (df['YEAR'] - df['YEAR'].min()) / (df['YEAR'].max() - df['YEAR'].min())
                    for cycle in [5, 7, 11]:
                        df[f'Cycle{cycle}_Sin'] = np.sin(2 * np.pi * df['YEAR'] / cycle)
                        df[f'Cycle{cycle}_Cos'] = np.cos(2 * np.pi * df['YEAR'] / cycle)
                    df = df.dropna().reset_index(drop=True)
                    if target_year and target_year > df['YEAR'].max():
                        last_row = df.iloc[-1].copy()
                        new_row = last_row.copy()
                        new_row['YEAR'] = target_year
                        for i in range(1, 8):
                            new_row[f'Lag{i}'] = last_row['ANNUAL'] if i == 1 else last_row[f'Lag{i-1}']
                        for window in [2, 3, 5, 7, 10]:
                            new_row[f'MA{window}'] = df['ANNUAL'].tail(window).mean()
                            new_row[f'STD{window}'] = df['ANNUAL'].tail(window).std()
                        for span in [2, 3, 5, 7]:
                            new_row[f'EMA{span}'] = df[f'EMA{span}'].iloc[-1]
                        new_row['Year_Norm'] = (target_year - df['YEAR'].min()) / (df['YEAR'].max() - df['YEAR'].min())
                        for cycle in [5, 7, 11]:
                            new_row[f'Cycle{cycle}_Sin'] = np.sin(2 * np.pi * target_year / cycle)
                            new_row[f'Cycle{cycle}_Cos'] = np.cos(2 * np.pi * target_year / cycle)
                        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
                    return df

                data_features = create_features(data, year_input if year_input > data['YEAR'].max() else None)
                feature_cols = [c for c in data_features.columns if c not in ['YEAR', 'ANNUAL']]

                if year_input in data['YEAR'].values:
                    train_data = data_features[data_features['YEAR'] < year_input]
                    test_data = data_features[data_features['YEAR'] == year_input]
                    X_train, y_train = train_data[feature_cols].values, train_data['ANNUAL'].values
                    X_test = test_data[feature_cols].values
                    scaler = RobustScaler()
                    X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)
                    tscv = TimeSeriesSplit(n_splits=min(5, len(X_train)//5))
                    model = LassoCV(cv=tscv, random_state=42, max_iter=10000)
                    model.fit(X_train_s, y_train)
                    predicted = float(model.predict(X_test_s)[0])
                else:
                    train_data = data_features[data_features['YEAR'] <= data['YEAR'].max()]
                    test_data = data_features[data_features['YEAR'] == year_input]
                    X_train, y_train = train_data[feature_cols].values, train_data['ANNUAL'].values
                    X_test = test_data[feature_cols].values
                    scaler = RobustScaler()
                    X_train_s, X_test_s = scaler.fit_transform(X_train), scaler.transform(X_test)
                    hist_mean = np.mean(y_train[-10:])
                    hist_std = np.std(y_train[-10:]) if np.std(y_train[-10:]) > 0 else 1
                    y_train_scaled = (y_train - hist_mean) / hist_std
                    xgb = XGBRegressor(
                        n_estimators=1000,
                        learning_rate=0.05,
                        max_depth=3,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        random_state=42
                    )
                    xgb.fit(X_train_s, y_train_scaled)
                    pred_scaled = float(xgb.predict(X_test_s)[0])
                    predicted_xgb = pred_scaled * hist_std + hist_mean
                    high_var_regions = ['Kerala', 'Assam & Meghalaya', 'West Bengal', 'Orissa']
                    w_xgb, w_hist = (0.75, 0.25) if region in high_var_regions else (0.6, 0.4)
                    predicted = w_xgb * predicted_xgb + w_hist * hist_mean

                actual = data.loc[data['YEAR'] == year_input, 'ANNUAL'].values[0] if year_input in data['YEAR'].values else None

                st.subheader(f"📍 {region} — Year {year_input}")
                if actual is not None:
                    st.write(f"**Actual:** {actual:.2f} mm")
                st.write(f"**Predicted:** {predicted:.2f} mm")

                if actual is not None:
                    error = abs(predicted - actual)
                    st.write(f"**Error:** {error:.2f} mm ({error/actual*100:.2f}%)")

                # ---------------- Plot Historical + Forecast ----------------
                chart_df = data[['YEAR', 'ANNUAL']].copy()
                plt.figure(figsize=(10, 5))
                plt.plot(chart_df['YEAR'], chart_df['ANNUAL'], marker='o', color='skyblue', label='Historical')
                plt.scatter(year_input, predicted, color='orange', s=120, label='Forecasted')
                plt.title(f"Rainfall Trend for {region}")
                plt.xlabel("Year")
                plt.ylabel("Rainfall (mm)")
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.5)
                st.pyplot(plt)
