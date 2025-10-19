# -*- coding: utf-8 -*-
"""
Streamlit App: Flood/Drought Classification + Rainfall Forecasting (XGBoost)
"""
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import SMOTE
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(page_title="Rainfall Prediction App", layout="wide")

# =======================
# Load Datasets
# =======================
@st.cache_data
def load_data():
    # Monthly data for classification
    df_class = pd.read_csv("rainfallpred.csv")
    # Annual data for forecasting
    df_forecast = pd.read_csv("Rainfall_Data_LL.csv")
    df_forecast.columns = df_forecast.columns.str.strip().str.upper()
    return df_class, df_forecast

df_class, df_forecast = load_data()

# =======================
# Sidebar: Task Selection
# =======================
task = st.sidebar.selectbox("Choose Task:", ["Flood/Drought Classification", "Rainfall Forecasting"])

# =======================
# Flood/Drought Classification
# =======================
if task == "Flood/Drought Classification":
    st.header("🌦️ Flood/Drought Classification")
    
    months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    df_class['Total'] = df_class[months].sum(axis=1)
    df_class['region_mean'] = df_class.groupby('SUBDIVISION')['Total'].transform('mean')
    df_class['region_std'] = df_class.groupby('SUBDIVISION')['Total'].transform('std')
    df_class['SPI_region'] = (df_class['Total'] - df_class['region_mean']) / df_class['region_std']
    df_class['drought_pctl'] = df_class.groupby('SUBDIVISION')['Total'].transform(lambda x: x.quantile(0.05))
    df_class['flood_pctl'] = df_class.groupby('SUBDIVISION')['Total'].transform(lambda x: x.quantile(0.95))

    def classify_region(row):
        if row['SPI_region'] <= -1 or row['Total'] < row['drought_pctl']:
            return "Drought"
        elif row['SPI_region'] >= 1.5 or row['Total'] > row['flood_pctl']:
            return "Flood"
        else:
            return "Normal"

    df_class['Label'] = df_class.apply(classify_region, axis=1)

    # Feature engineering
    df_class['Mean_Rain'] = df_class[months].mean(axis=1)
    df_class['Std_Rain'] = df_class[months].std(axis=1)
    df_class['CoeffVar'] = df_class['Std_Rain'] / (df_class['Mean_Rain'] + 1e-6)
    df_class['Dry_Months'] = (df_class[months] < df_class[months].mean(axis=1).mean()).sum(axis=1)
    df_class['Wet_Months'] = (df_class[months] > df_class[months].mean(axis=1).mean()).sum(axis=1)
    df_class['Max_Month'] = df_class[months].idxmax(axis=1).apply(lambda x: months.index(x) + 1)
    df_class['Prev_Total'] = df_class.groupby('SUBDIVISION')['Total'].shift(1)
    df_class['Diff_Total'] = df_class['Total'] - df_class['Prev_Total']
    df_class['Prev_SPI'] = df_class.groupby('SUBDIVISION')['SPI_region'].shift(1)
    df_class.fillna(0, inplace=True)

    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    region_encoded = pd.DataFrame(enc.fit_transform(df_class[['SUBDIVISION']]), columns=enc.get_feature_names_out(['SUBDIVISION']))
    df_class = pd.concat([df_class, region_encoded], axis=1)

    feature_cols = (months + ['Mean_Rain','Std_Rain','CoeffVar','Dry_Months','Wet_Months','Max_Month','Diff_Total','Prev_SPI'] + list(region_encoded.columns))
    X = df_class[feature_cols]
    y = df_class['Label']

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_res)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

    clf = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05,
                        subsample=0.9, colsample_bytree=0.9, gamma=0.1,
                        objective='multi:softprob', num_class=3, random_state=42)
    clf.fit(X_train, y_train)

    st.subheader("Enter Monthly Rainfall Data for Prediction")
    region_input = st.selectbox("Select Region:", sorted(df_class['SUBDIVISION'].unique()))
    year_input = st.number_input("Enter Year:", min_value=int(df_class['YEAR'].min()), max_value=int(df_class['YEAR'].max())+10, value=int(df_class['YEAR'].max())+1)
    monthly_input = []
    for m in months:
        val = st.number_input(f"{m} rainfall (mm)", value=0, step=1)
        monthly_input.append(val)

    if st.button("Predict Event"):
        manual_df = pd.DataFrame([monthly_input], columns=months)
        manual_df['Mean_Rain'] = manual_df[months].mean(axis=1)
        manual_df['Std_Rain'] = manual_df[months].std(axis=1)
        manual_df['CoeffVar'] = manual_df['Std_Rain'] / (manual_df['Mean_Rain'] + 1e-6)
        manual_df['Dry_Months'] = (manual_df[months] < manual_df[months].mean(axis=1).mean()).sum(axis=1)
        manual_df['Wet_Months'] = (manual_df[months] > manual_df[months].mean(axis=1).mean()).sum(axis=1)
        manual_df['Max_Month'] = manual_df[months].idxmax(axis=1).apply(lambda x: months.index(x) + 1)
        manual_df['Diff_Total'] = 0
        manual_df['Prev_SPI'] = 0

        region_row = pd.DataFrame(np.zeros((1, len(region_encoded.columns))), columns=region_encoded.columns)
        if f"SUBDIVISION_{region_input}" in region_row.columns:
            region_row[f"SUBDIVISION_{region_input}"] = 1

        manual_features = pd.concat([manual_df, region_row], axis=1)[feature_cols]
        pred_label = le.inverse_transform(np.argmax(clf.predict_proba(manual_features), axis=1))[0]

        st.success(f"🌦️ Predicted Event for {region_input} in {year_input}: {pred_label}")

# =======================
# Rainfall Forecasting (XGBoost)
# =======================
else:
    st.header("🌦️ Rainfall Forecasting (XGBoost Regression)")

    region_input = st.text_input("Enter Region Name (Exact or Partial)", "")
    year_input = st.number_input("Enter Year to Forecast", 
                                 min_value=int(df_forecast['YEAR'].min()), 
                                 max_value=int(df_forecast['YEAR'].max())+20, 
                                 value=int(df_forecast['YEAR'].max())+1)

    if st.button("Forecast"):
        region_match = df_forecast[df_forecast['SUBDIVISION'].str.contains(region_input, case=False, na=False)]
        if region_match.empty:
            st.error("❌ Region not found!")
            st.stop()

        data = region_match.groupby("YEAR")["ANNUAL"].mean().reset_index().dropna()
        data = data.sort_values("YEAR").reset_index(drop=True)
        last_historical_year = data['YEAR'].max()

        # Feature engineering
        def create_features(data):
            df = data.copy()
            for i in range(1,8): df[f'Lag{i}'] = df['ANNUAL'].shift(i)
            for window in [2,3,5,7,10]:
                df[f'MA{window}'] = df['ANNUAL'].rolling(window).mean()
                df[f'STD{window}'] = df['ANNUAL'].rolling(window).std()
            df['Year_Norm'] = (df['YEAR'] - df['YEAR'].min())/(df['YEAR'].max()-df['YEAR'].min())
            df['Year_Squared'] = df['Year_Norm']**2
            df['Year_Cubed'] = df['Year_Norm']**3
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)
            df.fillna(0, inplace=True)
            return df

        data_features = create_features(data)
        feature_cols = [c for c in data_features.columns if c not in ['YEAR','ANNUAL']]

        X_train = data_features[feature_cols].values
        y_train = data_features['ANNUAL'].values

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        xgb_reg = XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.05,
                               subsample=0.9, colsample_bytree=0.9, gamma=0.1,
                               random_state=42)
        xgb_reg.fit(X_train_scaled, y_train)

        prev_predictions = data['ANNUAL'].tolist()
        forecast_years, predicted_rainfall, event_labels = [], [], []

        drought_pctl = np.percentile(data['ANNUAL'],5)
        flood_pctl = np.percentile(data['ANNUAL'],95)

        for yr in range(last_historical_year+1, year_input+1):
            last_row = data_features.iloc[-1].copy()
            future_row = last_row.copy()
            future_row['YEAR'] = yr
            for i in range(1,8):
                future_row[f'Lag{i}'] = prev_predictions[-i] if i <= len(prev_predictions) else prev_predictions[0]
            for window in [2,3,5,7,10]:
                recent_values = prev_predictions[-window:] if len(prev_predictions) >= window else prev_predictions
                future_row[f'MA{window}'] = np.mean(recent_values)
                future_row[f'STD{window}'] = np.std(recent_values)
            future_row['Year_Norm'] = (yr - data['YEAR'].min())/(data['YEAR'].max()-data['YEAR'].min())
            future_row['Year_Squared'] = future_row['Year_Norm']**2
            future_row['Year_Cubed'] = future_row['Year_Norm']**3

            X_future_scaled = scaler.transform(pd.DataFrame([future_row])[feature_cols].values)
            pred = float(xgb_reg.predict(X_future_scaled)[0])
            prev_predictions.append(pred)

            if pred < drought_pctl:
                event = "Drought"
            elif pred > flood_pctl:
                event = "Flood"
            else:
                event = "Normal"

            forecast_years.append(yr)
            predicted_rainfall.append(pred)
            event_labels.append(event)

        forecast_df = pd.DataFrame({
            'Year': forecast_years,
            'Predicted_Rainfall_mm': predicted_rainfall,
            'Event': event_labels
        })

        st.subheader(f"🌦️ Forecast & Flood/Drought Classification for {region_input.title()}")
        st.dataframe(forecast_df)
        st.line_chart(forecast_df.set_index('Year')['Predicted
