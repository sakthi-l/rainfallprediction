# -*- coding: utf-8 -*-
"""
Streamlit App: Flood/Drought Classification + Hybrid Rainfall Forecasting
Merged by ChatGPT — includes:
- Flood/Drought classification (XGBoost classifier)
- Rainfall forecasting (Hybrid: Lasso for historical-year prediction, XGB + historical-scaling + adaptive-weights for future years)

Files expected in the same folder:
- rainfallpred.csv         (monthly dataset used for classification)
- Rainfall_Data_LL.csv     (annual dataset used for forecasting)

Run with:
    streamlit run rainfall_app_updated.py

"""

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LassoCV
from xgboost import XGBClassifier, XGBRegressor
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Rainfall Prediction App (Hybrid)", layout="wide")

# =======================
# Load Datasets
# =======================
@st.cache_data
def load_data():
    df_class = pd.read_csv("rainfallpred.csv")
    df_forecast = pd.read_csv("Rainfall_Data_LL.csv")
    df_forecast.columns = df_forecast.columns.str.strip().str.upper()
    return df_class, df_forecast

try:
    df_class, df_forecast = load_data()
except Exception as e:
    st.error(f"Error loading data files: {e}")
    st.stop()

# =======================
# Sidebar: Task Selection
# =======================
task = st.sidebar.selectbox("Choose Task:", ["Flood/Drought Classification", "Rainfall Forecasting (Hybrid)"])

# =======================
# Flood/Drought Classification
# =======================
if task == "Flood/Drought Classification":
    st.header("🌦️ Flood/Drought Classification")

    months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    # Safeguard: ensure months exist in df_class
    missing_months = [m for m in months if m not in df_class.columns]
    if missing_months:
        st.error(f"Monthly columns missing in 'rainfallpred.csv': {missing_months}")
        st.stop()

    df_class = df_class.copy()
    df_class['Total'] = df_class[months].sum(axis=1)
    df_class['region_mean'] = df_class.groupby('SUBDIVISION')['Total'].transform('mean')
    df_class['region_std'] = df_class.groupby('SUBDIVISION')['Total'].transform('std').replace(0, 1)
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
    df_class = pd.concat([df_class.reset_index(drop=True), region_encoded.reset_index(drop=True)], axis=1)

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
        colname = f"SUBDIVISION_{region_input}"
        if colname in region_row.columns:
            region_row[colname] = 1

        manual_features = pd.concat([manual_df.reset_index(drop=True), region_row.reset_index(drop=True)], axis=1)[feature_cols]
        pred_label = le.inverse_transform(np.argmax(clf.predict_proba(manual_features), axis=1))[0]

        st.success(f"🌦️ Predicted Event for {region_input} in {year_input}: {pred_label}")

# =======================
# Rainfall Forecasting (Hybrid Ensemble)
# =======================
else:
    st.header("🌦️ Rainfall Forecasting — Hybrid Ensemble + Scaling")

    # Region selector from the forecast dataset (cleaned)
    available_regions = sorted(df_forecast['SUBDIVISION'].dropna().unique())
    st.subheader("Region & Year Selection")
    region_input = st.selectbox("Select Region:", available_regions)

    min_year = int(df_forecast['YEAR'].min())
    max_year = int(df_forecast['YEAR'].max())
    year_input = st.number_input("Enter Year to Forecast", min_value=min_year, max_value=max_year+20, value=max_year+1)

    if st.button("Forecast"):
        with st.spinner("Running hybrid forecast — this may take a few seconds..."):
            try:
                region_match = df_forecast[df_forecast['SUBDIVISION'].str.contains(region_input, case=False, na=False)]
                if region_match.empty:
                    st.error("❌ Region not found!")
                    st.stop()

                data = region_match.groupby("YEAR")["ANNUAL"].mean().reset_index().dropna()
                data = data.sort_values("YEAR").reset_index(drop=True)
                last_historical_year = int(data['YEAR'].max())

                # Feature engineering function (adapted from provided script)
                def create_features(data, target_year=None):
                    df = data.copy()
                    for i in range(1, 8):
                        df[f'Lag{i}'] = df['ANNUAL'].shift(i)
                    for window in [2, 3, 5, 7, 10]:
                        df[f'MA{window}'] = df['ANNUAL'].rolling(window).mean()
                        df[f'STD{window}'] = df['ANNUAL'].rolling(window).std()
                        df[f'Min{window}'] = df['ANNUAL'].rolling(window).min()
                        df[f'Max{window}'] = df['ANNUAL'].rolling(window).max()
                        df[f'Range{window}'] = df[f'Max{window}'] - df[f'Min{window}']
                    for span in [2, 3, 5, 7]:
                        df[f'EMA{span}'] = df['ANNUAL'].ewm(span=span, adjust=False).mean()
                    df['Year_Norm'] = (df['YEAR'] - df['YEAR'].min()) / (df['YEAR'].max() - df['YEAR'].min())
                    df['Year_Squared'] = df['Year_Norm'] ** 2
                    df['Year_Cubed'] = df['Year_Norm'] ** 3
                    for cycle in [5, 7, 11]:
                        df[f'Cycle{cycle}_Sin'] = np.sin(2 * np.pi * df['YEAR'] / cycle)
                        df[f'Cycle{cycle}_Cos'] = np.cos(2 * np.pi * df['YEAR'] / cycle)
                    df['Rate_Change'] = df['ANNUAL'].pct_change()
                    df['Rate_Change_2'] = df['ANNUAL'].pct_change(periods=2)
                    df['Momentum_3'] = df['ANNUAL'] - df['ANNUAL'].shift(3)
                    df['Momentum_5'] = df['ANNUAL'] - df['ANNUAL'].shift(5)
                    df['Volatility_3'] = df['ANNUAL'].rolling(3).std() / df['ANNUAL'].rolling(3).mean()
                    df['Volatility_5'] = df['ANNUAL'].rolling(5).std() / df['ANNUAL'].rolling(5).mean()
                    df['Lag1_x_MA3'] = df['ANNUAL'].shift(1) * df['MA3']
                    df['Lag1_x_Year'] = df['ANNUAL'].shift(1) * df['Year_Norm']

                    df = df.reset_index(drop=True)

                    # If a future target year is requested, append a synthetic future row with features
                    if target_year and target_year > df['YEAR'].max():
                        last_row = df.iloc[-1].copy()
                        future_row = pd.Series(dtype=float)
                        future_row['YEAR'] = target_year
                        # Lags: use last known annual value and existing lag chain
                        for i in range(1, 8):
                            if i == 1:
                                future_row[f'Lag{i}'] = last_row['ANNUAL']
                            else:
                                prev_lag_col = f'Lag{i-1}'
                                future_row[f'Lag{i}'] = last_row.get(prev_lag_col, last_row['ANNUAL'])
                        recent_values = df['ANNUAL'].tail(10).values
                        for window in [2, 3, 5, 7, 10]:
                            if len(recent_values) >= window:
                                future_row[f'MA{window}'] = np.mean(recent_values[-window:])
                                future_row[f'STD{window}'] = np.std(recent_values[-window:])
                                future_row[f'Min{window}'] = np.min(recent_values[-window:])
                                future_row[f'Max{window}'] = np.max(recent_values[-window:])
                                future_row[f'Range{window}'] = future_row[f'Max{window}'] - future_row[f'Min{window}']
                            else:
                                future_row[f'MA{window}'] = last_row.get(f'MA{window}', last_row['ANNUAL'])
                                future_row[f'STD{window}'] = last_row.get(f'STD{window}', 0.0)
                                future_row[f'Min{window}'] = last_row.get(f'Min{window}', last_row['ANNUAL'])
                                future_row[f'Max{window}'] = last_row.get(f'Max{window}', last_row['ANNUAL'])
                                future_row[f'Range{window}'] = future_row[f'Max{window}'] - future_row[f'Min{window}']
                        for span in [2, 3, 5, 7]:
                            future_row[f'EMA{span}'] = df[f'EMA{span}'].iloc[-1] if f'EMA{span}' in df.columns else last_row['ANNUAL']
                        future_row['Year_Norm'] = (target_year - df['YEAR'].min()) / (df['YEAR'].max() - df['YEAR'].min())
                        future_row['Year_Squared'] = future_row['Year_Norm'] ** 2
                        future_row['Year_Cubed'] = future_row['Year_Norm'] ** 3
                        for cycle in [5, 7, 11]:
                            future_row[f'Cycle{cycle}_Sin'] = np.sin(2 * np.pi * target_year / cycle)
                            future_row[f'Cycle{cycle}_Cos'] = np.cos(2 * np.pi * target_year / cycle)
                        # Rate & momentum
                        future_row['Rate_Change'] = (future_row['Lag1'] - future_row.get('Lag2', future_row['Lag1'])) / future_row.get('Lag2', future_row['Lag1']) if future_row.get('Lag2', 0) != 0 else 0
                        future_row['Rate_Change_2'] = (future_row['Lag1'] - future_row.get('Lag3', future_row['Lag1'])) / future_row.get('Lag3', future_row['Lag1']) if future_row.get('Lag3', 0) != 0 else 0
                        future_row['Momentum_3'] = future_row['Lag1'] - future_row.get('Lag4', future_row['Lag1'])
                        future_row['Momentum_5'] = future_row['Lag1'] - future_row.get('Lag6', future_row['Lag1'])
                        future_row['Volatility_3'] = df['Volatility_3'].iloc[-1] if 'Volatility_3' in df.columns else 0
                        future_row['Volatility_5'] = df['Volatility_5'].iloc[-1] if 'Volatility_5' in df.columns else 0
                        future_row['Lag1_x_MA3'] = future_row['Lag1'] * future_row.get('MA3', future_row['Lag1'])
                        future_row['Lag1_x_Year'] = future_row['Lag1'] * future_row['Year_Norm']

                        df = pd.concat([df, pd.DataFrame([future_row])], ignore_index=True)

                    # drop rows that still contain NA in ANNUAL-derived features (keep original ANNUAL rows intact)
                    df = df.dropna(subset=[c for c in df.columns if c.startswith('Lag') or c.startswith('MA') or c.startswith('EMA')], how='any').reset_index(drop=True)
                    return df

                data_features = create_features(data, target_year=year_input if year_input > last_historical_year else None)
                feature_cols = [c for c in data_features.columns if c not in ['YEAR', 'ANNUAL']]

                # If forecasting a historical year (present in data) -> use LassoCV trained with TimeSeries split
                if year_input in data['YEAR'].values:
                    train_data = data_features[data_features['YEAR'] < year_input]
                    test_data = data_features[data_features['YEAR'] == year_input]

                    if train_data.empty or test_data.empty:
                        st.error("Not enough historical rows to train/test for the selected year")
                        st.stop()

                    X_train = train_data[feature_cols].values
                    y_train = train_data['ANNUAL'].values
                    X_test = test_data[feature_cols].values

                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    tscv = TimeSeriesSplit(n_splits=max(2, min(5, len(X_train)//2)))
                    lasso_cv = LassoCV(cv=tscv, random_state=42, max_iter=10000, n_alphas=100, alphas=np.logspace(-4, 1, 100))
                    lasso_cv.fit(X_train_scaled, y_train)
                    predicted = float(lasso_cv.predict(X_test_scaled)[0])
                    predicted_series = ({'Year': int(year_input), 'Predicted_Rainfall_mm': predicted, 'Method': 'Lasso_historical'})

                else:
                    # Future year -> XGBoost scaled + adaptive ensemble
                    train_data = data_features[data_features['YEAR'] <= last_historical_year]
                    # the future row is the last row in data_features when target_year appended
                    test_data = data_features[data_features['YEAR'] == year_input]

                    X_train = train_data[feature_cols].values
                    y_train = train_data['ANNUAL'].values
                    X_test = test_data[feature_cols].values

                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)

                    # Historical mean & std for scaling (use last N years)
                    N = min(10, len(y_train))
                    hist_mean = np.mean(y_train[-N:])
                    hist_std = np.std(y_train[-N:]) if np.std(y_train[-N:]) > 0 else 1.0

                    # Scale target as deviation
                    y_train_scaled = (y_train - hist_mean) / hist_std

                    xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=3, subsample=0.8, colsample_bytree=0.8, random_state=42)
                    xgb_model.fit(X_train_scaled, y_train_scaled)

                    pred_scaled = float(xgb_model.predict(X_test_scaled)[0])
                    predicted_xgb = pred_scaled * hist_std + hist_mean

                    # Adaptive ensemble weights per region
                    high_variability_regions = ['Kerala', 'Assam & Meghalaya', 'West Bengal', 'Orissa']
                    if region_input.title() in high_variability_regions:
                        w_xgb = 0.75
                        w_hist = 0.25
                    else:
                        w_xgb = 0.6
                        w_hist = 0.4

                    predicted = w_xgb * predicted_xgb + w_hist * hist_mean
                    predicted_series = ({'Year': int(year_input), 'Predicted_Rainfall_mm': predicted, 'Method': 'XGB_hybrid'})

                # Prepare output dataframe for display
                # include last historical year value for context
                context_rows = []
                last_few = data.tail(5)[['YEAR', 'ANNUAL']].rename(columns={'YEAR': 'Year', 'ANNUAL': 'Observed_Rainfall_mm'})
                forecast_row = pd.DataFrame([predicted_series]).rename(columns={'Year': 'Year', 'Predicted_Rainfall_mm': 'Predicted_Rainfall_mm'})

                # Event classification thresholds based on historical percentiles
                drought_pctl = np.percentile(data['ANNUAL'], 5)
                flood_pctl = np.percentile(data['ANNUAL'], 95)
                event = 'Normal'
                if predicted < drought_pctl:
                    event = 'Drought'
                elif predicted > flood_pctl:
                    event = 'Flood'

                forecast_df = pd.DataFrame({
                    'Year': [int(year_input)],
                    'Predicted_Rainfall_mm': [predicted],
                    'Event': [event]
                })

                st.subheader(f"🌦️ Forecast for {region_input} — Year {year_input}")
                st.write("**Method used:**", predicted_series.get('Method', 'Hybrid'))
                # show last few historical values
                st.write("Recent observed annual rainfall (last 5 years):")
                st.dataframe(last_few.reset_index(drop=True))
                st.write("Forecast:")
                st.dataframe(forecast_df)

                # Line chart: historical + predicted
                plot_df = pd.DataFrame({
                    'Year': list(data['YEAR']) + [int(year_input)],
                    'Rainfall_mm': list(data['ANNUAL']) + [predicted]
                })
                plot_df = plot_df.sort_values('Year')
                st.line_chart(plot_df.set_index('Year')['Rainfall_mm'])

                # Show percentile thresholds
                st.info(f"Drought threshold (5th pct): {drought_pctl:.2f} mm — Flood threshold (95th pct): {flood_pctl:.2f} mm")

                # If actual exists, show error
                actual_val = None
                if year_input in data['YEAR'].values:
                    actual_val = float(data.loc[data['YEAR'] == year_input, 'ANNUAL'].values[0])

                if actual_val is not None:
                    error = abs(predicted - actual_val)
                    error_pct = (error / actual_val * 100) if actual_val != 0 else np.nan
                    st.success(f"Actual: {actual_val:.2f} mm — Predicted: {predicted:.2f} mm — Error: {error:.2f} mm ({error_pct:.2f}%)")

            except Exception as e:
                st.error(f"Forecasting failed: {e}")

st.sidebar.markdown("---")
st.sidebar.write("Data sources: rainfallpred.csv (monthly) & Rainfall_Data_LL.csv (annual). Make sure both are in the app folder.")

# End of file
