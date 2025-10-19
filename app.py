# ============================================================
# 🌦️ RAINFALL FORECASTING & FLOOD/DROUGHT CLASSIFICATION APP
# Hybrid Ensemble + Historical Scaling
# ============================================================

import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import RobustScaler, LabelEncoder
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor, XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# APP CONFIGURATION
# ============================================================
st.set_page_config(page_title="Rainfall Forecasting App", layout="wide")
st.title("🌦️ Rainfall Forecasting & Flood/Drought Classification")

# ============================================================
# LOAD DATA
# ============================================================
@st.cache_data
def load_data():
    rainfall_data = pd.read_csv("Rainfall_Data_LL.csv")
    flood_data = pd.read_csv("rainfallpred.csv")
    rainfall_data.columns = rainfall_data.columns.str.strip().str.upper()
    flood_data.columns = flood_data.columns.str.strip().str.upper()
    return rainfall_data, flood_data

rainfall_data, flood_data = load_data()

# ============================================================
# SIDEBAR NAVIGATION
# ============================================================
st.sidebar.title("📂 Select Task")
task = st.sidebar.radio("Choose an option:", [
    "🌧️ Flood/Drought Classification",
    "🌦️ Rainfall Forecasting"
])

# ============================================================
# 🌧️ FLOOD/DROUGHT CLASSIFICATION
# ============================================================
if task == "🌧️ Flood/Drought Classification":
    st.header("🌧️ Flood/Drought Classification")

    df = flood_data.copy()

    if 'LABEL' in df.columns:
        X = df.drop(columns=['LABEL'])
        y = df['LABEL']
    else:
        st.error("Dataset must contain a 'LABEL' column for classification.")
        st.stop()

    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Split & Train
    from sklearn.model_selection import train_test_split
    from imblearn.over_sampling import SMOTE

    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)

    clf = XGBClassifier(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        objective='multi:softprob',
        num_class=len(le.classes_),
        random_state=42
    )
    clf.fit(X_res, y_res)

    st.subheader("📊 Model Accuracy")
    y_pred = clf.predict(X_test)
    acc = np.mean(y_pred == y_test)
    st.write(f"✅ Accuracy: **{acc*100:.2f}%**")

    # Predict user input
    st.subheader("🔍 Predict Flood or Drought")
    user_inputs = {}
    for col in X.columns:
        val = st.number_input(f"{col}", float(df[col].min()), float(df[col].max()), float(df[col].mean()))
        user_inputs[col] = val

    user_df = pd.DataFrame([user_inputs])
    pred_label = le.inverse_transform(clf.predict(user_df))[0]
    st.success(f"Predicted Event: **{pred_label}**")

# ============================================================
# 🌦️ RAINFALL FORECASTING
# ============================================================
elif task == "🌦️ Rainfall Forecasting":
    st.header("🌦️ Rainfall Forecasting (Hybrid Ensemble Model)")

    region = st.selectbox("Select Region", sorted(rainfall_data["SUBDIVISION"].unique()))
    year_input = st.number_input("Year to Forecast:", min_value=1901, max_value=2100, value=2025)

    region_data = rainfall_data[rainfall_data["SUBDIVISION"].str.contains(region, case=False, na=False)]
    if region_data.empty:
        st.error("❌ Region not found.")
        st.stop()

    data = region_data.groupby("YEAR")["ANNUAL"].mean().reset_index().dropna()
    data = data.sort_values("YEAR").reset_index(drop=True)

    # Feature engineering
    def create_features(data, target_year=None):
        df = data.copy()
        for i in range(1, 8):
            df[f'Lag{i}'] = df["ANNUAL"].shift(i)
        for window in [2,3,5,7,10]:
            df[f'MA{window}'] = df["ANNUAL"].rolling(window).mean()
            df[f'STD{window}'] = df["ANNUAL"].rolling(window).std()
            df[f'Min{window}'] = df["ANNUAL"].rolling(window).min()
            df[f'Max{window}'] = df["ANNUAL"].rolling(window).max()
        for span in [2,3,5,7]:
            df[f'EMA{span}'] = df["ANNUAL"].ewm(span=span, adjust=False).mean()
        df['Year_Norm'] = (df['YEAR'] - df['YEAR'].min()) / (df['YEAR'].max() - df['YEAR'].min())
        df['Rate_Change'] = df["ANNUAL"].pct_change()
        df = df.dropna().reset_index(drop=True)

        if target_year and target_year > df['YEAR'].max():
            last_row = df.iloc[-1].copy()
            future_row = last_row.copy()
            future_row['YEAR'] = target_year
            for i in range(1,8):
                future_row[f'Lag{i}'] = last_row['ANNUAL'] if i==1 else last_row[f'Lag{i-1}']
            for window in [2,3,5,7,10]:
                recent_vals = df['ANNUAL'].tail(window).values
                future_row[f'MA{window}'] = np.mean(recent_vals)
                future_row[f'STD{window}'] = np.std(recent_vals)
                future_row[f'Min{window}'] = np.min(recent_vals)
                future_row[f'Max{window}'] = np.max(recent_vals)
            for span in [2,3,5,7]:
                future_row[f'EMA{span}'] = df[f'EMA{span}'].iloc[-1]
            future_row['Rate_Change'] = 0
            df = pd.concat([df, pd.DataFrame([future_row])], ignore_index=True)
        return df

    data_features = create_features(data, year_input if year_input > data['YEAR'].max() else None)
    feature_cols = [c for c in data_features.columns if c not in ['YEAR','ANNUAL']]

    # Train & Predict
    if year_input in data['YEAR'].values:
        train_data = data_features[data_features['YEAR'] < year_input]
        test_data = data_features[data_features['YEAR'] == year_input]
        X_train, y_train = train_data[feature_cols].values, train_data['ANNUAL'].values
        X_test = test_data[feature_cols].values
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        lasso = LassoCV(cv=TimeSeriesSplit(n_splits=5), random_state=42, max_iter=10000)
        lasso.fit(X_train_scaled, y_train)
        predicted = float(lasso.predict(X_test_scaled)[0])
        actual = float(data.loc[data['YEAR']==year_input,'ANNUAL'].values[0])
        st.success(f"Predicted: **{predicted:.2f} mm** | Actual: **{actual:.2f} mm**")
    else:
        train_data = data_features[data_features['YEAR'] <= data['YEAR'].max()]
        test_data = data_features[data_features['YEAR']==year_input]
        X_train, y_train = train_data[feature_cols].values, train_data['ANNUAL'].values
        X_test = test_data[feature_cols].values
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        hist_mean = np.mean(y_train[-10:])
        hist_std = np.std(y_train[-10:]) if np.std(y_train[-10:])>0 else 1
        y_train_scaled = (y_train - hist_mean)/hist_std
        xgb = XGBRegressor(n_estimators=1000, learning_rate=0.05, max_depth=3, random_state=42)
        xgb.fit(X_train_scaled, y_train_scaled)
        pred_scaled = float(xgb.predict(X_test_scaled)[0])
        predicted_xgb = pred_scaled*hist_std + hist_mean
        predicted = 0.6*predicted_xgb + 0.4*hist_mean
        st.success(f"Forecasted Rainfall for {region} in {year_input}: **{predicted:.2f} mm**")

    # Plot
    st.subheader("📈 Historical Rainfall Trend")
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(data["YEAR"], data["ANNUAL"], label="Historical", marker='o')
    if year_input > data["YEAR"].max():
        ax.scatter(year_input, predicted, color='red', label="Forecast", s=100)
    ax.set_xlabel("Year")
    ax.set_ylabel("Rainfall (mm)")
    ax.legend()
    st.pyplot(fig)
