

import streamlit as st
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import LassoCV, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import RobustScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from xgboost import XGBRegressor, XGBClassifier
from imblearn.over_sampling import SMOTE

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="Rainfall Prediction System", page_icon="🌦️", layout="wide")

# ============================================================
# DATA LOADERS & FEATURE HELPERS (RAINFALL FORECAST)
# ============================================================
@st.cache_data
def load_rainfall_data():
    try:
        df = pd.read_csv("Rainfall_Data_LL.csv")
        df.columns = df.columns.str.strip().str.upper()
        return df
    except FileNotFoundError:
        st.error("❌ Rainfall_Data_LL.csv not found in app directory.")
        return None

def create_features(data, target_year=None):
    df = data.copy()
    # Basic safeguards
    if 'YEAR' not in df.columns or 'ANNUAL' not in df.columns:
        raise ValueError("Input must contain 'YEAR' and 'ANNUAL' columns.")

    # Lag features
    for i in range(1, 10):
        df[f'Lag{i}'] = df["ANNUAL"].shift(i)

    # Rolling statistics
    for window in [2, 3, 5, 7, 10, 15]:
        df[f'MA{window}'] = df["ANNUAL"].rolling(window).mean()
        df[f'STD{window}'] = df["ANNUAL"].rolling(window).std()
        df[f'Min{window}'] = df["ANNUAL"].rolling(window).min()
        df[f'Max{window}'] = df["ANNUAL"].rolling(window).max()
        df[f'Range{window}'] = df[f'Max{window}'] - df[f'Min{window}']
        df[f'Median{window}'] = df["ANNUAL"].rolling(window).median()
        df[f'Q25_{window}'] = df["ANNUAL"].rolling(window).quantile(0.25)
        df[f'Q75_{window}'] = df["ANNUAL"].rolling(window).quantile(0.75)

    # EMA
    for span in [2, 3, 5, 7, 10]:
        df[f'EMA{span}'] = df["ANNUAL"].ewm(span=span, adjust=False).mean()

    # Trend features
    df['Year_Norm'] = (df['YEAR'] - df['YEAR'].min()) / (df['YEAR'].max() - df['YEAR'].min() + 1e-6)
    df['Year_Squared'] = df['Year_Norm'] ** 2

    # Cyclical features
    for cycle in [3, 5, 7, 11, 15]:
        df[f'Cycle{cycle}_Sin'] = np.sin(2 * np.pi * df['YEAR'] / cycle)
        df[f'Cycle{cycle}_Cos'] = np.cos(2 * np.pi * df['YEAR'] / cycle)

    # Rate of change and momentum
    for period in [1, 2, 3, 5, 7]:
        df[f'Rate_Change_{period}'] = df["ANNUAL"].pct_change(periods=period)
        df[f'Momentum_{period}'] = df["ANNUAL"] - df["ANNUAL"].shift(period)

    # Volatility measures
    for window in [3, 5, 7, 10]:
        df[f'Volatility_{window}'] = df["ANNUAL"].rolling(window).std() / (df["ANNUAL"].rolling(window).mean() + 1e-6)

    # Interaction features
    df['Lag1_x_MA3'] = df['ANNUAL'].shift(1) * df["ANNUAL"].rolling(3).mean()
    df['Lag1_diff_MA5'] = df['ANNUAL'].shift(1) - df["ANNUAL"].rolling(5).mean()

    # Percentile and z-score features
    df['Percentile_Rank'] = df["ANNUAL"].rank(pct=True)
    for window in [5, 10, 15]:
        rolling_mean = df["ANNUAL"].rolling(window).mean()
        rolling_std = df["ANNUAL"].rolling(window).std()
        df[f'Zscore_{window}'] = (df["ANNUAL"] - rolling_mean) / (rolling_std + 1e-6)

    df = df.dropna().reset_index(drop=True)

    # If target_year is beyond data, append a future row with estimated features
    if target_year and target_year > df['YEAR'].max():
        last_row = df.iloc[-1].copy()
        future_row = pd.Series(dtype=float)
        future_row['YEAR'] = target_year
        # populate lag values
        for i in range(1, 10):
            if i == 1:
                future_row[f'Lag{i}'] = last_row['ANNUAL']
            else:
                future_row[f'Lag{i}'] = last_row.get(f'Lag{i-1}', last_row['ANNUAL'])
        # rolling stats from recent values
        recent_values = df['ANNUAL'].tail(20).values
        for window in [2, 3, 5, 7, 10, 15]:
            if len(recent_values) >= window:
                future_row[f'MA{window}'] = np.mean(recent_values[-window:])
                future_row[f'STD{window}'] = np.std(recent_values[-window:])
                future_row[f'Median{window}'] = np.median(recent_values[-window:])
                future_row[f'Q25_{window}'] = np.percentile(recent_values[-window:], 25)
                future_row[f'Q75_{window}'] = np.percentile(recent_values[-window:], 75)
            else:
                future_row[f'MA{window}'] = last_row.get(f'MA{window}', last_row['ANNUAL'])
                future_row[f'STD{window}'] = last_row.get(f'STD{window}', 0.0)
                future_row[f'Median{window}'] = last_row['ANNUAL']
                future_row[f'Q25_{window}'] = last_row['ANNUAL']
                future_row[f'Q75_{window}'] = last_row['ANNUAL']
        for span in [2, 3, 5, 7, 10]:
            future_row[f'EMA{span}'] = last_row.get(f'EMA{span}', last_row['ANNUAL'])
        future_row['Year_Norm'] = (target_year - df['YEAR'].min()) / (df['YEAR'].max() - df['YEAR'].min() + 1e-6)
        future_row['Year_Squared'] = future_row['Year_Norm'] ** 2
        for cycle in [3, 5, 7, 11, 15]:
            future_row[f'Cycle{cycle}_Sin'] = np.sin(2 * np.pi * target_year / cycle)
            future_row[f'Cycle{cycle}_Cos'] = np.cos(2 * np.pi * target_year / cycle)
        # safe defaults for other columns
        future_row['Percentile_Rank'] = 0.5
        for window in [5, 10, 15]:
            future_row[f'Zscore_{window}'] = 0.0
        df = pd.concat([df, pd.DataFrame([future_row])], ignore_index=True)

    return df

def predict_rainfall(df, region_name, year):
    region_match = df[df["SUBDIVISION"].str.contains(region_name, case=False, na=False)]
    if region_match.empty:
        return None, None, "Region not found"

    data = region_match.groupby("YEAR")["ANNUAL"].mean().reset_index().dropna()
    data = data.sort_values("YEAR").reset_index(drop=True)

    data_features = create_features(data, year if year > data['YEAR'].max() else None)
    feature_cols = [col for col in data_features.columns if col not in ['YEAR', 'ANNUAL']]

    if year in data['YEAR'].values:
        # Historical year: Lasso + Ridge ensemble
        train_data = data_features[data_features['YEAR'] < year]
        test_data = data_features[data_features['YEAR'] == year]

        X_train = train_data[feature_cols].values
        y_train = train_data['ANNUAL'].values
        X_test = test_data[feature_cols].values

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        tscv = TimeSeriesSplit(n_splits=max(2, min(10, len(X_train) // 3)))

        lasso_cv = LassoCV(cv=tscv, random_state=42, max_iter=20000, n_alphas=100)
        lasso_cv.fit(X_train_scaled, y_train)
        pred_lasso = float(lasso_cv.predict(X_test_scaled)[0])

        ridge = Ridge(alpha=1.0, random_state=42)
        ridge.fit(X_train_scaled, y_train)
        pred_ridge = float(ridge.predict(X_test_scaled)[0])

        predicted = 0.6 * pred_lasso + 0.4 * pred_ridge

    else:
        # Future year: ensemble of XGB, RF, GB
        train_data = data_features[data_features['YEAR'] <= data['YEAR'].max()]
        test_data = data_features[data_features['YEAR'] == year]

        X_train = train_data[feature_cols].values
        y_train = train_data['ANNUAL'].values
        X_test = test_data[feature_cols].values

        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        xgb_model = XGBRegressor(
            n_estimators=1500, learning_rate=0.03, max_depth=4,
            subsample=0.85, colsample_bytree=0.85, min_child_weight=3,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0, random_state=42, verbosity=0
        )
        xgb_model.fit(X_train_scaled, y_train)
        pred_xgb = float(xgb_model.predict(X_test_scaled)[0])

        rf_model = RandomForestRegressor(n_estimators=500, max_depth=8, min_samples_split=5, min_samples_leaf=2, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        pred_rf = float(rf_model.predict(X_test_scaled)[0])

        gb_model = GradientBoostingRegressor(n_estimators=500, learning_rate=0.05, max_depth=4, subsample=0.8, random_state=42)
        gb_model.fit(X_train_scaled, y_train)
        pred_gb = float(gb_model.predict(X_test_scaled)[0])

        high_variability_regions = ['Kerala', 'Assam & Meghalaya', 'West Bengal', 'Orissa', 'Coastal Karnataka', 'Konkan & Goa']
        if any(region.lower() in region_name.lower() for region in high_variability_regions):
            w_xgb, w_rf, w_gb, w_hist = 0.35, 0.30, 0.25, 0.10
        else:
            w_xgb, w_rf, w_gb, w_hist = 0.30, 0.25, 0.25, 0.20

        N = min(15, len(y_train))
        hist_mean = float(np.mean(y_train[-N:]))

        predicted = (w_xgb * pred_xgb) + (w_rf * pred_rf) + (w_gb * pred_gb) + (w_hist * hist_mean)

        # Bound prediction to reasonable range
        if len(y_train) >= 1:
            recent_min = np.min(y_train[-20:])
            recent_max = np.max(y_train[-20:])
            buffer = (recent_max - recent_min) * 0.3
            predicted = float(np.clip(predicted, recent_min - buffer, recent_max + buffer))

    actual = None
    if year in data['YEAR'].values:
        actual = float(data.loc[data['YEAR'] == year, 'ANNUAL'].values[0])

    return predicted, actual, data

# ============================================================
# FLOOD / DROUGHT MODEL (TRAINING)
# ============================================================
@st.cache_data
def load_and_train_flood_model():
    try:
        df = pd.read_csv("rainfallpred.csv")
    except FileNotFoundError:
        st.error("❌ rainfallpred.csv not found in app directory.")
        return None, None, None, None, None

    months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']
    df['Total'] = df[months].sum(axis=1)

    # Regional SPI normalization
    df['region_mean'] = df.groupby('SUBDIVISION')['Total'].transform('mean')
    df['region_std'] = df.groupby('SUBDIVISION')['Total'].transform('std').replace(0, 1e-6)
    df['SPI_region'] = (df['Total'] - df['region_mean']) / df['region_std']

    # Dynamic percentiles and label
    df['drought_pctl'] = df.groupby('SUBDIVISION')['Total'].transform(lambda x: x.quantile(0.10))
    df['flood_pctl'] = df.groupby('SUBDIVISION')['Total'].transform(lambda x: x.quantile(0.90))

    def classify_region(row):
        if row['SPI_region'] <= -1.2 or row['Total'] < row['drought_pctl']:
            return "Drought"
        elif row['SPI_region'] >= 1.2 or row['Total'] > row['flood_pctl']:
            return "Flood"
        else:
            return "Normal"

    df['Label'] = df.apply(classify_region, axis=1)

    # Feature engineering
    df['Mean_Rain'] = df[months].mean(axis=1)
    df['Std_Rain'] = df[months].std(axis=1)
    df['CoeffVar'] = df['Std_Rain'] / (df['Mean_Rain'] + 1e-6)
    df['Skewness'] = df[months].skew(axis=1)
    df['Kurtosis'] = df[months].kurtosis(axis=1)

    df['Monsoon_Rain'] = df[['JUN','JUL','AUG','SEP']].sum(axis=1)
    df['Winter_Rain'] = df[['DEC','JAN','FEB']].sum(axis=1)
    df['Post_Monsoon_Rain'] = df[['OCT','NOV']].sum(axis=1)

    df['Monsoon_Prop'] = df['Monsoon_Rain'] / (df['Total'] + 1e-6)
    df['Dry_Months'] = (df[months] < df[months].quantile(0.25, axis=1).values[:, None]).sum(axis=1)
    df['Wet_Months'] = (df[months] > df[months].quantile(0.75, axis=1).values[:, None]).sum(axis=1)
    df['Concentration_Index'] = df[months].max(axis=1) / (df['Total'] + 1e-6)

    df['Prev_Total'] = df.groupby('SUBDIVISION')['Total'].shift(1)
    df['Prev_SPI'] = df.groupby('SUBDIVISION')['SPI_region'].shift(1)
    df['Prev_Label'] = df.groupby('SUBDIVISION')['Label'].shift(1)
    df.fillna(0, inplace=True)

    le_temp = LabelEncoder()
    df['Prev_Label_Encoded'] = le_temp.fit_transform(df['Prev_Label'].fillna('Normal').astype(str))

    # Region encoding
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    region_encoded = pd.DataFrame(enc.fit_transform(df[['SUBDIVISION']]), columns=enc.get_feature_names_out(['SUBDIVISION']))
    df = pd.concat([df, region_encoded], axis=1)

    feature_cols = (months + ['Mean_Rain','Std_Rain','CoeffVar','Skewness','Kurtosis',
                             'Monsoon_Rain','Winter_Rain','Post_Monsoon_Rain',
                             'Monsoon_Prop','Dry_Months','Wet_Months','Concentration_Index',
                             'Prev_Total','Prev_SPI','Prev_Label_Encoded'] + list(region_encoded.columns))

    X = df[feature_cols]
    y = df['Label']

    # SMOTE balancing
    sm = SMOTE(random_state=42, k_neighbors=3)
    X_res, y_res = sm.fit_resample(X, y)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_res)

    X_train, X_test, y_train, y_test = train_test_split(X_res, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)

    clf = XGBClassifier(
        n_estimators=800, max_depth=7, learning_rate=0.03, subsample=0.85,
        colsample_bytree=0.85, min_child_weight=3, gamma=0.15,
        reg_alpha=0.1, reg_lambda=1.5, objective='multi:softprob',
        num_class=len(le.classes_), random_state=42, verbosity=0
    )
    clf.fit(X_train, y_train)

    return clf, le, df, feature_cols, months

def predict_flood_drought(clf, le, df, feature_cols, months, region_name, year, monthly_data=None):
    # Use existing row if present
    row = df[(df['SUBDIVISION'] == region_name) & (df['YEAR'] == year)]
    if not row.empty:
        features = row[feature_cols]
        pred = np.argmax(clf.predict_proba(features), axis=1)
        pred_label = le.inverse_transform(pred)[0]
        return pred_label, None

    # Manual input
    if monthly_data is None or len(monthly_data) != 12:
        return None, "Need 12 monthly rainfall values."

    manual_df = pd.DataFrame([monthly_data], columns=months)
    manual_df['Total'] = manual_df[months].sum(axis=1)

    region_data = df[df['SUBDIVISION'] == region_name]
    if region_data.empty:
        return None, f"Region '{region_name}' not found in dataset."

    region_mean = region_data['Total'].mean()
    region_std = region_data['Total'].std() if region_data['Total'].std() > 0 else 1e-6

    manual_df['Mean_Rain'] = manual_df[months].mean(axis=1)
    manual_df['Std_Rain'] = manual_df[months].std(axis=1)
    manual_df['CoeffVar'] = manual_df['Std_Rain'] / (manual_df['Mean_Rain'] + 1e-6)
    manual_df['Skewness'] = manual_df[months].skew(axis=1)
    manual_df['Kurtosis'] = manual_df[months].kurtosis(axis=1)

    manual_df['Monsoon_Rain'] = manual_df[['JUN','JUL','AUG','SEP']].sum(axis=1)
    manual_df['Post_Monsoon_Rain'] = manual_df[['OCT','NOV']].sum(axis=1)
    manual_df['Monsoon_Prop'] = manual_df['Monsoon_Rain'] / (manual_df['Total'] + 1e-6)

    monthly_vals = manual_df[months].values[0]
    q25 = np.percentile(monthly_vals, 25)
    q75 = np.percentile(monthly_vals, 75)
    manual_df['Dry_Months'] = (monthly_vals < q25).sum()
    manual_df['Wet_Months'] = (monthly_vals > q75).sum()
    manual_df['Concentration_Index'] = manual_df[months].max(axis=1) / (manual_df['Total'] + 1e-6)

    manual_df['SPI_region'] = (manual_df['Total'] - region_mean) / region_std
    manual_df['Prev_SPI'] = region_data['SPI_region'].iloc[-1] if len(region_data) > 0 else 0
    manual_df['Prev_Label_Encoded'] = 1
    manual_df['Prev_Total'] = region_mean
    manual_df['MA_3yr'] = region_mean
    manual_df['MA_5yr'] = region_mean

    # Region encoding columns
    region_encoded_cols = [col for col in feature_cols if col.startswith('SUBDIVISION_')]
    region_row = pd.DataFrame(np.zeros((1, len(region_encoded_cols))), columns=region_encoded_cols)
    if f"SUBDIVISION_{region_name}" in region_row.columns:
        region_row[f"SUBDIVISION_{region_name}"] = 1

    manual_features = pd.concat([manual_df, region_row], axis=1)
    for col in feature_cols:
        if col not in manual_features.columns:
            manual_features[col] = 0
    manual_features = manual_features[feature_cols]

    pred_idx = np.argmax(clf.predict_proba(manual_features)[0])
    pred_label = le.inverse_transform([pred_idx])[0]

    return pred_label, None

# ============================================================
# MAIN APP
# ============================================================
def main():
    st.title("🌦️ Rainfall Forecast & Flood/Drought Predictor (Minimal)")

    with st.sidebar:
        st.header("Navigation")
        mode = st.radio("Select Mode", ["Home", "Rainfall Forecasting", "Flood/Drought Prediction"])
        st.write("---")
        st.info("This minimal app provides text-only predictions. No charts or analytics.")

    if mode == "Home":
        st.markdown("**Available functionality:**")
        st.markdown("- Rainfall forecasting (multi-model ensemble)\n- Flood/Drought classification (XGBoost)\n\nPlace `Rainfall_Data_LL.csv` and `rainfallpred.csv` in the app directory before running.")
        return

    # Rainfall Forecasting
    if mode == "Rainfall Forecasting":
        st.header("Annual Rainfall Forecasting")
        df = load_rainfall_data()
        if df is None:
            return

        regions = sorted(df["SUBDIVISION"].unique())
        region = st.selectbox("Select Region", regions)
        year = st.number_input("Enter Year to predict", min_value=1900, max_value=2100, value=2025)

        if st.button("Predict Rainfall"):
            with st.spinner("Running rainfall prediction..."):
                predicted, actual, result = predict_rainfall(df, region, year)

            if isinstance(result, str):
                st.error(f"❌ {result}")
            else:
                # Display predictions as plain text (no charts, no metrics)
                st.success("Prediction completed.")
                st.markdown(f"**Region:** {region}")
                st.markdown(f"**Year:** {year}")
                st.markdown(f"**Predicted Annual Rainfall:** {predicted:.2f} mm")
                if actual is not None:
                    st.markdown(f"**Actual Annual Rainfall (available):** {actual:.2f} mm")

    # Flood/Drought Prediction
    elif mode == "Flood/Drought Prediction":
        st.header("Flood & Drought Risk Assessment")
        clf, le, df_fd, feature_cols, months = load_and_train_flood_model()
        if clf is None:
            return

        regions = sorted(df_fd["SUBDIVISION"].unique())
        region = st.selectbox("Select Region", regions)
        year = st.number_input("Enter Year", min_value=1900, max_value=2100, value=2025)
        use_manual = st.checkbox("Enter monthly rainfall manually")

        monthly_data = None
        if use_manual:
            st.subheader("Enter monthly rainfall (mm)")
            cols = st.columns(4)
            monthly_data = []
            for i, m in enumerate(months):
                with cols[i % 4]:
                    typical_val = float(df_fd[m].mean()) if m in df_fd.columns else 100.0
                    val = st.number_input(m, min_value=0.0, max_value=10000.0, value=typical_val, key=f"mon_{m}")
                    monthly_data.append(val)

        if st.button("Predict Condition"):
            with st.spinner("Predicting flood/drought condition..."):
                pred_label, error = predict_flood_drought(clf, le, df_fd, feature_cols, months, region, year, monthly_data)

            if error:
                st.error(f"❌ {error}")
            else:
                # Show only label and recommendations (no probabilities / metrics)
                st.success("Prediction completed.")
                st.markdown(f"**Region:** {region}")
                st.markdown(f"**Year:** {year}")
                st.markdown(f"**Predicted Condition:** **{pred_label}**")

                st.markdown("---")
                st.subheader("Recommended Actions")
                if pred_label == "Flood":
                    st.markdown("""
                    - Check drainage and flood defenses
                    - Prepare emergency kits and evacuation routes
                    - Coordinate with local disaster management authorities
                    """)
                elif pred_label == "Drought":
                    st.markdown("""
                    - Implement water conservation measures
                    - Prepare drought-resilient agriculture strategies
                    - Plan water rationing and monitor reservoirs
                    """)
                else:
                    st.markdown("""
                    - Routine monitoring and standard preparedness
                    - Maintain regular agriculture and infrastructure schedules
                    """)

if __name__ == "__main__":
    main()
