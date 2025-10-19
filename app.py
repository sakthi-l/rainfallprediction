# ===============================
# STREAMLIT APP: RAINFALL & FLOOD/DROUGHT
# ===============================
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split, TimeSeriesSplit
from sklearn.linear_model import LassoCV
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

st.set_page_config(page_title="Rainfall & Flood/Drought Prediction", layout="wide")

# -------------------------
# Load datasets
# -------------------------
@st.cache_data
def load_data():
    df_class = pd.read_csv("rainfallpred.csv")       # Flood/Drought classification
    df_forecast = pd.read_csv("Rainfall_Data_LL.csv")  # Rainfall forecasting
    return df_class, df_forecast

df_class, df_forecast = load_data()
st.success("✅ Data loaded successfully!")

# -------------------------
# Sidebar: Task Selection
# -------------------------
task = st.sidebar.selectbox("Choose Task:", ["Flood/Drought Classification", "Rainfall Forecasting"])

# ==========================
# FLOOD/DROUGHT CLASSIFICATION
# ==========================
if task == "Flood/Drought Classification":
    st.header("🌊 Flood/Drought Classification")

    # Region selection
    regions = df_class['SUBDIVISION'].unique()
    region_input = st.selectbox("Select Region:", sorted(regions))
    year_input = st.number_input("Enter Year:", min_value=int(df_class['YEAR'].min()), max_value=int(df_class['YEAR'].max()), value=int(df_class['YEAR'].max()))

    months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']

    # Prepare classification data
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

    # Balance classes
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)
    le = LabelEncoder()
    y_encoded = le.fit_transform(y_res)

    # Train XGBoost
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42)
    clf = XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, gamma=0.1, objective='multi:softprob', num_class=3, random_state=42)
    clf.fit(X_train, y_train)

    # Prediction
    row = df_class[(df_class['SUBDIVISION']==region_input) & (df_class['YEAR']==year_input)]
    if not row.empty:
        X_row = row[feature_cols]
        pred_label = le.inverse_transform(np.argmax(clf.predict_proba(X_row), axis=1))[0]
        st.success(f"🌦️ Prediction for {region_input} in {year_input}: {pred_label}")
    else:
        st.warning("⚠️ No historical data for this year. Please provide monthly input manually (future enhancement).")

# ==========================
# RAINFALL FORECASTING
# ==========================
# ==========================
# RAINFALL FORECASTING
# ==========================
else:
    st.header("🌧️ Rainfall Forecasting (Lasso)")

    region_input = st.text_input("Enter Region Name (Exact or Partial)", "")
    year_input = st.number_input("Enter Year to Forecast", 
                                 min_value=int(df_forecast['YEAR'].min()), 
                                 max_value=int(df_forecast['YEAR'].max())+20, 
                                 value=int(df_forecast['YEAR'].max()))

    if st.button("Forecast"):

        # Filter region
        region_match = df_forecast[df_forecast['SUBDIVISION'].str.contains(region_input, case=False, na=False)]
        if region_match.empty:
            st.error("❌ Region not found!")
            st.stop()

        # Aggregate yearly data
        data = region_match.groupby("YEAR")["ANNUAL"].mean().reset_index().dropna()
        data = data.sort_values("YEAR").reset_index(drop=True)
        last_historical_year = data['YEAR'].max()

        # Feature creation
        def create_features_safe(data):
            df = data.copy()
            for i in range(1,8): df[f'Lag{i}'] = df["ANNUAL"].shift(i)
            for window in [2,3,5,7,10]:
                df[f'MA{window}'] = df["ANNUAL"].rolling(window).mean()
                df[f'STD{window}'] = df["ANNUAL"].rolling(window).std()
                df[f'Min{window}'] = df["ANNUAL"].rolling(window).min()
                df[f'Max{window}'] = df["ANNUAL"].rolling(window).max()
                df[f'Range{window}'] = df[f'Max{window}'] - df[f'Min{window}']
            for span in [2,3,5,7]: df[f'EMA{span}'] = df["ANNUAL"].ewm(span=span, adjust=False).mean()
            df['Year_Norm'] = (df['YEAR'] - df['YEAR'].min()) / (df['YEAR'].max() - df['YEAR'].min())
            df['Year_Squared'] = df['Year_Norm']**2
            df['Year_Cubed'] = df['Year_Norm']**3
            for cycle in [5,7,11]:
                df[f'Cycle{cycle}_Sin'] = np.sin(2*np.pi*df['YEAR']/cycle)
                df[f'Cycle{cycle}_Cos'] = np.cos(2*np.pi*df['YEAR']/cycle)
            df['Rate_Change'] = df["ANNUAL"].pct_change().fillna(0)
            df['Rate_Change_2'] = df["ANNUAL"].pct_change(periods=2).fillna(0)
            df['Momentum_3'] = df["ANNUAL"] - df["ANNUAL"].shift(3)
            df['Momentum_5'] = df["ANNUAL"] - df["ANNUAL"].shift(5)
            df['Volatility_3'] = df["ANNUAL"].rolling(3).std()/df["ANNUAL"].rolling(3).mean()
            df['Volatility_5'] = df["ANNUAL"].rolling(5).std()/df["ANNUAL"].rolling(5).mean()
            df['Lag1_x_MA3'] = df['Lag1']*df['MA3']
            df['Lag1_x_Year'] = df['Lag1']*df['Year_Norm']
            df.fillna(method='ffill', inplace=True)
            df.fillna(method='bfill', inplace=True)
            df.fillna(0, inplace=True)
            return df

        data_features = create_features_safe(data)
        feature_cols = [c for c in data_features.columns if c not in ['YEAR','ANNUAL']]

        # -------------------------
        # Train Lasso
        # -------------------------
        X_train = data_features[feature_cols].values
        y_train = data_features['ANNUAL'].values
        scaler = RobustScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        tscv = TimeSeriesSplit(n_splits=min(5,len(X_train)//5))
        lasso_cv = LassoCV(cv=tscv, max_iter=10000, random_state=42)
        lasso_cv.fit(X_train_scaled, y_train)

        # -------------------------
        # Generate predictions
        # -------------------------
        predictions = []
        forecast_years = []

        # Historical years
        if year_input <= last_historical_year:
            X_test_scaled = scaler.transform(data_features[data_features['YEAR']==year_input][feature_cols].values)
            predicted = float(lasso_cv.predict(X_test_scaled)[0])
            actual = float(data.loc[data['YEAR']==year_input,'ANNUAL'].values[0])
            st.subheader(f"🌦️ Forecast for {region_input.title()} in {year_input}")
            st.write(f"Actual: {actual:.2f} mm")
            st.write(f"Predicted: {predicted:.2f} mm")
            st.write(f"Error: {abs(predicted-actual):.2f} mm ({abs(predicted-actual)/actual*100:.2f}%)")
        else:
            # Future years
            prev_predictions = data['ANNUAL'].tolist()
            for yr in range(last_historical_year+1, year_input+1):
                # Generate dynamic features using previous predictions
                last_row = data_features.iloc[-1].copy()
                future_row = last_row.copy()
                future_row['YEAR'] = yr
                future_row['ANNUAL'] = np.nan

                # Update lag features
                for i in range(1,8):
                    if i <= len(prev_predictions):
                        future_row[f'Lag{i}'] = prev_predictions[-i]
                    else:
                        future_row[f'Lag{i}'] = prev_predictions[0]

                # Update rolling averages dynamically
                for window in [2,3,5,7,10]:
                    recent_values = prev_predictions[-window:] if len(prev_predictions)>=window else prev_predictions
                    future_row[f'MA{window}'] = np.mean(recent_values)
                    future_row[f'STD{window}'] = np.std(recent_values)
                    future_row[f'Min{window}'] = np.min(recent_values)
                    future_row[f'Max{window}'] = np.max(recent_values)
                    future_row[f'Range{window}'] = future_row[f'Max{window}'] - future_row[f'Min{window}']

                # Trend features
                future_row['Year_Norm'] = (yr - data['YEAR'].min())/(data['YEAR'].max()-data['YEAR'].min())
                future_row['Year_Squared'] = future_row['Year_Norm']**2
                future_row['Year_Cubed'] = future_row['Year_Norm']**3

                # Momentum and rate of change
                future_row['Rate_Change'] = (prev_predictions[-1]-prev_predictions[-2])/prev_predictions[-2] if len(prev_predictions)>=2 else 0
                future_row['Rate_Change_2'] = (prev_predictions[-1]-prev_predictions[-3])/prev_predictions[-3] if len(prev_predictions)>=3 else 0
                future_row['Momentum_3'] = (prev_predictions[-1]-prev_predictions[-4]) if len(prev_predictions)>=4 else 0
                future_row['Momentum_5'] = (prev_predictions[-1]-prev_predictions[-6]) if len(prev_predictions)>=6 else 0

                # Fill missing features
                for col in feature_cols:
                    if col not in future_row:
                        future_row[col] = 0

                X_future = pd.DataFrame([future_row])[feature_cols]
                X_future_scaled = scaler.transform(X_future.values)
                pred = float(lasso_cv.predict(X_future_scaled)[0])
                prev_predictions.append(pred)
                predictions.append(pred)
                forecast_years.append(yr)

            st.subheader(f"🌦️ Forecast for {region_input.title()} up to {year_input}")
            forecast_df = pd.DataFrame({'Year': forecast_years, 'Predicted_Rainfall_mm': predictions})
            st.dataframe(forecast_df)
            st.line_chart(forecast_df.set_index('Year'))
