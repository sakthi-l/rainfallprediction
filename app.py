import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import TimeSeriesSplit

st.set_page_config(page_title="Flood/Drought & Rainfall Predictor", layout="wide")

# ==============================
# LOAD DATA
# ==============================
@st.cache_data
def load_data():
    df_class = pd.read_csv("rainfallpred.csv")
    df_forecast = pd.read_csv("Rainfall_Data_LL.csv")
    df_forecast.columns = df_forecast.columns.str.strip().str.upper()
    return df_class, df_forecast

df_class, df_forecast = load_data()

months = ['JAN','FEB','MAR','APR','MAY','JUN','JUL','AUG','SEP','OCT','NOV','DEC']

# ==============================
# SIDE MENU
# ==============================
st.sidebar.title("Options")
task = st.sidebar.radio("Choose Task:", ["Flood/Drought Classification", "Rainfall Forecasting"])

# ==============================
# FLOOD/DROUGHT CLASSIFICATION
# ==============================
if task == "Flood/Drought Classification":
    st.header("🌦️ Flood/Drought Classification")

    # Select region and year
    region = st.selectbox("Select Region", sorted(df_class['SUBDIVISION'].unique()))
    year = st.number_input("Enter Year", min_value=int(df_class['YEAR'].min()), max_value=int(df_class['YEAR'].max()), value=int(df_class['YEAR'].max()))

    # --- Training code (from your script) ---
    # Compute Total & SPI
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

    feature_cols = months + ['Mean_Rain','Std_Rain','CoeffVar','Dry_Months','Wet_Months','Max_Month','Diff_Total','Prev_SPI'] + list(region_encoded.columns)
    X = df_class[feature_cols]
    y = df_class['Label']

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X, y)

    le = LabelEncoder()
    y_encoded = le.fit_transform(y_res)

    clf = XGBClassifier(
        n_estimators=400, max_depth=6, learning_rate=0.05,
        subsample=0.9, colsample_bytree=0.9, gamma=0.1,
        objective='multi:softprob', num_class=3, random_state=42
    )
    clf.fit(X_res, y_encoded)

    # Prediction
    row = df_class[(df_class['SUBDIVISION']==region) & (df_class['YEAR']==year)]
    if not row.empty:
        features = row[feature_cols]
        pred_label = le.inverse_transform(np.argmax(clf.predict_proba(features), axis=1))[0]
    else:
        pred_label = "No data available"

    st.success(f"🌦️ Prediction for **{region}** in **{year}**: **{pred_label}**")

# ==============================
# RAINFALL FORECASTING
# ==============================
else:
        # ==============================
    # RAINFALL FORECASTING (ROBUST)
    # ==============================
    st.header("🌧️ Rainfall Forecasting (Lasso)")
    
    region_input = st.text_input("Enter Region Name (Exact or Partial)", "")
    year_input = st.number_input(
        "Enter Year to Forecast",
        min_value=int(df_forecast['YEAR'].min()),
        max_value=int(df_forecast['YEAR'].max()) + 10,
        value=int(df_forecast['YEAR'].max())
    )
    
    if st.button("Forecast"):
    
        # --- Find matching region ---
        region_match = df_forecast[df_forecast['SUBDIVISION'].str.contains(region_input, case=False, na=False)]
        if region_match.empty:
            st.error("❌ Region not found!")
        else:
            data = region_match.groupby("YEAR")["ANNUAL"].mean().reset_index().dropna()
            data = data.sort_values("YEAR").reset_index(drop=True)
    
            # --- Feature engineering ---
            def create_features_safe(data, target_year=None):
                df = data.copy()
    
                # Lag features (up to 7 years)
                for i in range(1, 8):
                    df[f'Lag{i}'] = df["ANNUAL"].shift(i)
    
                # Rolling statistics
                for window in [2, 3, 5, 7, 10]:
                    df[f'MA{window}'] = df["ANNUAL"].rolling(window).mean()
                    df[f'STD{window}'] = df["ANNUAL"].rolling(window).std()
                    df[f'Min{window}'] = df["ANNUAL"].rolling(window).min()
                    df[f'Max{window}'] = df["ANNUAL"].rolling(window).max()
                    df[f'Range{window}'] = df[f'Max{window}'] - df[f'Min{window}']
    
                # Exponential moving averages
                for span in [2, 3, 5, 7]:
                    df[f'EMA{span}'] = df["ANNUAL"].ewm(span=span, adjust=False).mean()
    
                # Trend & cyclical features
                df['Year_Norm'] = (df['YEAR'] - df['YEAR'].min()) / (df['YEAR'].max() - df['YEAR'].min())
                df['Year_Squared'] = df['Year_Norm'] ** 2
                df['Year_Cubed'] = df['Year_Norm'] ** 3
                for cycle in [5, 7, 11]:
                    df[f'Cycle{cycle}_Sin'] = np.sin(2 * np.pi * df['YEAR'] / cycle)
                    df[f'Cycle{cycle}_Cos'] = np.cos(2 * np.pi * df['YEAR'] / cycle)
    
                # Rate of change and momentum
                df['Rate_Change'] = df["ANNUAL"].pct_change().fillna(0)
                df['Rate_Change_2'] = df["ANNUAL"].pct_change(periods=2).fillna(0)
                df['Momentum_3'] = df["ANNUAL"] - df["ANNUAL"].shift(3)
                df['Momentum_5'] = df["ANNUAL"] - df["ANNUAL"].shift(5)
    
                # Volatility
                df['Volatility_3'] = df["ANNUAL"].rolling(3).std() / df["ANNUAL"].rolling(3).mean()
                df['Volatility_5'] = df["ANNUAL"].rolling(5).std() / df["ANNUAL"].rolling(5).mean()
    
                # Interactions
                df['Lag1_x_MA3'] = df['Lag1'] * df['MA3']
                df['Lag1_x_Year'] = df['Lag1'] * df['Year_Norm']
    
                # Fill any remaining NaNs with forward/backward fill
                df.fillna(method='ffill', inplace=True)
                df.fillna(method='bfill', inplace=True)
                df.fillna(0, inplace=True)  # fallback
    
                return df
    
            data_features = create_features_safe(data, year_input if year_input > data['YEAR'].max() else None)
            feature_cols = [col for col in data_features.columns if col not in ['YEAR','ANNUAL']]
    
            # --- Train/Test split ---
            train_data = data_features[data_features['YEAR'] <= data['YEAR'].max()]
            test_data = data_features[data_features['YEAR'] == year_input]
    
            # --- Features & scaling ---
            X_train = train_data[feature_cols].values
            y_train = train_data['ANNUAL'].values
            X_test = test_data[feature_cols].values
    
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
    
            # --- Lasso model ---
            from sklearn.model_selection import TimeSeriesSplit
            tscv = TimeSeriesSplit(n_splits=min(5, len(X_train)//5))
            lasso_cv = LassoCV(cv=tscv, random_state=42, max_iter=10000, n_alphas=100)
            lasso_cv.fit(X_train_scaled, y_train)
    
            predicted = float(lasso_cv.predict(X_test_scaled)[0])
            actual = None
            if year_input in data['YEAR'].values:
                actual = float(data.loc[data['YEAR']==year_input,'ANNUAL'].values[0])
    
            # --- Display results ---
            st.subheader(f"🌦️ Forecast for {region_input.title()} in {year_input}")
            if actual is not None: st.write(f"Actual: {actual:.2f} mm")
            st.write(f"Predicted: {predicted:.2f} mm")
            if actual is not None:
                error = abs(predicted-actual)
                error_pct = error/actual*100
                st.write(f"Error: {error:.2f} mm ({error_pct:.2f}%)")
