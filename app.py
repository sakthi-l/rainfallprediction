import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from pathlib import Path
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

# ---------------- STREAMLIT PAGE SETTINGS ----------------
st.set_page_config(page_title="Rainfall & Flood/Drought App", layout="wide")

# ---------------- PATH SETTINGS ----------------
DATA_DIRS = ["./data", "./", "/content"]
MODEL_DIR = Path("./models")
MODEL_DIR.mkdir(exist_ok=True)

# ---------------- FILE LOADERS ----------------
@st.cache_data
def find_file(names):
    for d in DATA_DIRS:
        for n in names:
            p = Path(d) / n
            if p.exists():
                return str(p)
    return None

@st.cache_data
def load_region_df():
    candidates = ["Rainfall_Data_LL.csv", "rainfall_dataset.csv", "region_rainfall.csv"]
    p = find_file(candidates)
    if p:
        return pd.read_csv(p)
    return None

@st.cache_data
def load_flood_df():
    candidates = ["rainfallpred.csv", "flood_drought.csv", "rainfall_pred.csv"]
    p = find_file(candidates)
    if p:
        return pd.read_csv(p)
    return None

# ---------------- FALLBACK TRAINERS ----------------
def train_region_model(df):
    """Fallback region rainfall prediction using LassoCV"""
    df2 = df.copy()
    months = [c for c in df2.columns if c.strip().upper()[:3] in
              ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
               "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]]
    if len(months) >= 12:
        df2['Total'] = df2[months].sum(axis=1)
    if 'Total' not in df2.columns or 'SUBDIVISION' not in df2.columns:
        raise ValueError("Missing required columns (SUBDIVISION, monthly or Total).")

    df2 = df2.sort_values(['SUBDIVISION', 'YEAR'])
    df2['Prev1'] = df2.groupby('SUBDIVISION')['Total'].shift(1)
    df2 = df2.dropna(subset=['Prev1'])

    X = df2[['YEAR', 'Prev1']]
    y = df2['Total']

    model = Pipeline([
        ('scaler', RobustScaler()),
        ('lasso', LassoCV(cv=5, random_state=0))
    ])
    model.fit(X, y)
    return model

def train_flood_model(df):
    """Fallback flood/drought classifier using XGBoost"""
    if 'Label' not in df.columns or 'SUBDIVISION' not in df.columns:
        raise ValueError("Missing required columns (Label, SUBDIVISION).")

    feature_cols = [c for c in df.columns if c not in ['SUBDIVISION', 'YEAR', 'Label']
                    and pd.api.types.is_numeric_dtype(df[c])]
    if not feature_cols:
        months = [c for c in df.columns if c.strip().upper()[:3] in
                  ["JAN", "FEB", "MAR", "APR", "MAY", "JUN",
                   "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]]
        feature_cols = months

    X = df[feature_cols].fillna(0)
    y = df['Label']
    clf = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    clf.fit(X, y)
    return clf, feature_cols

# ---------------- SIDEBAR NAVIGATION ----------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Region Rainfall Prediction", "Flood / Drought Classification"])

# ---------------- HOME PAGE ----------------
if page == "Home":
    st.title("🌦️ Rainfall & Flood/Drought Prediction App")
    st.write("""
    This app has **two tools**:
    - 📈 Region Rainfall Prediction
    - 🌧️ Flood/Drought Classification

    It can use your saved models or automatically train fallback models using CSV files.
    """)
    df_r = load_region_df()
    df_f = load_flood_df()
    st.write("**File Detection Status:**")
    st.write({
        'Region CSV Found': bool(df_r),
        'Flood CSV Found': bool(df_f),
        'Region Model Found': (MODEL_DIR / 'region_model.pkl').exists(),
        'Flood Model Found': (MODEL_DIR / 'flood_model.pkl').exists()
    })

# ---------------- REGION RAINFALL PAGE ----------------
if page == "Region Rainfall Prediction":
    st.header("📈 Region Rainfall Prediction")

    df_r = load_region_df()
    model_path = MODEL_DIR / 'region_model.pkl'
    model_region = None

    if model_path.exists():
        model_region = joblib.load(model_path)
        st.success("Loaded trained region model!")

    if df_r is None and model_region is None:
        st.error("Upload 'Rainfall_Data_LL.csv' to /data or add 'region_model.pkl' to /models.")
    else:
        if df_r is not None:
            st.dataframe(df_r.head())
            regions = sorted(df_r['SUBDIVISION'].unique())
        else:
            regions = []

        region = st.selectbox("Select Region", regions if regions else ["No data"])
        year = st.number_input("Year to Predict", min_value=1900, max_value=2100, value=2025)
        monthly_text = st.text_input("Monthly values (12 comma-separated numbers) [Optional]")

        if st.button("Predict Rainfall"):
            try:
                if model_region is None:
                    model_region = train_region_model(df_r)
                    joblib.dump(model_region, model_path)
                    st.info("Trained fallback model and saved to /models.")

                if monthly_text:
                    vals = [float(x) for x in monthly_text.split(',')]
                    total = sum(vals)
                    st.success(f"Provided total = {total:.2f} mm")
                else:
                    subdf = df_r[df_r['SUBDIVISION'] == region].sort_values('YEAR')
                    prev1 = subdf[subdf['YEAR'] < year]['Total'].iloc[-1] if 'Total' in subdf.columns else 0
                    X_in = pd.DataFrame({'YEAR': [year], 'Prev1': [prev1]})
                    pred = model_region.predict(X_in)[0]
                    st.metric("Predicted Annual Rainfall (mm)", f"{pred:.2f}")

            except Exception as e:
                st.error(f"Prediction Error: {e}")

# ---------------- FLOOD / DROUGHT PAGE ----------------
if page == "Flood / Drought Classification":
    st.header("🌧️ Flood / Drought Classification")

    df_f = load_flood_df()
    model_path = MODEL_DIR / 'flood_model.pkl'
    flood_model = None
    flood_features = None

    if model_path.exists():
        obj = joblib.load(model_path)
        if isinstance(obj, tuple):
            flood_model, flood_features = obj
        else:
            flood_model = obj
        st.success("Loaded trained flood/drought model!")

    if df_f is None and flood_model is None:
        st.error("Upload 'rainfallpred.csv' to /data or add 'flood_model.pkl' to /models.")
    else:
        if df_f is not None:
            st.dataframe(df_f.head())
            subdivisions = sorted(df_f['SUBDIVISION'].unique())
        else:
            subdivisions = []

        subdivision = st.selectbox("Select Subdivision", subdivisions if subdivisions else ["No data"])
        year = st.number_input("Year", min_value=1900, max_value=2100, value=2025)
        manual_text = st.text_input("Manual numeric features (comma-separated) [Optional]")

        if st.button("Classify"):
            try:
                if flood_model is None:
                    flood_model, flood_features = train_flood_model(df_f)
                    joblib.dump((flood_model, flood_features), model_path)
                    st.info("Trained fallback flood/drought model and saved to /models.")

                if manual_text:
                    vals = [float(x) for x in manual_text.split(',')]
                    X_in = np.array(vals).reshape(1, -1)
                else:
                    if flood_features is None:
                        flood_features = [c for c in df_f.columns if c not in ['SUBDIVISION', 'YEAR', 'Label']]
                    row = df_f[df_f['SUBDIVISION'] == subdivision].tail(1)
                    X_in = row[flood_features].fillna(0).values

                pred = flood_model.predict(X_in)[0]
                st.success(f"Prediction: {pred}")
                if hasattr(flood_model, 'predict_proba'):
                    proba = flood_model.predict_proba(X_in)[0]
                    st.write("Class probabilities:", proba)

            except Exception as e:
                st.error(f"Classification Error: {e}")

# ---------------- FOOTER ----------------
st.sidebar.markdown("---")
st.sidebar.info("Built with ❤️ using Streamlit. Upload models or datasets to customize predictions.")
