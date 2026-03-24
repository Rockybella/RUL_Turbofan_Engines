import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# --- PAGE CONFIG ---
st.set_page_config(page_title="Jet Engine Predictive Maintenance", layout="wide")
st.title("🚀 Jet Engine Remaining Useful Life (RUL) Predictor")
st.markdown("This dashboard predicts the remaining lifespan of NASA turbofan engines using a Random Forest model.")

# --- DATA LOADING (CACHED) ---
@st.cache_data
def load_data():
    col_names = ['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)]
    # Load Training Data
    df = pd.read_csv('CMAPSSData/train_FD001.txt', sep='\s+', header=None, names=col_names)
    
    # Target Engineering (Piecewise RUL)
    max_cycle = df.groupby('unit')['cycle'].transform('max')
    df['RUL'] = (max_cycle - df['cycle'])
    df['RUL_Piecewise'] = df['RUL'].clip(upper=125)
    
    # Feature Selection
    stats = df.describe().T
    constant_sensors = stats[stats['std'] == 0].index.tolist()
    drop_cols = constant_sensors + ['op3', 's1', 's5', 's10', 's16', 's18', 's19']
    features = [f for f in df.columns if f not in (['unit', 'cycle', 'RUL', 'RUL_Piecewise'] + drop_cols)]
    
    return df, features, drop_cols

df, features, drop_cols = load_data()

# --- MODEL TRAINING / LOADING ---
@st.cache_resource
def get_model(X, y):
    if os.path.exists('turbofan_model.pkl'):
        return joblib.load('turbofan_model.pkl')
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)
        joblib.dump(model, 'turbofan_model.pkl')
        return model

X = df[features]
y = df['RUL_Piecewise']
model = get_model(X, y)

# --- SIDEBAR NAVIGATION ---
st.sidebar.header("Navigation")
page = st.sidebar.radio("Go to", ["Fleet Overview", "Engine-Specific Analysis", "Model Validation"])

# --- PAGE 1: FLEET OVERVIEW ---
if page == "Fleet Overview":
    st.header("📊 Fleet-Wide Sensor Correlation")
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df[features].corr(), cmap='RdBu_r', center=0, ax=ax)
    st.pyplot(fig)
    
    st.header("🌡️ Raw Sensor Data (First 5 Engines)")
    st.write(df.head())

# --- PAGE 2: ENGINE-SPECIFIC ANALYSIS ---
elif page == "Engine-Specific Analysis":
    st.header("🔍 Individual Engine Health")
    
    unit_id = st.selectbox("Select Engine Unit ID", df['unit'].unique())
    unit_data = df[df['unit'] == unit_id]
    
    # Prediction
    unit_X = unit_data[features]
    preds = model.predict(unit_X)
    
    # Plotting
    fig, ax = plt.subplots(figsize=(10, 5))
    plt.style.use('dark_background')
    ax.plot(unit_data['cycle'], unit_data['RUL_Piecewise'], label="Actual RUL (Clipped)", color="white", linewidth=2)
    ax.plot(unit_data['cycle'], preds, label="Predicted RUL", color="cyan", linestyle="--")
    ax.set_xlabel("Cycles")
    ax.set_ylabel("RUL")
    ax.legend()
    st.pyplot(fig)
    
    st.metric("Current Cycle", int(unit_data['cycle'].iloc[-1]))
    st.metric("Predicted RUL Remaining", f"{int(preds[-1])} cycles")

# --- PAGE 3: MODEL VALIDATION ---
elif page == "Model Validation":
    st.header("📈 Test Set Performance")
    
    # Load Test Data
    test_df = pd.read_csv('CMAPSSData/test_FD001.txt', sep='\s+', header=None, names=['unit', 'cycle', 'op1', 'op2', 'op3'] + [f's{i}' for i in range(1, 22)])
    true_rul = pd.read_csv('CMAPSSData/RUL_FD001.txt', sep='\s+', header=None, names=['True_RUL'])
    
    # Get last snapshots
    test_last = test_df.groupby('unit').tail(1)
    X_test = test_last[features]
    test_preds = model.predict(X_test)
    y_test = np.clip(true_rul['True_RUL'].values, 0, 125)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, test_preds))
    st.subheader(f"Final Test RMSE: {rmse:.2f} cycles")
    
    # Scatter Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(y_test, test_preds, color='cyan', alpha=0.6)
    ax.plot([0, 125], [0, 125], 'w--')
    ax.set_xlabel("Actual RUL")
    ax.set_ylabel("Predicted RUL")
    st.pyplot(fig)
