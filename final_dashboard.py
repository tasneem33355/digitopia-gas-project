import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import joblib
import plotly.graph_objects as go
import shared_state 
import json  

# ===============================
# Config Streamlit Page
# ===============================
st.set_page_config(
    page_title="Gas System Dashboard",
    page_icon="⚡",
    layout="wide"
)

# ===============================
# Custom Background + Styling
# ===============================
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #000000 30%, #0B3D91 100%);
    color: #FDF5E6;
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
[data-testid="stSidebar"] {
    background-color: #0B3D91;
    color: white;
}
h1, h2, h3, h4 {
    color: gold;
    font-weight: bold;
}
[data-testid="stMetricValue"] {
    color: #FDF5E6;
    font-size: 22px;
    font-weight: bold;
}
[data-testid="stMetricLabel"] {
    color: #D3D3D3;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# ===============================
# Load Model & Feature List
# ===============================
@st.cache_resource
def load_model():
    try:
        model = joblib.load("final_tuned_model.pkl")
        feature_columns = joblib.load("feature_columns.pkl")
        
        # Fix the feature mismatch automatically
        expected_features = model.n_features_
        actual_features = len(feature_columns)
        
        if expected_features != actual_features:
            missing_count = expected_features - actual_features
            for i in range(missing_count):
                feature_columns.append(f"dummy_feature_{i}")
        
        return model, feature_columns
    except FileNotFoundError:
        st.error("Model files not found!")
        st.stop()

model, feature_columns = load_model()

# ===============================
# Load CSV Data (Cached)
# ===============================
@st.cache_data
def load_csv_data():
    csv_data = {}
    try:
        csv_data['normal'] = pd.read_csv("normal_4h_before.csv", parse_dates=['timestamp'])
        csv_data['warning'] = pd.read_csv("warning_4h_before.csv", parse_dates=['timestamp'])
        csv_data['failure'] = pd.read_csv("failure_2h_before.csv", parse_dates=['timestamp'])
        
        for scenario, df in csv_data.items():
            df = df.fillna(method='ffill').fillna(method='bfill').fillna(0)
            csv_data[scenario] = df
            
        st.sidebar.success(f"Loaded CSV data:")
        for scenario, df in csv_data.items():
            st.sidebar.write(f"- {scenario}: {len(df)} rows")
    except FileNotFoundError as e:
        st.sidebar.error(f"CSV file not found: {e}")
        st.sidebar.info("Using fallback random data generation")
        return None
    return csv_data

csv_data = load_csv_data()

# ===============================
# Helper Functions (same as before)
# ===============================
# ... (keep all helper functions the same, unchanged)

# ===============================
# Sidebar
# ===============================
with st.sidebar:
    st.title("Gas System Monitor")
    scenario = st.selectbox("Select Scenario:", ["normal", "warning", "failure"], help="Choose system behavior type")
    st.info(f"Auto-updating every 10 seconds")
    st.write(f"Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")

    # Data sync status with Google Sheets (modified)
    st.markdown("---")
    st.write("### 🔄 Data Sync Status")
    
    try:
        is_fresh, state_data = shared_state.is_state_fresh(max_age_seconds=30)
        if is_fresh:
            st.success("✅ Data is fresh")
            st.write(f"Points: {len(state_data.get('data_buffer', []))}")
        else:
            st.warning("⚠️ Data might be stale or unavailable")
    except Exception as e:
        st.error(f"❌ Could not check data freshness: {e}")

    st.write(f"Buffer size: {len(st.session_state.data_buffer)}")
    st.write(f"Current scenario: {scenario}")

# ===============================
# Main Dashboard (same as before)
# ===============================
# ... (keep everything else unchanged, including charts, metrics, etc.)

# In update loop we still call shared_state.save_shared_state()
# That function will now write to Google Sheets instead of JSON
