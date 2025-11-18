import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import joblib
import plotly.graph_objects as go
import shared_state  
import os  
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
# Load Model 2 (Failure Type Classification)
# ===============================
@st.cache_resource
def load_failure_type_model():
    """تحميل المودل الثاني الخاص بتصنيف نوع العطل"""
    try:
        from catboost import CatBoostClassifier
        model2 = CatBoostClassifier()
        model2.load_model("failure_type_catboost_full.cbm")
        feature_columns_model2 = joblib.load("feature_columns_model2.pkl")
        label_encoder = joblib.load("failure_label_encoder_full.pkl")
        return model2, feature_columns_model2, label_encoder
    except FileNotFoundError:
        st.sidebar.warning("⚠️ Failure Type Model not found - Feature disabled")
        return None, None, None
    except Exception as e:
        st.sidebar.error(f"❌ Error loading Failure Type Model: {e}")
        return None, None, None

model2, feature_columns_model2, label_encoder = load_failure_type_model()

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
# Helper Functions
# ===============================
def create_scenario_data(scenario="normal", n_points=1):
    timestamp = datetime.now()
    if csv_data is None:
        return create_fallback_data(scenario, timestamp)
    scenario_data = csv_data.get(scenario)
    if scenario_data is None or len(scenario_data) == 0:
        return create_fallback_data(scenario, timestamp)
    if f'{scenario}_row_index' not in st.session_state:
        st.session_state[f'{scenario}_row_index'] = 0
    row_index = st.session_state[f'{scenario}_row_index']
    if row_index >= len(scenario_data):
        row_index = 0
        st.session_state[f'{scenario}_row_index'] = 0
    current_row = scenario_data.iloc[row_index]
    st.session_state[f'{scenario}_row_index'] = row_index + 1
    return {
        "timestamp": timestamp,
        "pressure": float(current_row.get('pressure', 35.0)),
        "flow_rate": float(current_row.get('flow_rate', 70.0)),
        "temperature": float(current_row.get('temperature', 5.0)),
        "valve_status": int(current_row.get('valve_status', 0)),
        "pump_state": int(current_row.get('pump_state', 0)),
        "pump_speed": float(current_row.get('pump_speed', 1000.0)),
        "compressor_state": float(current_row.get('compressor_state', 0.5)),
        "energy_consumption": float(current_row.get('energy_consumption', 25.0)),
        "alarm_triggered": int(current_row.get('alarm_triggered', 0)),
        "hour": timestamp.hour,
        "dayofweek": timestamp.weekday()
    }

def create_fallback_data(scenario, timestamp):
    if scenario == "normal":
        pressure = np.random.normal(33.97, 9.05)
        flow_rate = np.random.normal(72.07, 21.06)
        temperature = np.random.normal(5.37, 1.86)
        energy_consumption = np.random.normal(24.14, 10.58)
        pump_speed = np.random.normal(999.74, 371.49)
    elif scenario == "warning":
        pressure = np.random.normal(34.86, 10.35)
        flow_rate = np.random.normal(66.27, 18.86)
        temperature = np.random.normal(4.96, 1.98)
        energy_consumption = np.random.normal(24.77, 9.67)
        pump_speed = np.random.normal(990.37, 403.25)
    elif scenario == "failure":
        pressure = np.random.normal(29.59, 6.19)
        flow_rate = np.random.normal(57.78, 12.08)
        temperature = np.random.normal(5.24, 1.53)
        energy_consumption = np.random.normal(24.51, 5.85)
        pump_speed = np.random.normal(1040.94, 302.46)
    return {
        "timestamp": timestamp,
        "pressure": np.clip(pressure, 5, 80),
        "flow_rate": np.clip(flow_rate, 5, 170),
        "temperature": np.clip(temperature, 0, 15),
        "valve_status": np.random.choice([0, 1]),
        "pump_state": np.random.choice([0, 1]),
        "pump_speed": np.clip(pump_speed, 0, 2000),
        "compressor_state": np.random.uniform(0, 1),
        "energy_consumption": np.clip(energy_consumption, 3, 70),
        "alarm_triggered": 0 if scenario == "normal" else np.random.choice([0, 1]),
        "hour": timestamp.hour,
        "dayofweek": timestamp.weekday()
    }

def create_features(df):
    if len(df) < 2:
        return None
    df_fe = df.copy()
    sensor_cols = ["pressure", "flow_rate", "temperature", "pump_speed", "energy_consumption"]
    LAGS = [1, 3, 6, 12, 30, 60, 120, 360]
    for col in sensor_cols:
        for lag in LAGS:
            df_fe[f"{col}_lag{lag}"] = df_fe[col].shift(lag)
    WINDOWS = [30, 60, 120, 360]
    for col in sensor_cols:
        for w in WINDOWS:
            df_fe[f"{col}_rollmean{w}"] = df_fe[col].shift(1).rolling(window=w, min_periods=1).mean()
            df_fe[f"{col}_rollstd{w}"] = df_fe[col].shift(1).rolling(window=w, min_periods=1).std()
            df_fe[f"{col}_rollmin{w}"] = df_fe[col].shift(1).rolling(window=w, min_periods=1).min()
            df_fe[f"{col}_rollmax{w}"] = df_fe[col].shift(1).rolling(window=w, min_periods=1).max()
    df_fe["hour"] = df_fe["timestamp"].dt.hour
    df_fe["dayofweek"] = df_fe["timestamp"].dt.dayofweek
    df_fe = df_fe.fillna(method='ffill').fillna(method='bfill').fillna(0)
    expected_features = model.n_features_
    feature_values = []
    for col in feature_columns:
        if col in df_fe.columns:
            feature_values.append(df_fe[col].iloc[-1])
        else:
            feature_values.append(0.0)
    while len(feature_values) < expected_features:
        feature_values.append(0.0)
    feature_values = feature_values[:expected_features]
    return np.array(feature_values).reshape(1, -1)

def predict_with_model(features):
    try:
        if features is None:
            return 0, [0.8, 0.1, 0.1]
        probabilities = model.predict_proba(features)[0]
        if probabilities[2] > 0.4:
            prediction = 2
        elif probabilities[1] > 0.4:
            prediction = 1
        else:
            prediction = 0
        return prediction, probabilities
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return 0, [0.8, 0.1, 0.1]

# ===============================
# NEW FUNCTION: Predict Failure Type
# ===============================
def predict_failure_type(df):
    """
    يتنبأ بنوع العطل باستخدام المودل الثاني
    يشتغل بس لما يكون في Failure
    """
    if model2 is None or feature_columns_model2 is None or label_encoder is None:
        return None
    
    try:
        # التأكد من وجود كل الأعمدة المطلوبة
        df_copy = df.copy()
        for col in feature_columns_model2:
            if col not in df_copy.columns:
                df_copy[col] = 0
        
        # ترتيب الأعمدة بنفس ترتيب التدريب
        X_input = df_copy[feature_columns_model2].astype(float).values
        
        # أخذ آخر صف للتنبؤ
        X_latest = X_input[-1:] if len(X_input) > 0 else X_input
        
        # Prediction
        y_pred_idx = model2.predict(X_latest).astype(int)
        y_pred_label = label_encoder.inverse_transform(y_pred_idx)
        
        return y_pred_label[0] if len(y_pred_label) > 0 else None
        
    except Exception as e:
        st.error(f"Failure Type Prediction Error: {str(e)}")
        return None

# ===============================
# Initialize Session State
# ===============================
if 'data_buffer' not in st.session_state:
    st.session_state.data_buffer = []
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'last_update' not in st.session_state:
    st.session_state.last_update = datetime.now()

# ===============================
# Sidebar
# ===============================
with st.sidebar:
    st.title("Gas System Monitor")
    scenario = st.selectbox("Select Scenario:", ["normal", "warning", "failure"], help="Choose system behavior type")
    st.info(f"Auto-updating every 10 seconds")
    st.write(f"Last update: {st.session_state.last_update.strftime('%H:%M:%S')}")

    st.markdown("""
<style>
[data-testid="stSidebar"] button, 
[data-testid="stSidebar"] div.stButton > button {
    background-color: #041C32;
    color: #FDF5E6;
    border: none;
    border-radius: 5px;
    padding: 0.35em 0.75em;
    font-weight: bold;
    transition: 0.3s;
}
[data-testid="stSidebar"] button:hover, 
[data-testid="stSidebar"] div.stButton > button:hover {
    background-color: #0B3D91;
    color: #FDF5E6;
}
</style>
""", unsafe_allow_html=True)

    if st.button("Reset Data"):
        st.session_state.data_buffer = []
        st.session_state.prediction_history = []
        for s in ['normal', 'warning', 'failure']:
            if f'{s}_row_index' in st.session_state:
                st.session_state[f'{s}_row_index'] = 0
        st.rerun()

    if st.button("🚀 Fast Simulation (Load 50 historical points)"):
        for i in range(50):
            new_point = create_scenario_data(scenario)
            st.session_state.data_buffer.append(new_point)
        st.success(f"Loaded 50 historical data points! Total: {len(st.session_state.data_buffer)}")
        st.rerun()

    if csv_data:
        st.success("✅ Using real historical CSV data")
        current_index = st.session_state.get(f'{scenario}_row_index', 0)
        total_rows = len(csv_data.get(scenario, []))
        if total_rows > 0:
            st.write(f"📊 {scenario.upper()}: Row {current_index + 1}/{total_rows}")
            progress = (current_index / total_rows) if total_rows > 0 else 0
            st.progress(progress)
    else:
        st.warning("⚠️ Using fallback random data")

    st.write(f"Data points: {len(st.session_state.data_buffer)}")
    if st.session_state.data_buffer:
        last_data = st.session_state.data_buffer[-1]
        st.write(f"Current values:")
        st.write(f"- Pressure: {last_data['pressure']:.1f}")
        st.write(f"- Flow: {last_data['flow_rate']:.1f}")
        st.write(f"- Temperature: {last_data['temperature']:.1f}")

    st.markdown("---")
    st.write("### 🔄 Data Sync Status")

    try:
        is_fresh, state_data = shared_state.is_state_fresh(max_age_seconds=30)
        if is_fresh:
            st.success(f"✅ Data is fresh from Google Sheets")
            st.write(f"Points: {len(state_data.get('data_buffer', []))}")
        else:
            st.warning("⚠️ Data might be stale")
    except Exception as e:
        st.error(f"❌ Could not check Google Sheets: {e}")

    st.write(f"Buffer size: {len(st.session_state.data_buffer)}")
    st.write(f"Current scenario: {scenario}")

# ===============================
# Main Dashboard
# ===============================
st.title("Gas System Fault Detection Dashboard")
with st.expander("ℹ️ About the Data", expanded=False):
    st.write("""
    **Historical Data Source**: This dashboard uses real sensor data from 4 hours before actual events occurred.
    **Model Training**: The AI model was trained using a 2-hour prediction horizon, meaning it predicts what will happen 2 hours in the future based on current sensor readings.
    **Why This Approach**: By using pre-event data, the model can identify subtle patterns and trends that indicate developing problems before they become critical failures.
    **Scenarios**:
    - **Normal**: Data from periods of stable, healthy operation
    - **Warning**: Data from 4 hours before warning-level events
    - **Failure**: Data from 4 hours before critical system failures
    """)

current_time = datetime.now()
if (current_time - st.session_state.last_update).total_seconds() >= 10:
    new_point = create_scenario_data(scenario)
    st.session_state.data_buffer.append(new_point)

    if len(st.session_state.data_buffer) > 500:
        st.session_state.data_buffer = st.session_state.data_buffer[-500:]

    st.session_state.last_update = current_time

    row_indices = {}
    for s in ['normal', 'warning', 'failure']:
        row_indices[s] = st.session_state.get(f'{s}_row_index', 0)

    df = pd.DataFrame(st.session_state.data_buffer)
    features = create_features(df)
    prediction, probabilities = predict_with_model(features)

    prediction_data = {
        'prediction': int(prediction),
        'probabilities': np.array(probabilities).tolist(),
        'confidence': float(np.max(probabilities))
    }

    try:
        save_success = shared_state.save_shared_state(
            st.session_state.data_buffer[-50:],  
            scenario,
            row_indices,
            prediction_data
        )
    except Exception as e:
        st.sidebar.error(f"❌ Sync error: {str(e)}")

    st.rerun()

if st.session_state.data_buffer:
    df = pd.DataFrame(st.session_state.data_buffer)
    features = create_features(df)
    prediction, probabilities = predict_with_model(features)

    st.session_state.prediction_history.append({
        'timestamp': df.iloc[-1]['timestamp'],
        'prediction': prediction,
        'confidence': np.max(probabilities),
        'probabilities': probabilities
    })

    if len(st.session_state.prediction_history) > 20:
        st.session_state.prediction_history = st.session_state.prediction_history[-20:]

    col1, col2, col3, col4 = st.columns(4)
    status_colors = {0: "🟢", 1: "🟡", 2: "🔴"}
    status_names = {0: "NORMAL", 1: "WARNING", 2: "FAILURE"}

    with col1:
        st.metric("System Status (2h Prediction)", f"{status_colors[prediction]} {status_names[prediction]}")
    with col2:
        st.metric("Confidence", f"{np.max(probabilities):.1%}")
    with col3:
        alerts = len([p for p in st.session_state.prediction_history if p['prediction'] > 0])
        st.metric("Recent Alerts", str(alerts))
    with col4:
        health = (1 - probabilities[2]) * 100
        st.metric("System Health", f"{health:.0f}%")

    # ===============================
    # NEW SECTION: Failure Type Display
    # ===============================
    if prediction == 2:  # فقط لو في Failure
        st.markdown("---")
        failure_type = predict_failure_type(df)
        
        if failure_type:
            # عرض نوع العطل بشكل بارز
            st.markdown("""
            <div style='background: linear-gradient(135deg, #8B0000, #DC143C); 
                        padding: 20px; 
                        border-radius: 15px; 
                        border: 3px solid #FF6347;
                        box-shadow: 0 8px 20px rgba(220, 20, 60, 0.5);'>
                <h2 style='color: #FDF5E6; text-align: center; margin: 0;'>
                    ⚠️ FAILURE TYPE DETECTED ⚠️
                </h2>
                <h1 style='color: gold; text-align: center; margin: 10px 0; font-size: 36px;'>
                    """ + str(failure_type) + """
                </h1>
                <p style='color: #FDF5E6; text-align: center; margin: 0; font-size: 16px;'>
                    Immediate action required to prevent system damage
                </p>
            </div>
            """, unsafe_allow_html=True)
            
            st.sidebar.markdown("---")
            st.sidebar.error(f"🚨 FAILURE TYPE: **{failure_type}**")
        else:
            st.warning("⚠️ Failure detected but type classification unavailable")

    st.sidebar.write("### Debug Info")
    st.sidebar.write(f"Prediction: {prediction} ({status_names[prediction]})")
    st.sidebar.write(f"Probabilities:")
    st.sidebar.write(f"- Normal: {probabilities[0]:.3f}")
    st.sidebar.write(f"- Warning: {probabilities[1]:.3f}")
    st.sidebar.write(f"- Failure: {probabilities[2]:.3f}")

    st.subheader("Real-Time Monitoring")
    recent_data = df.tail(50) if len(df) > 50 else df

    chart_col1, chart_col2 = st.columns(2)
    with chart_col1:
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(
            x=recent_data['timestamp'],
            y=recent_data['pressure'],
            mode='lines+markers',
            name='Pressure',
            line=dict(color='#5DADE2', width=3)
        ))
        fig1.update_layout(
            title="<b style='color:gold;'>Pressure (bar)</b>",
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="white")
        )
        st.plotly_chart(fig1, use_container_width=True)

    with chart_col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=recent_data['timestamp'],
            y=recent_data['flow_rate'],
            mode='lines+markers',
            name='Flow Rate',
            line=dict(color='#76D7C4', width=3)
        ))
        fig2.update_layout(
            title="<b style='color:gold;'>Flow Rate</b>",
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="white")
        )
        st.plotly_chart(fig2, use_container_width=True)

    chart_col3, chart_col4 = st.columns(2)
    with chart_col3:
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=recent_data['timestamp'],
            y=recent_data['temperature'],
            mode='lines+markers',
            name='Temperature',
            line=dict(color='#F39C12', width=3)
        ))
        fig3.update_layout(
            title="<b style='color:gold;'>Temperature (°C)</b>",
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="white")
        )
        st.plotly_chart(fig3, use_container_width=True)

    with chart_col4:
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=recent_data['timestamp'],
            y=recent_data['energy_consumption'],
            mode='lines+markers',
            name='Energy',
            line=dict(color='#AF7AC5', width=3)
        ))
        fig4.update_layout(
            title="<b style='color:gold;'>Energy Consumption</b>",
            height=300,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color="white")
        )
        st.plotly_chart(fig4, use_container_width=True)

    st.subheader("Recent Predictions")
    if st.session_state.prediction_history:
        pred_df = pd.DataFrame(st.session_state.prediction_history[-10:])
        status_names = {0: "NORMAL", 1: "WARNING", 2: "FAILURE"}
        pred_df['status'] = pred_df['prediction'].map(status_names)
        pred_df['time'] = pred_df['timestamp'].dt.strftime('%H:%M:%S')
        pred_df['confidence'] = pred_df['confidence'].apply(lambda x: f"{x:.1%}")
        display_df = pred_df[['time', 'status', 'confidence']].iloc[::-1]

        st.dataframe(
            display_df.style.set_properties(
                **{
                    'background-color': '#041C32',
                    'color': '#FDF5E6',
                    'font-weight': 'bold',
                    'border': '1px solid #0B3D91'
                }
            ),
            hide_index=True,
            use_container_width=True
        )

else:
    st.info("Waiting for data... Dashboard will start automatically.")

# Floating Chatbot Button (Round Style)
chatbot_url = "https://digitopia-gas-project-5uqgx5ubnmmhjyrau7knyc.streamlit.app/"

st.markdown(
    f"""
    <style>
    .chatbot-btn {{
        position: fixed;
        bottom: 30px;
        right: 30px;
        width: 70px;
        height: 70px;
        background: linear-gradient(135deg, #1E90FF, #4CA1AF);
        color: white;
        border-radius: 50%;
        text-align: center;
        font-size: 32px;
        font-weight: bold;
        line-height: 70px;
        box-shadow: 0 8px 20px rgba(0,0,0,0.35);
        z-index: 9999;
        cursor: pointer;
        transition: all 0.4s cubic-bezier(.25,.8,.25,1);
    }}
    .chatbot-btn:hover {{
        background: linear-gradient(135deg, #0B5ED7, #2C9AB7);
        transform: scale(1.2) rotate(10deg);
        box-shadow: 0 12px 25px rgba(0,0,0,0.45);
    }}
    .chatbot-btn:active {{
        transform: scale(1) rotate(0deg);
        box-shadow: 0 6px 15px rgba(0,0,0,0.3);
    }}
    </style>

    <a href="{chatbot_url}" target="_blank" class="chatbot-btn">💬</a>
    """,
    unsafe_allow_html=True
)

# Auto-refresh every few seconds
time.sleep(2)
st.rerun()