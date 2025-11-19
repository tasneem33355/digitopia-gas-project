import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import shared_state
import os
import json

import langchain
import langchain_google_genai

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage


# Config Streamlit Page
st.set_page_config(page_title="Maintenance Chatbot", page_icon="🤖", layout="wide")

# Custom Background (Same as Original)
page_bg = """
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #000000 30%, #0B3D91 100%);  
    color: #FDF5E6;  
}
[data-testid="stHeader"] {
    background: rgba(0,0,0,0);
}
input[type="text"] {
    background-color: #041C32;   
    color: #FDF5E6;             
    border: 1px solid #0B3D91;   
    padding: 8px;
    border-radius: 8px;
}
</style>

"""
st.markdown(page_bg, unsafe_allow_html=True)


# Load Model & Feature List
@st.cache_resource
def load_model():
    try:
        model = joblib.load("final_tuned_model.pkl")
        feature_columns = joblib.load("feature_columns.pkl")

        # Fix feature mismatch
        expected_features = model.n_features_
        actual_features = len(feature_columns)

        if expected_features != actual_features:
            missing_count = expected_features - actual_features
            for i in range(missing_count):
                feature_columns.append(f"dummy_feature_{i}")

        return model, feature_columns
    except FileNotFoundError:
        return None, None


model, feature_columns = load_model()


# Load CSV Data (Same as Dashboard)
@st.cache_data
def load_csv_data():
    """Load historical data from CSV files (same as dashboard)"""
    csv_data = {}

    try:
        csv_data["normal"] = pd.read_csv(
            "normal_4h_before.csv", parse_dates=["timestamp"]
        )
        csv_data["warning"] = pd.read_csv(
            "warning_4h_before.csv", parse_dates=["timestamp"]
        )
        csv_data["failure"] = pd.read_csv(
            "failure_2h_before.csv", parse_dates=["timestamp"]
        )

        for scenario, df in csv_data.items():
            df = df.fillna(method="ffill").fillna(method="bfill").fillna(0)
            csv_data[scenario] = df

        return csv_data
    except FileNotFoundError:
        return None


csv_data = load_csv_data()


# Data Generation (Same Logic as Dashboard)
def create_scenario_data(scenario="normal"):
    """Create data using same logic as dashboard"""
    timestamp = datetime.now()

    if csv_data is None:
        return create_fallback_data(scenario, timestamp)

    scenario_data = csv_data.get(scenario)
    if scenario_data is None or len(scenario_data) == 0:
        return create_fallback_data(scenario, timestamp)

    if f"{scenario}_row_index" not in st.session_state:
        st.session_state[f"{scenario}_row_index"] = 0

    row_index = st.session_state[f"{scenario}_row_index"]

    if row_index >= len(scenario_data):
        row_index = 0
        st.session_state[f"{scenario}_row_index"] = 0

    current_row = scenario_data.iloc[row_index]
    st.session_state[f"{scenario}_row_index"] = row_index + 1

    return {
        "timestamp": timestamp,
        "pressure": float(current_row.get("pressure", 35.0)),
        "flow_rate": float(current_row.get("flow_rate", 70.0)),
        "temperature": float(current_row.get("temperature", 5.0)),
        "valve_status": int(current_row.get("valve_status", 0)),
        "pump_state": int(current_row.get("pump_state", 0)),
        "pump_speed": float(current_row.get("pump_speed", 1000.0)),
        "compressor_state": float(current_row.get("compressor_state", 0.5)),
        "energy_consumption": float(current_row.get("energy_consumption", 25.0)),
        "alarm_triggered": int(current_row.get("alarm_triggered", 0)),
        "hour": timestamp.hour,
        "dayofweek": timestamp.weekday(),
    }


def create_fallback_data(scenario, timestamp):
    """Fallback data generation"""
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
        "dayofweek": timestamp.weekday(),
    }


# Feature Engineering (Same as Dashboard)
def create_features(df):
    """Apply EXACT same feature engineering as dashboard"""
    if len(df) < 2:
        return None

    df_fe = df.copy()
    sensor_cols = [
        "pressure",
        "flow_rate",
        "temperature",
        "pump_speed",
        "energy_consumption",
    ]

    # Lag features
    LAGS = [1, 3, 6, 12, 30, 60, 120, 360]
    for col in sensor_cols:
        for lag in LAGS:
            df_fe[f"{col}_lag{lag}"] = df_fe[col].shift(lag)

    # Rolling features
    WINDOWS = [30, 60, 120, 360]
    for col in sensor_cols:
        for w in WINDOWS:
            df_fe[f"{col}_rollmean{w}"] = (
                df_fe[col].shift(1).rolling(window=w, min_periods=1).mean()
            )
            df_fe[f"{col}_rollstd{w}"] = (
                df_fe[col].shift(1).rolling(window=w, min_periods=1).std()
            )
            df_fe[f"{col}_rollmin{w}"] = (
                df_fe[col].shift(1).rolling(window=w, min_periods=1).min()
            )
            df_fe[f"{col}_rollmax{w}"] = (
                df_fe[col].shift(1).rolling(window=w, min_periods=1).max()
            )

    # Time features
    df_fe["hour"] = df_fe["timestamp"].dt.hour
    df_fe["dayofweek"] = df_fe["timestamp"].dt.dayofweek

    df_fe = df_fe.fillna(method="ffill").fillna(method="bfill").fillna(0)

    if model is None or feature_columns is None:
        return None

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
    """Make prediction using the model (same as dashboard)"""
    try:
        if features is None or model is None:
            return 0, [0.8, 0.1, 0.1]

        probabilities = model.predict_proba(features)[0]

        # Same logic as dashboard
        if probabilities[2] > 0.4:  # Failure
            prediction = 2
        elif probabilities[1] > 0.4:  # Warning
            prediction = 1
        else:
            prediction = 0

        return prediction, probabilities
    except Exception:
        return 0, [0.8, 0.1, 0.1]


# Initialize Session State
if "data_buffer" not in st.session_state:
    st.session_state.data_buffer = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "current_scenario" not in st.session_state:
    st.session_state.current_scenario = "normal"
if "last_update" not in st.session_state:
    st.session_state.last_update = datetime.now()


# Get Current System Data (Synchronized with Dashboard) - UPDATED
def get_current_system_data():
    """Get synchronized data from dashboard - FIXED VERSION"""

    try:
        is_fresh, shared_state_data = shared_state.is_state_fresh(max_age_seconds=20)

        if is_fresh and shared_state_data:
            st.sidebar.success("✅ Using dashboard data")

            st.session_state.data_buffer = shared_state_data["data_buffer"]
            st.session_state.current_scenario = shared_state_data["current_scenario"]

            for scenario, index in shared_state_data["row_indices"].items():
                st.session_state[f"{scenario}_row_index"] = index

            if st.session_state.data_buffer:
                df = pd.DataFrame(st.session_state.data_buffer)
                current_data = df.iloc[-1]

                prediction_data = shared_state_data["prediction_data"]
                prediction = prediction_data["prediction"]
                probabilities = np.array(prediction_data["probabilities"])

                st.sidebar.info(
                    f"📊 Synced: {len(st.session_state.data_buffer)} points"
                )
                return current_data, prediction, probabilities
        else:
            st.sidebar.warning("⚠️ No fresh dashboard data")

    except Exception as e:
        st.sidebar.error(f"❌ Sync error: {str(e)}")

    st.sidebar.info("🔄 Generating independent data")

    current_time = datetime.now()

    if (current_time - st.session_state.last_update).total_seconds() >= 10:
        new_point = create_scenario_data(st.session_state.current_scenario)
        st.session_state.data_buffer.append(new_point)

        if len(st.session_state.data_buffer) > 500:
            st.session_state.data_buffer = st.session_state.data_buffer[-500:]

        st.session_state.last_update = current_time

    if not st.session_state.data_buffer:
        new_point = create_scenario_data(st.session_state.current_scenario)
        st.session_state.data_buffer.append(new_point)

    df = pd.DataFrame(st.session_state.data_buffer)
    current_data = df.iloc[-1]

    features = create_features(df)
    prediction, probabilities = predict_with_model(features)

    return current_data, prediction, probabilities


llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.3,
    google_api_key=st.secrets["GOOGLE_API_KEY"],
)


# Enhanced Chatbot Logic (Same Intelligence as Dashboard)
def chatbot_response(query, current_data, prediction, probabilities):
    """Enhanced chatbot with same intelligence as dashboard"""
    # Keep original query for prompts, but normalize for keyword checks
    original_query = query
    query = query.lower()

    # Control whether the LLM handles all queries (slower) or only unknown/suggestive ones.
    # Default is False so quick-action buttons use the fast built-in handlers.
    USE_LLM_FOR_ALL = st.sidebar.checkbox(
        "Use LLM for all user questions",
        value=False,
        help=(
            "When enabled, every user query (including quick-action buttons) is sent to the LLM. "
            "When disabled, common metric queries use built-in fast answers while the LLM remains "
            "available for open questions or mitigation/suggestion requests."
        ),
    )
    if USE_LLM_FOR_ALL:
        try:
            prompt = (
                "You are a helpful, safety-conscious gas-pipeline maintenance assistant. A user asked:\n\n"
                f"{original_query}\n\n"
                "System snapshot:\n"
                f"- Status: { {0: 'NORMAL', 1: 'WARNING', 2: 'FAILURE'}[prediction] }\n"
                f"- Probabilities: normal={probabilities[0]:.3f}, warning={probabilities[1]:.3f}, failure={probabilities[2]:.3f}\n"
                f"- Pressure: {current_data.get('pressure', 'N/A')}\n"
                f"- Flow rate: {current_data.get('flow_rate', 'N/A')}\n"
                f"- Temperature: {current_data.get('temperature', 'N/A')}\n"
                f"- Energy consumption: {current_data.get('energy_consumption', 'N/A')}\n\n"
                "Respond concisely (1-4 sentences). If the user asks for steps or mitigation, include a short ordered list of immediate actions. Prioritize safety and clarity."
            )

            llm_result = llm.generate([[HumanMessage(content=prompt)]])
            if hasattr(llm_result, "generations"):
                try:
                    answer = llm_result.generations[0][0].text
                except Exception:
                    answer = str(llm_result)
            elif isinstance(llm_result, str):
                answer = llm_result
            elif hasattr(llm_result, "content"):
                answer = llm_result.content
            else:
                answer = str(llm_result)

            return f"<div style='font-size:16px; color:#F7F9F9; background:#1A3F66; padding:12px; border-radius:8px;'>{answer}</div>"
        except Exception as e:
            # Fall through to the built-in handlers if LLM fails
            st.sidebar.error(f"LLM error: {e}")
            query = query  # continue to handlers

    status_names = {0: "NORMAL", 1: "WARNING", 2: "FAILURE"}
    status_colors = {0: "🟢", 1: "🟡", 2: "🔴"}

    if "pressure" in query:
        pressure = current_data["pressure"]
        return f"<span style='font-size:18px; color:#5DADE2;'>📊 Current pressure is <b>{pressure:.1f} bar</b>. Status: {status_colors[prediction]} <b>{status_names[prediction]}</b></span>"

    elif "failure" in query:
        confidence = np.max(probabilities)
        health = (1 - probabilities[2]) * 100
        return f"<span style='font-size:18px; color:#E74C3C;'>🔮 <b>AI Prediction:</b> {status_colors[prediction]} <b>{status_names[prediction]}</b> (Confidence: {confidence:.1%}, Health: {health:.0f}%)</span>"

    elif "energy" in query:
        energy = current_data["energy_consumption"]
        return f"<span style='font-size:18px; color:#AF7AC5;'>🔋 Current energy consumption is <b>{energy:.1f} kWh</b>. System: {status_colors[prediction]} <b>{status_names[prediction]}</b></span>"

    elif "temperature" in query:
        temp = current_data["temperature"]
        return f"<span style='font-size:18px; color:#F39C12;'>🌡️ Current temperature is <b>{temp:.1f} °C</b>. Status: {status_colors[prediction]} <b>{status_names[prediction]}</b></span>"

    elif "flow" in query:
        flow = current_data["flow_rate"]
        return f"<span style='font-size:18px; color:#76D7C4;'>💧 Current flow rate is <b>{flow:.1f} m³/s</b>. System: {status_colors[prediction]} <b>{status_names[prediction]}</b></span>"

    elif any(word in query for word in ["status", "system", "overview"]):
        confidence = np.max(probabilities)
        health = (1 - probabilities[2]) * 100
        return f"""<span style='font-size:18px; color:#58D68D;'>📋 <b>System Overview:</b><br>
        • Status: {status_colors[prediction]} <b>{status_names[prediction]}</b><br>
        • Confidence: <b>{confidence:.1%}</b><br>
        • Health: <b>{health:.0f}%</b><br>
        • Pressure: <b>{current_data["pressure"]:.1f} bar</b><br>
        • Flow: <b>{current_data["flow_rate"]:.1f} m³/s</b><br>
        • Temperature: <b>{current_data["temperature"]:.1f} °C</b><br>
        • Energy: <b>{current_data["energy_consumption"]:.1f} kWh</b></span>"""

    # If user asks for help, suggestions, or a solution, call the LLM with context
    suggest_keywords = [
        "recommend",
        "suggest",
        "solution",
        "advice",
        "what should",
        "how to",
        "mitigation",
        "steps",
        "fix",
    ]
    if any(k in query for k in suggest_keywords):
        try:
            # Build a concise context prompt for the LLM
            prompt = (
                "You are an expert gas-pipeline maintenance engineer. Provide a short, actionable diagnosis "
                "and step-by-step remediation plan based on the system snapshot below. Be concise and prioritize safety.\n\n"
                "System Snapshot:\n"
                f"- Status: {status_names[prediction]}\n"
                f"- Probabilities: normal={probabilities[0]:.3f}, warning={probabilities[1]:.3f}, failure={probabilities[2]:.3f}\n"
                f"- Pressure: {current_data.get('pressure', 'N/A'):.1f} bar\n"
                f"- Flow rate: {current_data.get('flow_rate', 'N/A'):.1f} m³/s\n"
                f"- Temperature: {current_data.get('temperature', 'N/A'):.1f} °C\n"
                f"- Energy consumption: {current_data.get('energy_consumption', 'N/A'):.1f} kWh\n"
                f"- Pump speed: {current_data.get('pump_speed', 'N/A'):.1f}\n\n"
                "Deliver:\n1) One-line diagnosis.\n2) Top 3 immediate actions (ordered).\n3) Suggested monitoring or tests to run.\n"
            )

            llm_result = llm.generate([[HumanMessage(content=prompt)]])
            if hasattr(llm_result, "generations"):
                try:
                    answer = llm_result.generations[0][0].text
                except Exception:
                    answer = str(llm_result)
            elif isinstance(llm_result, str):
                answer = llm_result
            elif hasattr(llm_result, "content"):
                answer = llm_result.content
            else:
                answer = str(llm_result)

            return f"<div style='font-size:16px; color:#F7F9F9; background:#1A3F66; padding:12px; border-radius:8px;'>{answer}</div>"
        except Exception as e:
            return f"<span style='font-size:18px; color:#F1948A;'>⚠️ Unable to reach LLM: {str(e)}. Try again later.</span>"

    # Default fallback: provide quick guidance or ask user to be more specific
    try:
        # Use LLM to help rephrase or answer unknown queries briefly
        prompt = (
            "You are a friendly assistant for a gas-pipeline operator. A user asked: '"
            + query
            + "'.\n"
            "Provide a short helpful reply (1-3 sentences). If the question requires system context, include a short checklist of next steps.\n\n"
            "System context:\n"
            f"Status={status_names[prediction]}, probabilities={probabilities.tolist()}, pressure={current_data.get('pressure', 'N/A')}, flow={current_data.get('flow_rate', 'N/A')}\n"
        )

        llm_result = llm.generate([[HumanMessage(content=prompt)]])
        if hasattr(llm_result, "generations"):
            try:
                answer = llm_result.generations[0][0].text
            except Exception:
                answer = str(llm_result)
        elif isinstance(llm_result, str):
            answer = llm_result
        elif hasattr(llm_result, "content"):
            answer = llm_result.content
        else:
            answer = str(llm_result)

        return f"<div style='font-size:16px; color:#F7F9F9; background:#1A3F66; padding:12px; border-radius:8px;'>{answer}</div>"
    except Exception:
        return f"<span style='font-size:18px; color:#F1948A;'>❓ Sorry, I didn't understand. Try asking about pressure, failure, temperature, energy, flow, or system status. Current: {status_colors[prediction]} <b>{status_names[prediction]}</b></span>"


# Title
st.markdown(
    "<h1 style='color:white;'>🤖 Smart Maintenance Assistant</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    "<p style='color:gold;'>Ask me about <b>Pressure</b>, <b>Temperature</b>, <b>Energy</b>, <b>Flow</b>, or <b>Failure prediction</b> </p>",
    unsafe_allow_html=True,
)

# Chat History
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Get Live Data (Synchronized with Dashboard)
current_data, prediction, probabilities = get_current_system_data()

st.markdown(
    """
<style>
div.stButton > button {
    background-color: #FFD700;  
    color: #4B2E05;             
    border-radius: 8px;
    padding: 0.35em 0.75em;
    font-weight: bold;
    transition: 0.3s;
}
div.stButton > button:hover {
    background-color: #FFC200;  
    color: #4B2E05;
}
input[type="text"] {
    background-color: #FFF8DC;  
    color: #4B2E05;
    border: 1px solid #BFC9CA;
    padding: 8px;
    border-radius: 8px;
}
</style>
""",
    unsafe_allow_html=True,
)

# Quick Actions
st.subheader("🚀 Quick Actions")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    if st.button("📊 System Status"):
        reply = chatbot_response(
            "system status", current_data, prediction, probabilities
        )
        st.session_state.chat_history.append(("🤖", reply))

with col2:
    if st.button("🔮 Failure Prediction"):
        reply = chatbot_response(
            "failure prediction", current_data, prediction, probabilities
        )
        st.session_state.chat_history.append(("🤖", reply))

with col3:
    if st.button("📈 Pressure"):
        reply = chatbot_response("pressure", current_data, prediction, probabilities)
        st.session_state.chat_history.append(("🤖", reply))

with col4:
    if st.button("🌡️ Temperature"):
        reply = chatbot_response("temperature", current_data, prediction, probabilities)
        st.session_state.chat_history.append(("🤖", reply))

with col5:
    if st.button("🔋 Energy"):
        reply = chatbot_response("energy", current_data, prediction, probabilities)
        st.session_state.chat_history.append(("🤖", reply))


def _handle_user_input():
    """Callback for `st.text_input` on_change: process and clear input safely."""
    ui = st.session_state.get("user_input", "")
    if not ui:
        return

    reply = chatbot_response(ui, current_data, prediction, probabilities)
    st.session_state.chat_history.append(
        ("👤", f"<span style='font-size:18px; color:#1A3F66;'>{ui}</span>")
    )
    st.session_state.chat_history.append(("🤖", reply))

    # Clear the input for the next message — this is safe inside the callback
    st.session_state["user_input"] = ""


# User Input (uses on_change to avoid modifying session_state after widget creation)
st.text_input("💬 Your question:", key="user_input", on_change=_handle_user_input)

# Display Chat History (Same as Original)
st.markdown("---")
st.subheader("📜 Chat History")

for role, msg in reversed(st.session_state.chat_history):
    if role == "👤":
        st.markdown(
            f"<div style='background-color:#89CFF0; padding:12px; border-radius:15px; "
            f"text-align:right; margin:8px; font-size:16px; color:#FDF5E6;'>"
            f"<b>{role}:</b> {msg}</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"<div style='background-color:#1A3F66; padding:12px; border-radius:15px; "
            f"text-align:left; margin:8px; font-size:16px; color:black;'>"
            f"<b>{role}:</b> {msg}</div>",
            unsafe_allow_html=True,
        )

# Clear Chat (Same as Original)
if st.button("🗑️ Clear Chat"):
    st.session_state.chat_history = []
    st.rerun()

# Sidebar Sync Status (UPDATED to Google Sheets)
with st.sidebar:
    st.markdown("---")
    st.write("### 🔄 Sync Status")

    try:
        is_fresh, state_data = shared_state.is_state_fresh(max_age_seconds=20)
        if is_fresh and state_data:
            st.success("✅ Data is fresh")
            st.write(f"Points: {len(state_data.get('data_buffer', []))}")
            last_update = state_data["last_update"]
            age = (datetime.now() - datetime.fromisoformat(last_update)).total_seconds()
            st.write(f"Last update: {age:.1f}s ago")
        else:
            st.warning("⚠️ No fresh dashboard data")
    except Exception as e:
        st.error(f"❌ Error reading shared state: {e}")

    if st.button("🔄 Force Refresh"):
        st.session_state.data_buffer = []
        st.rerun()

# Scenario Selector (Hidden but functional)
if "scenario" in st.session_state:
    st.session_state.current_scenario = st.session_state.scenario

# Custom Sidebar Style
st.markdown(
    """
    <style>
    /* Sidebar container */
    [data-testid="stSidebar"] {
        background-color: #0E4D92; 
    }

    /* Sidebar text */
    [data-testid="stSidebar"] * {
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
