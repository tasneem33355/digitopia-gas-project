import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib
import shared_state
import os
import json
import re

import langchain
import langchain_google_genai

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
import SYSTEM_PROMPT
from pydantic import BaseModel, Field

try:
    from langchain_core.output_parsers import PydanticOutputParser
except Exception:
    PydanticOutputParser = None


# JSON validator for agent responses
REQUIRED_KEYS = {
    "fault_type",
    "confidence",
    "summary",
    "immediate_actions",
    "preventive_tips",
    "recommended_checks",
    "escalation_required",
    "escalation_steps",
}


def parse_and_validate_json(text: str):
    """Try to parse `text` as JSON and ensure required keys exist.

    Returns (obj, None) on success or (None, error_string) on failure.
    """
    try:
        obj = json.loads(text)
    except Exception:
        return None, "invalid_json"

    if not isinstance(obj, dict):
        return None, "not_object"

    missing = REQUIRED_KEYS - set(obj.keys())
    if missing:
        return None, f"missing_keys:{sorted(list(missing))}"

    return obj, None


def enforce_json_from_agent(initial_prompt: str, max_retries: int = 1):
    """Request a JSON object from `agent` using `initial_prompt` and validate it.

    If the first response is invalid, attempt `max_retries` corrective re-prompts.
    Returns a tuple (obj, raw_text, error) where `obj` is the parsed dict or None.
    """
    # Try to use a PydanticOutputParser for stricter format enforcement when available
    parser = None
    if PydanticOutputParser is not None:

        class DiagnosisSchema(BaseModel):
            fault_type: str = Field(...)
            confidence: float = Field(...)
            summary: str = Field(...)
            immediate_actions: list[str] = Field(...)
            preventive_tips: list[str] = Field(...)
            recommended_checks: list[str] = Field(...)
            escalation_required: bool = Field(...)
            escalation_steps: list[str] = Field(...)

        try:
            parser = PydanticOutputParser(pydantic_object=DiagnosisSchema)
            format_instructions = parser.get_format_instructions()
            initial_prompt = format_instructions + "\n\n" + initial_prompt
        except Exception:
            parser = None

    # Obtain initial raw response (prefer agent, fallback to low-level LLM)
    try:
        raw = agent.run(initial_prompt)
    except Exception as e:
        try:
            llm_result = llm.generate([[HumanMessage(content=initial_prompt)]])
            if hasattr(llm_result, "generations"):
                raw = llm_result.generations[0][0].text
            elif isinstance(llm_result, str):
                raw = llm_result
            elif hasattr(llm_result, "content"):
                raw = llm_result.content
            else:
                raw = str(llm_result)
        except Exception as e2:
            return None, "", f"agent_and_llm_failed: {e}; {e2}"

    # Helper to attempt parsing a raw string using parser then fallback validator
    def try_parse(raw_text: str):
        if not raw_text:
            return None, "empty"

        if parser is not None:
            try:
                parsed_obj = parser.parse(raw_text)
                if hasattr(parsed_obj, "dict"):
                    return parsed_obj.dict(), None
                if isinstance(parsed_obj, dict):
                    return parsed_obj, None
            except Exception:
                pass

        obj, err = parse_and_validate_json(raw_text)
        if obj is not None:
            return obj, None
        return None, err

    obj, err = try_parse(raw)
    if obj is not None:
        return obj, raw, None

    # Try corrective re-prompts
    for _ in range(max_retries):
        correction_prompt = (
            "Your previous reply was not valid JSON following the required schema.\n"
            "Return only a single JSON object following this schema exactly:\n"
            '{\n  "fault_type": string,\n  "confidence": float,\n  "summary": string,\n  "immediate_actions": [string],\n  "preventive_tips": [string],\n  "recommended_checks": [string],\n  "escalation_required": boolean,\n  "escalation_steps": [string]\n}\n\n'
            "Previous reply:\n" + raw + "\n\nNow provide the corrected JSON only."
        )

        try:
            raw2 = agent.run(correction_prompt)
        except Exception:
            try:
                llm_result = llm.generate([[HumanMessage(content=correction_prompt)]])
                if hasattr(llm_result, "generations"):
                    raw2 = llm_result.generations[0][0].text
                elif isinstance(llm_result, str):
                    raw2 = llm_result
                elif hasattr(llm_result, "content"):
                    raw2 = llm_result.content
                else:
                    raw2 = str(llm_result)
            except Exception as e:
                return None, raw, f"reprompt_failed: {e}"

        obj2, err2 = try_parse(raw2)
        if obj2 is not None:
            return obj2, raw2, None

        # prepare for next retry
        raw = raw2
        err = err2

    return None, raw, err


def render_diagnosis_html(obj: dict) -> str:
    """Render the validated diagnosis JSON as structured HTML."""
    fault = obj.get("fault_type", "unknown")
    confidence = obj.get("confidence", 0.0)
    summary = obj.get("summary", "")
    immediate = obj.get("immediate_actions", []) or []
    preventive = obj.get("preventive_tips", []) or []
    checks = obj.get("recommended_checks", []) or []
    escalation_required = obj.get("escalation_required", False)
    escalation_steps = obj.get("escalation_steps", []) or []

    def render_list(items):
        if not items:
            return "<li>None</li>"
        return "".join(f"<li>{str(i)}</li>" for i in items)

    html = f"""
<div style='background:#1A3F66;color:#F7F9F9;padding:14px;border-radius:10px;'>
    <h3 style='margin:0 0 8px 0;'>🔎 Diagnosis: <span style='color:#FFD700'>{fault}</span></h3>
    <p style='margin:0 0 8px 0;'><strong>Confidence:</strong> {confidence:.2f} — <strong>Summary:</strong> {summary}</p>
    <div style='display:flex; gap:18px; align-items:flex-start;'>
        <div style='flex:1;'>
            <h4 style='margin:8px 0 6px 0;'>Immediate Actions</h4>
            <ol style='margin:0 0 8px 18px;'>{render_list(immediate)}</ol>
        </div>
        <div style='flex:1;'>
            <h4 style='margin:8px 0 6px 0;'>Preventive Tips</h4>
            <ul style='margin:0 0 8px 18px;'>{render_list(preventive)}</ul>
        </div>
        <div style='flex:1;'>
            <h4 style='margin:8px 0 6px 0;'>Recommended Checks</h4>
            <ul style='margin:0 0 8px 18px;'>{render_list(checks)}</ul>
        </div>
    </div>
    <p style='margin:6px 0 0 0;'><strong>Escalation Required:</strong> {"Yes" if escalation_required else "No"}</p>
    <div style='margin-top:6px;'>
        <strong>Escalation Steps:</strong>
        <ol style='margin:6px 0 0 18px;'>{render_list(escalation_steps)}</ol>
    </div>
</div>
"""
    return html


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


def _extract_json_from_text(text: str):
    """Try to extract the first JSON object found in `text`.

    Returns the parsed object or None if parsing fails.
    """
    if not isinstance(text, str):
        return None

    # Remove markdown fences
    cleaned = re.sub(r"```[\s\S]*?```", lambda m: m.group(0).strip("`"), text)

    # Find first { ... } block
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None

    candidate = cleaned[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        # Try tolerant parsing by replacing single quotes
        try:
            candidate2 = candidate.replace("'", '"')
            return json.loads(candidate2)
        except Exception:
            return None


def _normalize_assistant_json(d: dict):
    """Normalize/complete the assistant JSON into a conforming schema.

    Expected schema keys:
      - fault_type (str)
      - confidence (float)
      - summary (str)
      - immediate_actions (list[str])
      - preventive_tips (list[str])
      - recommended_checks (list[str])
      - escalation_required (bool)
      - escalation_steps (list[str])
    """
    if not isinstance(d, dict):
        return None

    out = {}
    out["fault_type"] = str(d.get("fault_type") or d.get("status") or "None")
    try:
        out["confidence"] = float(d.get("confidence", 1.0))
    except Exception:
        out["confidence"] = 1.0

    out["summary"] = str(
        d.get("summary") or d.get("diagnosis") or "No summary provided"
    )

    def to_list(k):
        v = d.get(k) or []
        if isinstance(v, str):
            return [v]
        if isinstance(v, list):
            return [str(x) for x in v]
        return []

    out["immediate_actions"] = to_list("immediate_actions")
    out["preventive_tips"] = to_list("preventive_tips")
    out["recommended_checks"] = to_list("recommended_checks")
    out["escalation_required"] = bool(d.get("escalation_required", False))
    out["escalation_steps"] = to_list("escalation_steps")

    return out


def _wrap_with_badge(html: str, status: str) -> str:
    """Prepend a small colored badge indicating parse status: parsed / repaired / raw."""
    colors = {
        "parsed": "#58D68D",
        "repaired": "#F7DC6F",
        "raw": "#F1948A",
    }
    labels = {"parsed": "Parsed", "repaired": "Repaired", "raw": "Raw"}
    color = colors.get(status, "#D3D3D3")
    label = labels.get(status, status)
    badge = f"<div style='display:inline-block;background:{color};color:#000;padding:4px 8px;border-radius:6px;margin-bottom:8px;font-weight:600;'>{label}</div>"
    return badge + html


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
    model="gemini-2.5-flash-lite",
    temperature=0.3,
    google_api_key=st.secrets["GOOGLE_API_KEY"],
    max_output_tokens=None,
)
agent = create_agent(model=llm, system_prompt=SYSTEM_PROMPT.system_prompt)

# Persistent sidebar control: let user choose whether LLM should handle all queries.
# Placing this at top-level ensures the checkbox is always rendered and its
# value is available in `st.session_state` across reruns.
st.sidebar.checkbox(
    "Use LLM for all user questions",
    key="use_llm_for_all",
    value=False,
    help=(
        "When enabled, every user query (including quick-action buttons) is sent to the LLM. "
        "When disabled, common metric queries use built-in fast answers while the LLM remains "
        "available for open questions or mitigation/suggestion requests."
    ),
)


# Enhanced Chatbot Logic (Same Intelligence as Dashboard)
def chatbot_response(query, current_data, prediction, probabilities):
    """Enhanced chatbot with same intelligence as dashboard"""
    # Keep original query for prompts, but normalize for keyword checks
    original_query = query
    query = query.lower()

    # Control whether the LLM handles all queries (slower) or only unknown/suggestive ones.
    # Read the persistent sidebar checkbox value placed in `st.session_state`.
    USE_LLM_FOR_ALL = st.session_state.get("use_llm_for_all", False)
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

            # Use the agent wrapper (created earlier) instead of calling the
            # low-level LLM generate API directly. `agent.run` returns a
            # concise string response for the provided prompt.
            # Request a validated JSON response from the agent
            obj, raw, err = enforce_json_from_agent(prompt, max_retries=1)
            if obj is not None:
                # Render the validated JSON as structured HTML
                return render_diagnosis_html(obj)
            else:
                # If validation failed, show the raw agent text (with warning)
                warning = "⚠️ Assistant returned non-conforming JSON. Showing raw response below."
                safe_raw = (raw or "").replace("<", "&lt;").replace(">", "&gt;")
                return f"<div style='color:#F1948A;'>{warning}</div><pre style='background:#1A3F66;color:#F7F9F9;padding:12px;border-radius:8px;'>{safe_raw}</pre>"

        except Exception as e:
            # Fall through to the built-in handlers if LLM fails
            st.sidebar.error(f"LLM error: {e}")
            query = query  # continue to handlers

    status_names = {0: "NORMAL", 1: "WARNING", 2: "FAILURE"}
    status_colors = {0: "🟢", 1: "🟡", 2: "🔴"}

    if "pressure" == query:
        pressure = current_data["pressure"]
        return f"<span style='font-size:18px; color:#5DADE2;'>📊 Current pressure is <b>{pressure:.1f} bar</b>. Status: {status_colors[prediction]} <b>{status_names[prediction]}</b></span>"

    elif "failure" == query:
        confidence = np.max(probabilities)
        health = (1 - probabilities[2]) * 100
        return f"<span style='font-size:18px; color:#E74C3C;'>🔮 <b>AI Prediction:</b> {status_colors[prediction]} <b>{status_names[prediction]}</b> (Confidence: {confidence:.1%}, Health: {health:.0f}%)</span>"

    elif "energy" == query:
        energy = current_data["energy_consumption"]
        return f"<span style='font-size:18px; color:#AF7AC5;'>🔋 Current energy consumption is <b>{energy:.1f} kWh</b>. System: {status_colors[prediction]} <b>{status_names[prediction]}</b></span>"

    elif "temperature" == query:
        temp = current_data["temperature"]
        return f"<span style='font-size:18px; color:#F39C12;'>🌡️ Current temperature is <b>{temp:.1f} °C</b>. Status: {status_colors[prediction]} <b>{status_names[prediction]}</b></span>"

    elif "flow" == query:
        flow = current_data["flow_rate"]
        return f"<span style='font-size:18px; color:#76D7C4;'>💧 Current flow rate is <b>{flow:.1f} m³/s</b>. System: {status_colors[prediction]} <b>{status_names[prediction]}</b></span>"

    elif any(word == query.lower() for word in ["system status", "overview"]):
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

            obj, raw, err = enforce_json_from_agent(prompt, max_retries=1)
            if obj is not None:
                return _wrap_with_badge(render_diagnosis_html(obj), "parsed")
            else:
                # Attempt best-effort repair: extract JSON-like text and normalize to schema
                repaired_html = None
                if raw:
                    parsed = _extract_json_from_text(raw)
                    if parsed:
                        normalized = _normalize_assistant_json(parsed)
                        if normalized:
                            note = (
                                "<div style='color:#F7DC6F;'>⚠️ Assistant returned non-conforming JSON; "
                                "showing a best-effort repaired version.</div>"
                            )
                            repaired_html = note + render_diagnosis_html(normalized)

                if repaired_html:
                    return _wrap_with_badge(repaired_html, "repaired")

                warning = "⚠️ Assistant returned non-conforming JSON. Showing raw response below."
                safe_raw = (raw or "").replace("<", "&lt;").replace(">", "&gt;")
                raw_block = f"<div style='color:#F1948A;'>{warning}</div><pre style='background:#1A3F66;color:#F7F9F9;padding:12px;border-radius:8px;'>{safe_raw}</pre>"
                return _wrap_with_badge(raw_block, "raw")
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

        obj, raw, err = enforce_json_from_agent(prompt, max_retries=1)
        if obj is not None:
            return _wrap_with_badge(render_diagnosis_html(obj), "parsed")
        else:
            repaired_html = None
            if raw:
                parsed = _extract_json_from_text(raw)
                if parsed:
                    normalized = _normalize_assistant_json(parsed)
                    if normalized:
                        note = (
                            "<div style='color:#F7DC6F;'>⚠️ Assistant returned non-conforming JSON; "
                            "showing a best-effort repaired version.</div>"
                        )
                        repaired_html = note + render_diagnosis_html(normalized)

            if repaired_html:
                return _wrap_with_badge(repaired_html, "repaired")

            warning = (
                "⚠️ Assistant returned non-conforming JSON. Showing raw response below."
            )
            safe_raw = (raw or "").replace("<", "&lt;").replace(">", "&gt;")
            raw_block = f"<div style='color:#F1948A;'>{warning}</div><pre style='background:#1A3F66;color:#F7F9F9;padding:12px;border-radius:8px;'>{safe_raw}</pre>"
            return _wrap_with_badge(raw_block, "raw")
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
