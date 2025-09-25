import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import shared_state  

# ===============================
# Config Streamlit Page
# ===============================
st.set_page_config(
    page_title="Maintenance Chatbot",
    page_icon="🤖",
    layout="wide"
)

# ===============================
# Custom Background (same as before)
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

# ===============================
# Load Model & Feature List
# ===============================
# ... (keep as before)

# ===============================
# Get Current System Data (modified)
# ===============================
def get_current_system_data():
    """Get synchronized data from dashboard using Google Sheets"""
    try:
        is_fresh, shared_state_data = shared_state.is_state_fresh(max_age_seconds=20)
        
        if is_fresh and shared_state_data:
            st.sidebar.success("✅ Using dashboard data")
            
            st.session_state.data_buffer = shared_state_data['data_buffer']
            st.session_state.current_scenario = shared_state_data['current_scenario']
            
            for scenario, index in shared_state_data['row_indices'].items():
                st.session_state[f'{scenario}_row_index'] = index
            
            if st.session_state.data_buffer:
                df = pd.DataFrame(st.session_state.data_buffer)
                current_data = df.iloc[-1]
                
                prediction_data = shared_state_data['prediction_data']
                prediction = prediction_data['prediction']
                probabilities = np.array(prediction_data['probabilities'])
                
                st.sidebar.info(f"📊 Synced: {len(st.session_state.data_buffer)} points")
                return current_data, prediction, probabilities
        else:
            st.sidebar.warning("⚠️ No fresh dashboard data")
    except Exception as e:
        st.sidebar.error(f"❌ Sync error: {str(e)}")
    
    st.sidebar.info("🔄 Generating independent data")
    # fallback data generation remains unchanged
    # ...

# ===============================
# Sidebar Sync Status (modified)
# ===============================
with st.sidebar:
    st.markdown("---")
    st.write("### 🔄 Sync Status")
    
    try:
        is_fresh, state_data = shared_state.is_state_fresh(max_age_seconds=20)
        if is_fresh:
            st.success("✅ Data is fresh")
            st.write(f"Points: {len(state_data.get('data_buffer', []))}")
        else:
            st.warning("⚠️ No fresh dashboard data")
    except Exception as e:
        st.error(f"❌ Error reading shared state: {e}")
    
    if st.button("🔄 Force Refresh"):
        st.session_state.data_buffer = []
        st.rerun()

# ===============================
# Rest of the chatbot UI/logic
# ===============================
# ... (keep all chatbot_response, UI, quick actions, chat history the same)
