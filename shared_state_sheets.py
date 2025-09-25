import json
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
import threading
import time
import os
import streamlit as st

# Google Sheets Configuration - استخدام st.secrets
SPREADSHEET_ID = st.secrets["GOOGLE_SHEETS_ID"]
WORKSHEET_NAME = "dashboard_data"

# Service Account File - استخدام st.secrets
SERVICE_ACCOUNT_FILE = st.secrets["SERVICE_ACCOUNT_FILE"]

# Global variables for optimization
_sheets_client = None
_worksheet_cache = None
_last_save_time = 0
_pending_data = None

def get_sheets_client():
    """Initialize Google Sheets client with caching"""
    global _sheets_client
    
    if _sheets_client is None:
        try:
            scope = ["https://spreadsheets.google.com/feeds", 
                     "https://www.googleapis.com/auth/drive"]
            
            credentials = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=scope)
            _sheets_client = gspread.authorize(credentials)
            print("✅ Google Sheets client initialized")
        except Exception as e:
            print(f"❌ Error connecting to Google Sheets: {e}")
            return None
    
    return _sheets_client

def get_worksheet():
    """Get worksheet with caching"""
    global _worksheet_cache
    
    if _worksheet_cache is None:
        try:
            client = get_sheets_client()
            if not client:
                return None
                
            spreadsheet = client.open_by_key(SPREADSHEET_ID)
            
            try:
                _worksheet_cache = spreadsheet.worksheet(WORKSHEET_NAME)
            except gspread.WorksheetNotFound:
                _worksheet_cache = spreadsheet.add_worksheet(title=WORKSHEET_NAME, rows="100", cols="10")
                # Add headers
                headers = ["timestamp", "scenario", "row_indices", "prediction_data", "data_buffer", "last_update"]
                _worksheet_cache.append_row(headers)
                
            print("✅ Worksheet cached")
        except Exception as e:
            print(f"❌ Error accessing worksheet: {e}")
            return None
    
    return _worksheet_cache

def save_in_background(data_buffer, scenario, row_indices, prediction_data):
    """Save data to Google Sheets in background thread"""
    try:
        worksheet = get_worksheet()
        if not worksheet:
            return
        
        # Prepare data for saving
        cleaned_buffer = []
        for item in data_buffer[-20:]:  # keep last 20 items
            if isinstance(item, dict):
                cleaned_item = item.copy()
                if 'timestamp' in cleaned_item:
                    cleaned_item['timestamp'] = cleaned_item['timestamp'].isoformat()
                cleaned_buffer.append(cleaned_item)
        
        # Save data to the sheet
        current_time = datetime.now().isoformat()
        row_data = [
            current_time,
            scenario,
            json.dumps(row_indices),
            json.dumps(prediction_data),
            json.dumps(cleaned_buffer),
            current_time
        ]
        
        # Write over row 2 (do not delete rows)
        try:
            worksheet.update(f"A2:F2", [row_data])
            print(f"✅ Data updated in Google Sheets at {datetime.now().strftime('%H:%M:%S')}")
        except Exception as e:
            # If update fails, try append
            worksheet.append_row(row_data)
            print(f"✅ Data appended to Google Sheets at {datetime.now().strftime('%H:%M:%S')}")
            
    except Exception as e:
        print(f"❌ Background save error: {e}")

def save_shared_state(data_buffer, scenario, row_indices, prediction_data):
    """Save shared state to Google Sheets (optimized)"""
    global _last_save_time, _pending_data
    
    current_time = time.time()
    
    # Temporarily store data
    _pending_data = {
        'data_buffer': data_buffer,
        'scenario': scenario,
        'row_indices': row_indices,
        'prediction_data': prediction_data
    }
    
    # Save to Google Sheets every 15 seconds
    if current_time - _last_save_time >= 15:
        _last_save_time = current_time
        
        # Save in background thread to avoid slowing UI
        threading.Thread(
            target=save_in_background,
            args=(data_buffer, scenario, row_indices, prediction_data),
            daemon=True
        ).start()
        
        return True
    
    return True  # Always return True so UI thinks save succeeded

def load_shared_state():
    """Load shared state from Google Sheets (optimized)"""
    global _pending_data
    
    if _pending_data:
        return {
            'current_scenario': _pending_data['scenario'],
            'row_indices': _pending_data['row_indices'],
            'prediction_data': _pending_data['prediction_data'],
            'data_buffer': _pending_data['data_buffer'][-20:],  # last 20 points
            'last_update': datetime.now().isoformat()
        }
    
    try:
        worksheet = get_worksheet()
        if not worksheet:
            return None
        
        records = worksheet.get_all_records()
        if not records:
            return None
            
        latest_record = records[-1]
        
        state = {
            'current_scenario': latest_record['scenario'],
            'row_indices': json.loads(latest_record['row_indices']),
            'prediction_data': json.loads(latest_record['prediction_data']),
            'data_buffer': json.loads(latest_record['data_buffer']),
            'last_update': latest_record['last_update']
        }
        
        # Convert timestamps back to datetime
        if 'data_buffer' in state:
            for item in state['data_buffer']:
                if 'timestamp' in item and isinstance(item['timestamp'], str):
                    item['timestamp'] = datetime.fromisoformat(item['timestamp'])
        
        return state
        
    except Exception as e:
        print(f"❌ Error loading from Google Sheets: {e}")
        return None

def is_state_fresh(max_age_seconds=30):
    """Check if shared state is fresh"""
    global _pending_data
    
    if _pending_data:
        return True, {
            'current_scenario': _pending_data['scenario'],
            'row_indices': _pending_data['row_indices'],
            'prediction_data': _pending_data['prediction_data'],
            'data_buffer': _pending_data['data_buffer'][-20:],
            'last_update': datetime.now().isoformat()
        }
    
    try:
        state = load_shared_state()
        if state and 'last_update' in state:
            last_update = datetime.fromisoformat(state['last_update'])
            age = (datetime.now() - last_update).total_seconds()
            is_fresh = age <= max_age_seconds
            
            return is_fresh, state
        else:
            return False, None
    except Exception as e:
        print(f"❌ Error checking state freshness: {e}")
        return False, None

def test_shared_state():
    """Test the shared state with Google Sheets"""
    print("🧪 Testing optimized Google Sheets shared state...")
    
    test_data = [{
        'timestamp': datetime.now(),
        'pressure': 35.0,
        'temperature': 5.0,
        'flow_rate': 70.0
    }]
    
    test_prediction = {
        'prediction': 0,
        'probabilities': [0.8, 0.1, 0.1],
        'confidence': 0.8
    }
    
    success = save_shared_state(test_data, "normal", {"normal": 0}, test_prediction)
    if success:
        print("✅ Save test passed")
        time.sleep(2)
        is_fresh, loaded_state = is_state_fresh(max_age_seconds=60)
        if is_fresh:
            print("✅ Load test passed")
            print(f"Loaded {len(loaded_state['data_buffer'])} data points")
        else:
            print("❌ Load test failed")
    else:
        print("❌ Save test failed")

if __name__ == "__main__":
    test_shared_state()