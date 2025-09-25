import json
import os
from datetime import datetime
import gspread
from google.oauth2.service_account import Credentials
import threading
import time

# Global variables for optimization
_sheets_client = None
_worksheet_cache = None
_last_save_time = 0
_pending_data = None

# Configuration
WORKSHEET_NAME = "dashboard_data"
SHARED_STATE_FILE = "shared_state.json"

def get_secrets():
    """Get secrets safely - only when streamlit is ready"""
    try:
        import streamlit as st
        return st.secrets
    except Exception as e:
        print(f"Warning: Could not access streamlit secrets: {e}")
        # Fallback - return None so app can work without Google Sheets
        return None

def get_sheets_client():
    """Initialize Google Sheets client with caching"""
    global _sheets_client
    
    if _sheets_client is None:
        try:
            secrets = get_secrets()
            if not secrets:
                print("❌ No secrets available - Google Sheets disabled")
                return None
            
            scope = ["https://spreadsheets.google.com/feeds", 
                     "https://www.googleapis.com/auth/drive"]
            
            service_account_info = {
                "type": secrets["SERVICE_ACCOUNT"]["type"],
                "project_id": secrets["SERVICE_ACCOUNT"]["project_id"],
                "private_key_id": secrets["SERVICE_ACCOUNT"]["private_key_id"],
                "private_key": secrets["SERVICE_ACCOUNT"]["private_key"],
                "client_email": secrets["SERVICE_ACCOUNT"]["client_email"],
                "client_id": secrets["SERVICE_ACCOUNT"]["client_id"],
                "auth_uri": secrets["SERVICE_ACCOUNT"]["auth_uri"],
                "token_uri": secrets["SERVICE_ACCOUNT"]["token_uri"],
                "auth_provider_x509_cert_url": secrets["SERVICE_ACCOUNT"]["auth_provider_x509_cert_url"],
                "client_x509_cert_url": secrets["SERVICE_ACCOUNT"]["client_x509_cert_url"],
                "universe_domain": secrets["SERVICE_ACCOUNT"]["universe_domain"]
            }
            
            credentials = Credentials.from_service_account_info(service_account_info, scopes=scope)
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
            secrets = get_secrets()
            if not secrets:
                return None
                
            client = get_sheets_client()
            if not client:
                return None
                
            spreadsheet_id = secrets["GOOGLE_SHEETS_ID"]
            spreadsheet = client.open_by_key(spreadsheet_id)
            
            try:
                _worksheet_cache = spreadsheet.worksheet(WORKSHEET_NAME)
            except gspread.WorksheetNotFound:
                _worksheet_cache = spreadsheet.add_worksheet(title=WORKSHEET_NAME, rows="100", cols="10")
                # إضافة headers
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
            print("⚠️ No worksheet available - skipping save")
            return
        
        # تحضير البيانات للحفظ
        cleaned_buffer = []
        for item in data_buffer[-20:]:  # قلل العدد لـ 20 بدل 50
            if isinstance(item, dict):
                cleaned_item = item.copy()
                if 'timestamp' in cleaned_item:
                    cleaned_item['timestamp'] = cleaned_item['timestamp'].isoformat()
                cleaned_buffer.append(cleaned_item)
        
        # حفظ البيانات في الـ sheet
        current_time = datetime.now().isoformat()
        row_data = [
            current_time,
            scenario,
            json.dumps(row_indices),
            json.dumps(prediction_data),
            json.dumps(cleaned_buffer),
            current_time
        ]
        
        # بدل مسح الصفوف، اكتب فوق الصف الثاني (index 2)
        try:
            worksheet.update(f"A2:F2", [row_data])
            print(f"✅ Data updated in Google Sheets at {datetime.now().strftime('%H:%M:%S')}")
        except Exception as e:
            # لو فشل التحديث، جرب الإضافة
            worksheet.append_row(row_data)
            print(f"✅ Data appended to Google Sheets at {datetime.now().strftime('%H:%M:%S')}")
            
    except Exception as e:
        print(f"❌ Background save error: {e}")

def save_to_local_file(data_buffer, scenario, row_indices, prediction_data):
    """حفظ البيانات في ملف محلي كـ fallback"""
    try:
        # تحضير البيانات للحفظ
        cleaned_buffer = []
        for item in data_buffer[-20:]:  # آخر 20 نقطة فقط
            if isinstance(item, dict):
                cleaned_item = item.copy()
                if 'timestamp' in cleaned_item:
                    cleaned_item['timestamp'] = cleaned_item['timestamp'].isoformat()
                cleaned_buffer.append(cleaned_item)
        
        state_data = {
            'current_scenario': scenario,
            'row_indices': row_indices,
            'prediction_data': prediction_data,
            'data_buffer': cleaned_buffer,
            'last_update': datetime.now().isoformat()
        }
        
        with open(SHARED_STATE_FILE, 'w') as f:
            json.dump(state_data, f, indent=2)
        
        print(f"✅ Data saved to local file: {SHARED_STATE_FILE}")
        return True
    except Exception as e:
        print(f"❌ Error saving to local file: {e}")
        return False

def load_from_local_file():
    """قراءة البيانات من الملف المحلي"""
    try:
        if not os.path.exists(SHARED_STATE_FILE):
            print(f"ℹ️ Local file {SHARED_STATE_FILE} does not exist")
            return None
        
        with open(SHARED_STATE_FILE, 'r') as f:
            state_data = json.load(f)
        
        # تحويل timestamps من strings إلى datetime objects
        if 'data_buffer' in state_data and state_data['data_buffer']:
            for item in state_data['data_buffer']:
                if 'timestamp' in item and isinstance(item['timestamp'], str):
                    try:
                        item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                    except Exception as e:
                        print(f"⚠️ Error parsing timestamp: {e}")
                        item['timestamp'] = datetime.now()
        
        print(f"✅ Loaded {len(state_data['data_buffer'])} data points from local file")
        return state_data
        
    except Exception as e:
        print(f"❌ Error loading from local file: {e}")
        return None

def save_shared_state(data_buffer, scenario, row_indices, prediction_data):
    """حفظ الحالة المشتركة مع fallback إلى ملف محلي"""
    global _last_save_time, _pending_data
    
    current_time = time.time()
    
    # حفظ البيانات مؤقتاً
    _pending_data = {
        'data_buffer': data_buffer,
        'scenario': scenario,
        'row_indices': row_indices,
        'prediction_data': prediction_data
    }
    
    # جرب الحفظ في Google Sheets أولاً
    google_sheets_available = get_secrets() is not None
    
    if google_sheets_available and current_time - _last_save_time >= 15:
        _last_save_time = current_time
        
        # اعمل الحفظ في background thread عشان ميبطئش الواجهة
        threading.Thread(
            target=save_in_background,
            args=(data_buffer, scenario, row_indices, prediction_data),
            daemon=True
        ).start()
        print("🔄 Saving to Google Sheets in background...")
    
    # احفظ في الملف المحلي كـ fallback (دائماً)
    save_to_local_file(data_buffer, scenario, row_indices, prediction_data)
    
    return True

def load_shared_state():
    """قراءة الحالة المشتركة من Google Sheets مع fallback للملف المحلي"""
    # أولاً جرب Google Sheets
    try:
        worksheet = get_worksheet()
        if worksheet:
            records = worksheet.get_all_records()
            if records:
                # اخذ آخر سجل
                latest_record = records[-1]
                
                # تأكد من وجود البيانات المطلوبة
                required_fields = ['scenario', 'row_indices', 'prediction_data', 'data_buffer', 'last_update']
                all_fields_present = all(field in latest_record and latest_record[field] for field in required_fields)
                
                if all_fields_present:
                    # تحويل البيانات من strings إلى objects
                    state = {
                        'current_scenario': latest_record['scenario'],
                        'row_indices': json.loads(latest_record['row_indices']),
                        'prediction_data': json.loads(latest_record['prediction_data']),
                        'data_buffer': json.loads(latest_record['data_buffer']),
                        'last_update': latest_record['last_update']
                    }
                    
                    # تحويل timestamps من strings إلى datetime objects
                    if 'data_buffer' in state and state['data_buffer']:
                        for item in state['data_buffer']:
                            if 'timestamp' in item and isinstance(item['timestamp'], str):
                                try:
                                    item['timestamp'] = datetime.fromisoformat(item['timestamp'])
                                except Exception as e:
                                    print(f"⚠️ Error parsing timestamp: {e}")
                                    item['timestamp'] = datetime.now()
                    
                    print(f"✅ Loaded {len(state['data_buffer'])} data points from Google Sheets")
                    return state
                else:
                    print("⚠️ Missing or empty fields in Google Sheets")
            else:
                print("⚠️ No records found in Google Sheets")
        else:
            print("⚠️ No worksheet available in Google Sheets")
    except Exception as e:
        print(f"❌ Error loading from Google Sheets: {e}")
    
    # إذا فشل Google Sheets، جرب الملف المحلي
    print("🔄 Falling back to local file...")
    return load_from_local_file()

def is_state_fresh(max_age_seconds=30):
    """فحص إن الحالة المشتركة حديثة"""
    global _pending_data
    
    # أولاً جرب تحميل البيانات من مصادر مختلفة
    try:
        state = load_shared_state()
        if state and 'last_update' in state:
            last_update = datetime.fromisoformat(state['last_update'])
            age = (datetime.now() - last_update).total_seconds()
            is_fresh = age <= max_age_seconds
            
            if is_fresh:
                print(f"✅ Fresh data loaded (age: {age:.1f}s)")
                return True, state
            else:
                print(f"⚠️ Stale data loaded (age: {age:.1f}s)")
        else:
            print("⚠️ No valid data from external sources")
    except Exception as e:
        print(f"❌ Error loading data: {e}")
    
    # لو البيانات مش fresh، جرب الـ pending data
    if _pending_data:
        print("🔄 Using pending data as fallback")
        return True, {
            'current_scenario': _pending_data['scenario'],
            'row_indices': _pending_data['row_indices'],
            'prediction_data': _pending_data['prediction_data'],
            'data_buffer': _pending_data['data_buffer'][-20:],
            'last_update': datetime.now().isoformat()
        }
    
    print("❌ No fresh data available")
    return False, None

def clear_pending_data():
    """مسح البيانات المؤقتة"""
    global _pending_data
    _pending_data = None
    print("✅ Pending data cleared")

def get_pending_data_status():
    """فحص حالة البيانات المؤقتة"""
    global _pending_data
    if _pending_data:
        return f"Pending data exists for scenario: {_pending_data['scenario']}"
    else:
        return "No pending data"

def test_shared_state():
    """اختبار الـ shared state مع Google Sheets"""
    print("🧪 Testing optimized Google Sheets shared state...")
    
    # بيانات تجريبية
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
    
    # اختبار الحفظ
    success = save_shared_state(test_data, "normal", {"normal": 0}, test_prediction)
    if success:
        print("✅ Save test passed")
        
        # انتظر شوية عشان الـ background thread يخلص
        time.sleep(2)
        
        # اختبار التحميل
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