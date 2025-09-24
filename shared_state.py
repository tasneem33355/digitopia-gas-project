import json
import os
from datetime import datetime

STATE_FILE = "shared_state.json"

def save_shared_state(data_buffer, scenario, row_indices, prediction_data):
    """حفظ الحالة المشتركة في ملف"""
    try:
        # تحويل timestamps لـ strings قبل الحفظ
        cleaned_buffer = []
        for item in data_buffer[-50:]:  # آخر 50 نقطة فقط
            if isinstance(item, dict):
                cleaned_item = item.copy()
                if 'timestamp' in cleaned_item:
                    cleaned_item['timestamp'] = cleaned_item['timestamp'].isoformat()
                cleaned_buffer.append(cleaned_item)
        
        state = {
            'data_buffer': cleaned_buffer,
            'current_scenario': scenario,
            'row_indices': row_indices,
            'prediction_data': prediction_data,
            'last_update': datetime.now().isoformat()
        }
        
        # تأكد إن المجلد موجود
        os.makedirs(os.path.dirname(os.path.abspath(STATE_FILE)) if os.path.dirname(STATE_FILE) else ".", exist_ok=True)
        
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        print(f"✅ State saved successfully at {datetime.now()}")
        return True
        
    except Exception as e:
        print(f"❌ Error saving state: {e}")
        return False

def load_shared_state():
    """قراءة الحالة المشتركة من الملف"""
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                state = json.load(f)
            
            # تحويل timestamps من strings لـ datetime objects
            if 'data_buffer' in state:
                for item in state['data_buffer']:
                    if 'timestamp' in item and isinstance(item['timestamp'], str):
                        item['timestamp'] = datetime.fromisoformat(item['timestamp'])
            
            print(f"✅ State loaded successfully: {len(state.get('data_buffer', []))} points")
            return state
        except Exception as e:
            print(f"❌ Error loading state: {e}")
            return None
    else:
        print(f"⚠️ State file not found: {STATE_FILE}")
        return None

def is_state_fresh(max_age_seconds=15):
    """فحص إن الحالة المشتركة حديثة"""
    try:
        state = load_shared_state()
        if state and 'last_update' in state:
            last_update = datetime.fromisoformat(state['last_update'])
            age = (datetime.now() - last_update).total_seconds()
            is_fresh = age <= max_age_seconds
            
            if is_fresh:
                print(f"✅ State is fresh (age: {age:.1f}s)")
            else:
                print(f"⚠️ State is old (age: {age:.1f}s)")
            
            return is_fresh, state
        else:
            print("❌ No valid state found")
            return False, None
    except Exception as e:
        print(f"❌ Error checking state freshness: {e}")
        return False, None

def test_shared_state():
    """اختبار الـ shared state"""
    print("🧪 Testing shared state...")
    
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
        
        # اختبار التحميل
        is_fresh, loaded_state = is_state_fresh(max_age_seconds=30)
        if is_fresh:
            print("✅ Load test passed")
            print(f"Loaded {len(loaded_state['data_buffer'])} data points")
        else:
            print("❌ Load test failed")
    else:
        print("❌ Save test failed")

if __name__ == "__main__":
    test_shared_state()