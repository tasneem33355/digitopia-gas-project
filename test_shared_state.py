#!/usr/bin/env python3
"""
Test script for shared_state functionality
Run this to test if shared_state is working properly
"""

import sys
import os
from datetime import datetime
import time

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    import shared_state
    print("✅ shared_state module imported successfully")
except ImportError as e:
    print(f"❌ Failed to import shared_state: {e}")
    sys.exit(1)

def test_local_file_functionality():
    """Test local file functionality specifically"""
    print("\n💾 Testing local file functionality...")
    
    # Test data
    test_data = [{
        'timestamp': datetime.now(),
        'pressure': 35.0,
        'temperature': 5.0,
        'flow_rate': 70.0,
        'valve_status': 1,
        'pump_state': 1,
        'pump_speed': 1000.0,
        'compressor_state': 0.5,
        'energy_consumption': 25.0,
        'alarm_triggered': 0
    }]
    
    test_prediction = {
        'prediction': 0,
        'probabilities': [0.8, 0.1, 0.1],
        'confidence': 0.8
    }
    
    test_row_indices = {
        'normal': 5,
        'warning': 3,
        'failure': 1
    }
    
    # Test local file save
    print("📝 Testing local file save...")
    try:
        success = shared_state.save_to_local_file(
            test_data, 
            "normal", 
            test_row_indices, 
            test_prediction
        )
        if success:
            print("✅ Local file save test passed")
        else:
            print("❌ Local file save test failed")
            return False
    except Exception as e:
        print(f"❌ Local file save test failed with error: {e}")
        return False
    
    # Test local file load
    print("📖 Testing local file load...")
    try:
        loaded_state = shared_state.load_from_local_file()
        if loaded_state:
            print("✅ Local file load test passed")
            print(f"   - Loaded {len(loaded_state.get('data_buffer', []))} data points")
            print(f"   - Scenario: {loaded_state.get('current_scenario', 'Unknown')}")
            print(f"   - Last update: {loaded_state.get('last_update', 'Unknown')}")
        else:
            print("❌ Local file load test failed - no data returned")
            return False
    except Exception as e:
        print(f"❌ Local file load test failed with error: {e}")
        return False
    
    return True

def test_basic_functionality():
    """Test basic shared_state functionality"""
    print("\n🧪 Testing integrated functionality...")
    
    # Test data
    test_data = [{
        'timestamp': datetime.now(),
        'pressure': 35.0,
        'temperature': 5.0,
        'flow_rate': 70.0,
        'valve_status': 1,
        'pump_state': 1,
        'pump_speed': 1000.0,
        'compressor_state': 0.5,
        'energy_consumption': 25.0,
        'alarm_triggered': 0
    }]
    
    test_prediction = {
        'prediction': 0,
        'probabilities': [0.8, 0.1, 0.1],
        'confidence': 0.8
    }
    
    test_row_indices = {
        'normal': 5,
        'warning': 3,
        'failure': 1
    }
    
    # Test save (integrated function)
    print("📝 Testing integrated save...")
    try:
        success = shared_state.save_shared_state(
            test_data, 
            "normal", 
            test_row_indices, 
            test_prediction
        )
        if success:
            print("✅ Integrated save test passed")
        else:
            print("❌ Integrated save test failed")
            return False
    except Exception as e:
        print(f"❌ Integrated save test failed with error: {e}")
        return False
    
    # Wait a bit for background operations
    print("⏳ Waiting for background operations...")
    time.sleep(3)
    
    # Test load (integrated function)
    print("📖 Testing integrated load...")
    try:
        loaded_state = shared_state.load_shared_state()
        if loaded_state:
            print("✅ Integrated load test passed")
            print(f"   - Loaded {len(loaded_state.get('data_buffer', []))} data points")
            print(f"   - Scenario: {loaded_state.get('current_scenario', 'Unknown')}")
            print(f"   - Last update: {loaded_state.get('last_update', 'Unknown')}")
        else:
            print("❌ Integrated load test failed - no data returned")
            return False
    except Exception as e:
        print(f"❌ Integrated load test failed with error: {e}")
        return False
    
    # Test freshness check
    print("⏰ Testing freshness check...")
    try:
        is_fresh, state_data = shared_state.is_state_fresh(max_age_seconds=60)
        if is_fresh and state_data:
            print("✅ Freshness test passed")
            print(f"   - Data is fresh")
            print(f"   - Contains {len(state_data.get('data_buffer', []))} points")
        else:
            print("⚠️ Data is not fresh or missing")
            return False
    except Exception as e:
        print(f"❌ Freshness test failed with error: {e}")
        return False
    
    return True

def test_google_sheets_connection():
    """Test Google Sheets connection specifically"""
    print("\n🔗 Testing Google Sheets connection...")
    
    try:
        client = shared_state.get_sheets_client()
        if client:
            print("✅ Google Sheets client initialized")
        else:
            print("❌ Failed to initialize Google Sheets client")
            return False
    except Exception as e:
        print(f"❌ Google Sheets client error: {e}")
        return False
    
    try:
        worksheet = shared_state.get_worksheet()
        if worksheet:
            print("✅ Worksheet accessed successfully")
            print(f"   - Worksheet title: {worksheet.title}")
        else:
            print("❌ Failed to access worksheet")
            return False
    except Exception as e:
        print(f"❌ Worksheet access error: {e}")
        return False
    
    return True

def main():
    """Main test function"""
    print("🚀 Starting shared_state tests...")
    print("=" * 50)
    
    # Test Google Sheets connection first
    sheets_ok = test_google_sheets_connection()
    
    # Test local file functionality
    local_ok = test_local_file_functionality()
    
    # Test integrated functionality
    basic_ok = test_basic_functionality()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Google Sheets: {'✅ PASS' if sheets_ok else '❌ FAIL'}")
    print(f"   Local File: {'✅ PASS' if local_ok else '❌ FAIL'}")
    print(f"   Integrated functionality: {'✅ PASS' if basic_ok else '❌ FAIL'}")
    
    if local_ok and basic_ok:
        print("\n🎉 Core functionality working! shared_state can work with local files.")
        if sheets_ok:
            print("🌟 Google Sheets integration is also working!")
        else:
            print("⚠️ Google Sheets not configured, but local fallback is working.")
        return 0
    else:
        print("\n⚠️ Some core tests failed. Check the errors above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)