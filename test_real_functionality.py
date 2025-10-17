#!/usr/bin/env python3
"""
Test script to verify real data functionality
"""

import requests
import json
import time

def test_api_endpoints():
    """Test the main API endpoints"""
    base_url = "http://localhost:5000"
    
    print("🔍 Testing API endpoints...")
    
    # Test 1: Status endpoint
    print("\n1. Testing status endpoint...")
    try:
        response = requests.get(f"{base_url}/api/status")
        print(f"Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Features: {data.get('features', {})}")
    except Exception as e:
        print(f"❌ Status test failed: {e}")
    
    # Test 2: Network analysis endpoint
    print("\n2. Testing network analysis endpoint...")
    try:
        test_payload = {"site_url": "https://chachisha.game"}
        response = requests.post(f"{base_url}/api/network-analysis", json=test_payload)
        print(f"Network Analysis: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data.get('success')}")
            if data.get('success'):
                print(f"Site URL: {data.get('site_url')}")
                print(f"Data Collection: {data.get('data_collection', {}).get('success')}")
        else:
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Network analysis test failed: {e}")
    
    # Test 3: Start capture endpoint
    print("\n3. Testing start capture endpoint...")
    try:
        test_payload = {"site_url": "https://chachisha.game"}
        response = requests.post(f"{base_url}/api/start-capture", json=test_payload)
        print(f"Start Capture: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data.get('success')}")
            if data.get('success'):
                print(f"Capture Result: {data.get('capture_result', {})}")
        else:
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Start capture test failed: {e}")
    
    # Test 4: Prediction endpoint
    print("\n4. Testing prediction endpoint...")
    try:
        test_payload = {"model": "ensemble"}
        response = requests.post(f"{base_url}/api/predict", json=test_payload)
        print(f"Prediction: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Predicted Multiplier: {data.get('multiplier')}")
            print(f"Confidence: {data.get('confidence')}")
            print(f"Method: {data.get('method')}")
            if 'data_points_used' in data:
                print(f"Data Points Used: {data.get('data_points_used')}")
        else:
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
    
    # Test 5: Real-time data endpoint
    print("\n5. Testing real-time data endpoint...")
    try:
        response = requests.get(f"{base_url}/api/realtime-data")
        print(f"Real-time Data: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"Success: {data.get('success')}")
            print(f"Active Connections: {data.get('active_connections')}")
            print(f"Current Site: {data.get('current_site')}")
            if data.get('realtime_data'):
                print(f"Latest Data: {data.get('realtime_data')}")
        else:
            print(f"Response: {response.text}")
    except Exception as e:
        print(f"❌ Real-time data test failed: {e}")

def test_full_workflow():
    """Test the complete workflow"""
    print("\n🚀 Testing complete workflow...")
    
    base_url = "http://localhost:5000"
    test_site = "https://chachisha.game"
    
    # Step 1: Start network analysis
    print(f"\n📡 Step 1: Analyzing {test_site}")
    try:
        response = requests.post(f"{base_url}/api/network-analysis", 
                               json={"site_url": test_site})
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Network analysis started: {data.get('success')}")
        else:
            print(f"❌ Network analysis failed: {response.status_code}")
            return
    except Exception as e:
        print(f"❌ Network analysis error: {e}")
        return
    
    # Step 2: Start data capture
    print(f"\n🎯 Step 2: Starting data capture")
    try:
        response = requests.post(f"{base_url}/api/start-capture", 
                               json={"site_url": test_site})
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Data capture started: {data.get('success')}")
            if data.get('success'):
                capture_result = data.get('capture_result', {})
                print(f"Connections started: {capture_result.get('connections_started', 0)}")
        else:
            print(f"❌ Data capture failed: {response.status_code}")
    except Exception as e:
        print(f"❌ Data capture error: {e}")
    
    # Step 3: Wait a bit and get predictions
    print(f"\n⏳ Step 3: Waiting for data collection...")
    time.sleep(3)
    
    print(f"\n🔮 Step 4: Getting predictions")
    for i in range(3):
        try:
            response = requests.post(f"{base_url}/api/predict", 
                                   json={"model": "ensemble"})
            if response.status_code == 200:
                data = response.json()
                print(f"Prediction {i+1}: {data.get('multiplier')}x "
                      f"(confidence: {data.get('confidence')}, "
                      f"method: {data.get('method')})")
                if 'data_points_used' in data:
                    print(f"  Using {data.get('data_points_used')} real data points")
            time.sleep(1)
        except Exception as e:
            print(f"❌ Prediction {i+1} error: {e}")
    
    # Step 5: Check real-time data
    print(f"\n📊 Step 5: Checking real-time data")
    try:
        response = requests.get(f"{base_url}/api/realtime-data")
        if response.status_code == 200:
            data = response.json()
            print(f"Active connections: {data.get('active_connections')}")
            print(f"Current site: {data.get('current_site')}")
            
            historical = data.get('historical_data', [])
            print(f"Historical data points: {len(historical)}")
            
            if data.get('realtime_data'):
                print(f"Latest real-time data available: ✅")
            else:
                print(f"No real-time data yet: ⏳")
    except Exception as e:
        print(f"❌ Real-time data check error: {e}")

if __name__ == "__main__":
    print("🧪 Aviator Predictor - Real Functionality Test")
    print("=" * 50)
    
    # Basic API tests
    test_api_endpoints()
    
    print("\n" + "=" * 50)
    
    # Full workflow test
    test_full_workflow()
    
    print("\n" + "=" * 50)
    print("✅ Test completed!")
    print("\nIf you see real data points being used in predictions,")
    print("the system is working with real data! 🎉")