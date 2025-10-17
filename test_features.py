#!/usr/bin/env python3
"""
Test script to demonstrate all enhanced features of the Aviator Predictor
"""

import requests
import json
import time

def test_aviator_features():
    base_url = "http://localhost:5000"
    
    print("🚀 Testing Enhanced Aviator Predictor Features")
    print("=" * 50)
    
    # Test 1: Get Prediction
    print("\n1. 🎯 Testing Prediction API...")
    try:
        response = requests.get(f"{base_url}/api/prediction")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Prediction: {data['prediction']['multiplier']}x")
            print(f"   ✅ Confidence: {data['prediction']['confidence']*100:.1f}%")
            print(f"   ✅ Method: {data['prediction']['method']}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Network Analysis
    print("\n2. 🔧 Testing Network Analysis...")
    try:
        payload = {"site_url": "https://example.com"}
        response = requests.post(f"{base_url}/api/network-analysis", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Found {len(data['websocket_urls'])} WebSocket URLs")
            print(f"   ✅ Found {len(data['api_endpoints'])} API endpoints")
            print(f"   ✅ Analysis completed successfully")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: Start Capture
    print("\n3. 📡 Testing Data Capture...")
    try:
        payload = {"site_url": "https://test-aviator.com"}
        response = requests.post(f"{base_url}/api/start-capture", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Capture started: {data['message']}")
            print(f"   ✅ Capture ID: {data.get('capture_id', 'N/A')}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 4: Manual Endpoint
    print("\n4. 🔗 Testing Manual Endpoint Connection...")
    try:
        payload = {
            "endpoint_url": "wss://example.com/aviator-ws",
            "type": "websocket"
        }
        response = requests.post(f"{base_url}/api/manual-endpoint", json=payload)
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Connection: {data['message']}")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 5: Recent Data
    print("\n5. 📈 Testing Recent Data...")
    try:
        response = requests.get(f"{base_url}/api/recent-data?limit=5")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Retrieved {len(data['data'])} recent rounds")
            if data['data']:
                latest = data['data'][0]
                print(f"   ✅ Latest: Round {latest['round_id']} = {latest['crash_multiplier']}x")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 6: Statistics
    print("\n6. 📊 Testing Statistics...")
    try:
        response = requests.get(f"{base_url}/api/statistics")
        if response.status_code == 200:
            data = response.json()
            stats = data['statistics']
            print(f"   ✅ Total Rounds: {stats['total_rounds']}")
            print(f"   ✅ Average Multiplier: {stats['avg_multiplier']}x")
            print(f"   ✅ Max Multiplier: {stats['max_multiplier']}x")
            print(f"   ✅ Accuracy: {stats['avg_accuracy']}%")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 7: Web Interface
    print("\n7. 🌐 Testing Web Interface...")
    try:
        response = requests.get(f"{base_url}/")
        if response.status_code == 200 and "Aviator Prediction System" in response.text:
            print("   ✅ Web interface accessible")
            print("   ✅ Enhanced dashboard loaded")
        else:
            print(f"   ❌ Failed: {response.status_code}")
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Feature testing completed!")
    print("\n📋 Summary of Enhanced Features:")
    print("   • Network Analysis - Discover WebSocket & API endpoints")
    print("   • Data Capture - Start/stop real-time data collection")
    print("   • Manual Endpoints - Connect to custom URLs")
    print("   • Live Predictions - AI-powered multiplier prediction")
    print("   • Recent Data Table - View latest game rounds")
    print("   • Statistics Dashboard - 24h analytics")
    print("   • Modern UI - Enhanced dashboard with alerts")
    print("   • Real-time Updates - Auto-refresh functionality")

if __name__ == "__main__":
    test_aviator_features()
