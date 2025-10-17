#!/usr/bin/env python3
"""
Quick Real Data Test - Check if app is getting actual live data
"""

import requests
import json
import time

def test_live_data():
    """Quick test to see if real data is flowing"""
    
    print("🧪 QUICK REAL DATA TEST")
    print("=" * 40)
    
    try:
        # Test 1: Check live rounds
        print("1️⃣ Checking for live rounds...")
        response = requests.get("http://localhost:5000/api/live-rounds", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            live_rounds = data.get('live_rounds', [])
            
            if live_rounds:
                print(f"✅ REAL DATA FOUND: {len(live_rounds)} live rounds!")
                print("\n📊 Recent Live Multipliers:")
                for i, round_data in enumerate(live_rounds[:5]):
                    multiplier = round_data.get('multiplier', 'N/A')
                    timestamp = round_data.get('timestamp', '')[:19]
                    source = round_data.get('source', 'unknown')
                    print(f"   {i+1}. {multiplier}x at {timestamp} from {source}")
                
                print("\n🎉 SUCCESS: Your app is collecting REAL live data!")
                return True
            else:
                print("❌ No live rounds found yet")
        else:
            print(f"❌ API error: {response.status_code}")
    
    except Exception as e:
        print(f"❌ Connection error: {e}")
    
    # Test 2: Check connection status
    print("\n2️⃣ Checking connection status...")
    try:
        response = requests.get("http://localhost:5000/api/connection-status", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            if data.get('success'):
                collector = data.get('data_collector', {})
                health = data.get('system_health', {})
                
                print(f"   🔗 Active Connections: {collector.get('active_connections', 0)}")
                print(f"   🎯 Current Site: {collector.get('current_site', 'None')}")
                print(f"   📊 Real Data Available: {'✅' if health.get('real_data_available') else '❌'}")
                print(f"   🔄 Data Collection Active: {'✅' if health.get('active_data_collection') else '❌'}")
                
                if health.get('real_data_available'):
                    print("\n✅ SUCCESS: Real data is available!")
                    return True
                elif collector.get('active_connections', 0) > 0:
                    print("\n⏳ Connections active but no data yet - keep waiting!")
                    return False
                else:
                    print("\n❌ No active connections")
            
    except Exception as e:
        print(f"❌ Status check error: {e}")
    
    # Test 3: Check recent predictions
    print("\n3️⃣ Checking prediction method...")
    try:
        response = requests.post("http://localhost:5000/api/predict", 
                               json={"model": "ensemble"}, timeout=5)
        
        if response.status_code == 200:
            prediction = response.json()
            method = prediction.get('method', 'unknown')
            data_points = prediction.get('data_points_used', 0)
            
            print(f"   🔧 Prediction Method: {method}")
            
            if method in ['real_data_heuristic', 'ensemble']:
                print("   ✅ Using real data for predictions!")
                return True
            elif data_points and int(data_points) > 0:
                print(f"   ✅ Using {data_points} real data points!")
                return True
            else:
                print("   ⏳ Still using fallback patterns (no real data yet)")
                
    except Exception as e:
        print(f"❌ Prediction test error: {e}")
    
    return False

def quick_start_collection():
    """Quick start data collection"""
    print("\n🔄 STARTING DATA COLLECTION")
    print("-" * 40)
    
    # Try betika
    print("Starting collection from betika.com...")
    try:
        response = requests.post("http://localhost:5000/api/force-data-collection",
                               json={"site_url": "https://betika.com"}, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            if result.get('success'):
                print(f"✅ Collection started: {result.get('connections_started', 0)} connections")
            else:
                print(f"❌ Failed to start collection")
        
    except Exception as e:
        print(f"❌ Error starting collection: {e}")

def main():
    """Main test function"""
    print("🎯 AVIATOR PREDICTOR - QUICK REAL DATA TEST")
    print("=" * 50)
    
    # Check if app is running
    print("🔍 Checking if app is running...")
    try:
        response = requests.get("http://localhost:5000/api/status", timeout=5)
        if response.status_code == 200:
            print("✅ App is running!")
        else:
            print(f"❌ App returned status {response.status_code}")
            print("💡 Make sure to run: python app.py")
            return
    except Exception as e:
        print(f"❌ Cannot connect to app: {e}")
        print("💡 Make sure to run: python app.py")
        return
    
    print()
    
    # Test for real data
    has_real_data = test_live_data()
    
    if not has_real_data:
        print("\n" + "=" * 50)
        print("💡 NO REAL DATA YET - Let's start collection!")
        quick_start_collection()
        
        print("\n⏱️ Waiting 30 seconds for data collection...")
        print("(Timeout errors in logs are normal)")
        time.sleep(30)
        
        print("\n🔍 Testing again after collection...")
        has_real_data = test_live_data()
    
    print("\n" + "=" * 50)
    if has_real_data:
        print("🎉 SUCCESS: Your app is collecting REAL live data!")
        print("📊 Use this to see live data: python live_data_viewer.py")
    else:
        print("⏳ No real data yet - this is normal!")
        print("💡 Recommendations:")
        print("   • Wait 2-3 more minutes for connections")
        print("   • Try different sites with the live viewer")
        print("   • Check logs for successful connections")
        print("   • Timeout errors are normal for betting sites")
    
    print("\n📞 Next steps:")
    print("   • python live_data_viewer.py (main tool)")
    print("   • python monitor_connections.py (status)")
    print("   • python troubleshoot_timeouts.py (help)")

if __name__ == "__main__":
    main()