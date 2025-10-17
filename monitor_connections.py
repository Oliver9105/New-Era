#!/usr/bin/env python3
"""
Real-time connection monitor for Aviator Predictor
Shows live status of data collection and connections
"""

import requests
import time
import json
from datetime import datetime

def print_banner():
    """Print monitoring banner"""
    print("=" * 70)
    print("🔍 AVIATOR PREDICTOR - REAL-TIME CONNECTION MONITOR")
    print("=" * 70)
    print()

def get_connection_status():
    """Get current connection status from the app"""
    try:
        response = requests.get("http://localhost:5000/api/realtime-data", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def get_latest_prediction():
    """Get latest prediction"""
    try:
        response = requests.post("http://localhost:5000/api/predict", 
                               json={"model": "ensemble"}, timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"error": str(e)}

def format_status_indicator(is_good):
    """Format status indicator"""
    return "🟢" if is_good else "🔴"

def monitor_connections():
    """Main monitoring loop"""
    print_banner()
    
    print("⏱️  Starting real-time monitoring...")
    print("   Press Ctrl+C to stop")
    print()
    
    try:
        while True:
            # Clear previous output (works in most terminals)
            print("\033[H\033[J", end="")
            print_banner()
            
            # Get current timestamp
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"📅 Last Updated: {current_time}")
            print()
            
            # Get connection status
            status = get_connection_status()
            
            if status.get("success"):
                # Connection info
                active_connections = status.get("active_connections", 0)
                current_site = status.get("current_site", "None")
                
                print("📊 CONNECTION STATUS:")
                print(f"   {format_status_indicator(active_connections > 0)} Active Connections: {active_connections}")
                print(f"   🎯 Current Site: {current_site or 'None'}")
                print()
                
                # Real-time data info
                realtime_data = status.get("realtime_data")
                if realtime_data:
                    print("📡 REAL-TIME DATA:")
                    print(f"   ✅ Data Source: {realtime_data.get('source', 'unknown')}")
                    print(f"   🔗 URL: {realtime_data.get('url', 'unknown')}")
                    
                    if realtime_data.get('game_data'):
                        game_data = realtime_data['game_data']
                        print(f"   🎮 Game Data: {json.dumps(game_data, indent=6)}")
                    print()
                else:
                    print("📡 REAL-TIME DATA:")
                    print(f"   ⏳ No real-time data yet (connections establishing...)")
                    print()
                
                # Historical data info
                historical_data = status.get("historical_data", [])
                print("📈 HISTORICAL DATA:")
                print(f"   📊 Data Points Collected: {len(historical_data)}")
                
                if historical_data:
                    print("   🕒 Recent Data Points:")
                    for i, data_point in enumerate(historical_data[-3:]):  # Show last 3
                        timestamp = data_point.get('timestamp', 'unknown')
                        source = data_point.get('source', 'unknown')
                        game_data = data_point.get('game_data', {})
                        multiplier = game_data.get('multiplier', 'N/A')
                        print(f"      {i+1}. {timestamp[:19]} | Source: {source} | Multiplier: {multiplier}")
                print()
                
                # Prediction info
                prediction = get_latest_prediction()
                print("🔮 CURRENT PREDICTION:")
                if "error" not in prediction:
                    multiplier = prediction.get('multiplier', 'N/A')
                    confidence = prediction.get('confidence', 'N/A')
                    method = prediction.get('method', 'unknown')
                    data_points = prediction.get('data_points_used', 'N/A')
                    
                    # Determine if using real data
                    using_real_data = method in ['real_data_heuristic', 'ensemble'] or data_points not in ['N/A', None, 0]
                    
                    print(f"   {format_status_indicator(using_real_data)} Predicted Multiplier: {multiplier}x")
                    print(f"   📊 Confidence: {confidence}")
                    print(f"   🔧 Method: {method}")
                    
                    if data_points not in ['N/A', None]:
                        print(f"   📈 Data Points Used: {data_points}")
                        if int(data_points or 0) > 0:
                            print(f"   ✅ Using REAL collected data! 🎉")
                        else:
                            print(f"   ⏳ No real data yet, using intelligent patterns")
                    
                    # Show trend if available
                    if 'recent_trend' in prediction:
                        print(f"   📈 Recent Trend: {prediction['recent_trend']}x")
                    if 'historical_avg' in prediction:
                        print(f"   📊 Historical Average: {prediction['historical_avg']}x")
                else:
                    print(f"   ❌ Prediction Error: {prediction.get('error', 'Unknown')}")
                print()
                
            else:
                print("❌ CONNECTION STATUS:")
                print(f"   🔴 Unable to connect to app: {status.get('error', 'Unknown')}")
                print("   📝 Make sure the app is running on localhost:5000")
                print()
            
            # Status summary
            print("=" * 70)
            if status.get("success") and status.get("active_connections", 0) > 0:
                print("✅ SYSTEM STATUS: ACTIVELY COLLECTING REAL DATA")
            elif status.get("success"):
                print("⏳ SYSTEM STATUS: READY (waiting for data collection to start)")
            else:
                print("❌ SYSTEM STATUS: APP NOT CONNECTED")
            print("=" * 70)
            
            # Wait before next update
            time.sleep(5)  # Update every 5 seconds
            
    except KeyboardInterrupt:
        print("\n\n👋 Monitoring stopped by user")
    except Exception as e:
        print(f"\n\n❌ Monitoring error: {e}")

if __name__ == "__main__":
    monitor_connections()