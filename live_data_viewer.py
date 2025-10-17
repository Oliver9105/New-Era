#!/usr/bin/env python3
"""
Live Data Viewer - Shows real betting site data in real-time
Focuses on actual server data, not predictions
"""

import requests
import time
import json
from datetime import datetime
import threading

def print_header():
    """Print live data viewer header"""
    print("=" * 80)
    print("🎮 AVIATOR PREDICTOR - LIVE SERVER DATA VIEWER")
    print("=" * 80)
    print("📡 Showing REAL data fetched from betting sites")
    print("🎯 Not predictions - actual live game results!")
    print("=" * 80)
    print()

def get_live_rounds():
    """Get live round data from server"""
    try:
        response = requests.get("http://localhost:5000/api/live-rounds", timeout=5)
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def force_data_collection(site_url):
    """Force start data collection for a specific site"""
    try:
        response = requests.post("http://localhost:5000/api/force-data-collection", 
                               json={"site_url": site_url}, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return {"success": False, "error": f"HTTP {response.status_code}"}
    except Exception as e:
        return {"success": False, "error": str(e)}

def display_live_rounds(rounds_data):
    """Display live rounds data"""
    if not rounds_data.get('success'):
        print(f"❌ Error getting live data: {rounds_data.get('error', 'Unknown')}")
        return
    
    live_rounds = rounds_data.get('live_rounds', [])
    
    print("📊 LIVE GAME ROUNDS (Real Server Data):")
    print("-" * 80)
    
    if not live_rounds:
        print("⏳ No live data available yet...")
        print("💡 Data collection may still be connecting to betting sites")
        print("⚠️  Timeout errors are normal - betting sites have protection")
        return
    
    # Show current site info
    current_site = rounds_data.get('current_site', 'None')
    active_connections = rounds_data.get('active_connections', 0)
    
    print(f"🎯 Current Site: {current_site}")
    print(f"🔗 Active Connections: {active_connections}")
    print(f"📈 Total Rounds Collected: {rounds_data.get('total_rounds', 0)}")
    print()
    
    # Display recent rounds
    print("🎮 RECENT LIVE ROUNDS:")
    print(f"{'Time':<20} {'Multiplier':<12} {'Round ID':<15} {'Source':<15} {'Site'}")
    print("-" * 80)
    
    for i, round_data in enumerate(live_rounds[:15]):  # Show last 15 rounds
        timestamp = round_data.get('timestamp', '')[:19]  # Remove milliseconds
        multiplier = f"{round_data.get('multiplier', 'N/A')}x"
        round_id = str(round_data.get('round_id', 'unknown'))[:14]
        source = round_data.get('source', 'unknown')[:14]
        site = round_data.get('site', '')
        
        # Extract domain from site URL
        if site and '://' in site:
            site = site.split('://')[1].split('/')[0][:20]
        
        print(f"{timestamp:<20} {multiplier:<12} {round_id:<15} {source:<15} {site}")
    
    print()

def display_real_time_status():
    """Display real-time connection status"""
    try:
        response = requests.get("http://localhost:5000/api/connection-status", timeout=5)
        if response.status_code == 200:
            data = response.json()
            
            if data.get('success'):
                collector_status = data.get('data_collector', {})
                system_health = data.get('system_health', {})
                
                print("📡 REAL-TIME CONNECTION STATUS:")
                print(f"   🔗 Active Connections: {collector_status.get('active_connections', 0)}")
                print(f"   🎯 Current Site: {collector_status.get('current_site', 'None')}")
                print(f"   📊 Real Data Available: {'✅' if system_health.get('real_data_available') else '❌'}")
                print(f"   🔄 Data Collection Active: {'✅' if system_health.get('active_data_collection') else '❌'}")
                print()
                
                # Show recent database entries
                recent_results = data.get('database', {}).get('recent_results', [])
                if recent_results:
                    print("💾 RECENT DATABASE ENTRIES (Server Data):")
                    for result in recent_results[:5]:
                        timestamp = result.get('timestamp', '')[:19]
                        multiplier = f"{result.get('multiplier', 'N/A')}x"
                        platform = result.get('platform', 'unknown')
                        print(f"   {timestamp} | {multiplier} | {platform}")
                    print()
                
    except Exception as e:
        print(f"⚠️  Could not get connection status: {e}")

def interactive_mode():
    """Interactive mode for testing different sites"""
    print("🔧 INTERACTIVE MODE")
    print("=" * 50)
    
    while True:
        print("\nOptions:")
        print("1. Force data collection from betika.com")
        print("2. Force data collection from custom site")
        print("3. Show current live data")
        print("4. Show connection status")
        print("5. Return to auto-refresh mode")
        print("6. Exit")
        
        try:
            choice = input("\nEnter choice (1-6): ").strip()
            
            if choice == '1':
                print("\n🔄 Starting data collection from betika.com...")
                result = force_data_collection("https://betika.com")
                if result.get('success'):
                    print(f"✅ Collection started: {result.get('message')}")
                    print(f"🔗 Connections: {result.get('connections_started', 0)}")
                else:
                    print(f"❌ Failed: {result.get('error')}")
                    
            elif choice == '2':
                site_url = input("Enter betting site URL: ").strip()
                if site_url:
                    if not site_url.startswith(('http://', 'https://')):
                        site_url = 'https://' + site_url
                    print(f"\n🔄 Starting data collection from {site_url}...")
                    result = force_data_collection(site_url)
                    if result.get('success'):
                        print(f"✅ Collection started: {result.get('message')}")
                    else:
                        print(f"❌ Failed: {result.get('error')}")
                        
            elif choice == '3':
                print("\n📊 Current Live Data:")
                rounds_data = get_live_rounds()
                display_live_rounds(rounds_data)
                
            elif choice == '4':
                print("\n📡 Connection Status:")
                display_real_time_status()
                
            elif choice == '5':
                break
                
            elif choice == '6':
                print("👋 Goodbye!")
                return
                
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            return
        except Exception as e:
            print(f"❌ Error: {e}")

def auto_refresh_mode():
    """Auto-refreshing display of live data"""
    print("🔄 AUTO-REFRESH MODE (Updates every 10 seconds)")
    print("Press Ctrl+C to enter interactive mode")
    print()
    
    try:
        while True:
            # Clear screen (works in most terminals)
            print("\033[H\033[J", end="")
            
            print_header()
            
            # Show current time
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(f"🕐 Last Updated: {current_time}")
            print()
            
            # Show connection status
            display_real_time_status()
            
            # Show live rounds
            rounds_data = get_live_rounds()
            display_live_rounds(rounds_data)
            
            print("=" * 80)
            print("💡 TIP: If no live data is showing, try:")
            print("   • Wait 2-3 minutes for connections to establish")
            print("   • Use interactive mode (Ctrl+C) to force collection")
            print("   • Check different betting sites")
            print("=" * 80)
            
            # Wait before next update
            time.sleep(10)
            
    except KeyboardInterrupt:
        print("\n\n🔧 Entering interactive mode...")
        interactive_mode()

def main():
    """Main function"""
    print_header()
    
    print("🔍 Checking app connection...")
    
    # Test connection to app
    try:
        response = requests.get("http://localhost:5000/api/status", timeout=5)
        if response.status_code == 200:
            print("✅ Connected to Aviator Predictor app")
        else:
            print(f"❌ App returned status {response.status_code}")
            print("💡 Make sure the app is running: python app.py")
            return
    except Exception as e:
        print(f"❌ Cannot connect to app: {e}")
        print("💡 Make sure the app is running on localhost:5000")
        return
    
    print()
    print("📡 This tool shows REAL data from betting sites")
    print("🎯 You'll see actual multipliers from live games")
    print("⏱️  Timeout errors in app logs are normal")
    print()
    
    # Start auto-refresh mode
    auto_refresh_mode()

if __name__ == "__main__":
    main()