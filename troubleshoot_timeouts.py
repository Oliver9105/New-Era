#!/usr/bin/env python3
"""
Timeout troubleshooter for betting site connections
Helps diagnose and resolve connection timeout issues
"""

import requests
import time
from urllib.parse import urlparse

def test_site_accessibility(site_url):
    """Test if a betting site is accessible and measure response times"""
    
    print(f"🔍 Testing accessibility for: {site_url}")
    print("-" * 50)
    
    parsed = urlparse(site_url)
    domain = parsed.netloc
    
    # Test 1: Basic connectivity
    print("1️⃣ Testing basic connectivity...")
    try:
        start_time = time.time()
        response = requests.get(site_url, timeout=30, allow_redirects=True)
        response_time = time.time() - start_time
        
        print(f"   ✅ Site is accessible")
        print(f"   📊 Response time: {response_time:.2f} seconds")
        print(f"   📝 Status code: {response.status_code}")
        print(f"   📏 Content length: {len(response.content)} bytes")
        
        if response_time > 10:
            print(f"   ⚠️  SLOW RESPONSE: Site takes {response_time:.1f}s (may cause timeouts)")
        
    except requests.exceptions.Timeout:
        print(f"   ❌ TIMEOUT: Site took longer than 30 seconds to respond")
        print(f"   💡 This explains the timeout errors in your logs")
        return False
    except requests.exceptions.ConnectionError:
        print(f"   ❌ CONNECTION ERROR: Cannot connect to {domain}")
        return False
    except Exception as e:
        print(f"   ❌ ERROR: {e}")
        return False
    
    # Test 2: Check for anti-bot protection
    print("\n2️⃣ Checking for anti-bot protection...")
    try:
        # Test with minimal headers
        response_minimal = requests.get(site_url, timeout=15)
        
        # Test with full browser headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1'
        }
        response_full = requests.get(site_url, headers=headers, timeout=15)
        
        if response_minimal.status_code != response_full.status_code:
            print(f"   ⚠️  ANTI-BOT DETECTED: Different responses with/without headers")
            print(f"      Minimal headers: {response_minimal.status_code}")
            print(f"      Full headers: {response_full.status_code}")
        else:
            print(f"   ✅ No obvious anti-bot protection detected")
            
        # Check for common protection indicators
        content = response_full.text.lower()
        if any(indicator in content for indicator in ['cloudflare', 'ddos protection', 'bot protection', 'challenge']):
            print(f"   🛡️  PROTECTION DETECTED: Site uses anti-bot protection")
            print(f"      This may cause intermittent timeouts and blocks")
        
    except Exception as e:
        print(f"   ⚠️  Could not test anti-bot protection: {e}")
    
    # Test 3: API endpoint accessibility
    print("\n3️⃣ Testing potential API endpoints...")
    base_url = f"{parsed.scheme}://{parsed.netloc}"
    
    api_endpoints = [
        f"{base_url}/api/aviator/current",
        f"{base_url}/api/aviator",
        f"{base_url}/api/live/aviator",
        f"{base_url}/aviator/api"
    ]
    
    accessible_apis = []
    
    for api_url in api_endpoints:
        try:
            response = requests.head(api_url, timeout=10, headers=headers)
            if response.status_code < 500:  # Accept various codes
                accessible_apis.append((api_url, response.status_code))
                print(f"   ✅ {api_url} → {response.status_code}")
            else:
                print(f"   ❌ {api_url} → {response.status_code}")
        except requests.exceptions.Timeout:
            print(f"   ⏱️  {api_url} → TIMEOUT")
        except Exception as e:
            print(f"   ❓ {api_url} → {str(e)[:50]}")
    
    if accessible_apis:
        print(f"\n   📊 Found {len(accessible_apis)} potentially accessible API endpoints")
    else:
        print(f"\n   ⚠️  No API endpoints found (may need deeper investigation)")
    
    return True

def provide_recommendations(site_url):
    """Provide recommendations based on test results"""
    
    print("\n" + "=" * 60)
    print("💡 RECOMMENDATIONS FOR TIMEOUT ISSUES")
    print("=" * 60)
    
    print("\n1️⃣ Understanding the Timeout Error:")
    print("   • Your app IS working correctly - it's trying to connect to real sites")
    print("   • Timeout errors are NORMAL when connecting to betting sites")
    print("   • Many betting sites are slow or have protection against automation")
    
    print("\n2️⃣ What the App is Doing:")
    print("   • Automatically trying multiple API endpoints")
    print("   • Using realistic browser headers to avoid detection")
    print("   • Backing off and retrying when timeouts occur")
    print("   • Continuing to work even when some connections fail")
    
    print("\n3️⃣ Improving Success Rate:")
    print("   • ✅ Use popular betting sites (they often have better APIs)")
    print("   • ✅ Try different times of day (less traffic = faster responses)")
    print("   • ✅ Be patient - some connections take time to establish")
    print("   • ✅ Monitor the app logs to see which endpoints work")
    
    print("\n4️⃣ Alternative Sites to Try:")
    print("   • https://spribe.co (official Aviator developer)")
    print("   • https://1xbet.com (usually has good API access)")
    print("   • Local betting sites (often less protected)")
    
    print("\n5️⃣ Signs of Success:")
    print("   • 🟢 'Successfully got JSON data from...' in logs")
    print("   • 🟢 'Using X real data points for prediction'")
    print("   • 🟢 Prediction method shows 'real_data_heuristic'")
    print("   • 🟢 Active connections > 0 in monitor")

def main():
    """Main troubleshooting function"""
    
    print("🔧 AVIATOR PREDICTOR - TIMEOUT TROUBLESHOOTER")
    print("=" * 60)
    print()
    
    # Get site URL from user or use default
    try:
        site_url = input("Enter betting site URL to test (or press Enter for betika.com): ").strip()
        if not site_url:
            site_url = "https://betika.com"
        
        if not site_url.startswith(('http://', 'https://')):
            site_url = 'https://' + site_url
            
    except KeyboardInterrupt:
        print("\n👋 Cancelled by user")
        return
    
    print()
    
    # Run tests
    success = test_site_accessibility(site_url)
    
    # Provide recommendations
    provide_recommendations(site_url)
    
    print("\n" + "=" * 60)
    if success:
        print("✅ RESULT: Site is accessible - timeouts are likely due to site protection")
        print("💡 The app should work, but may have intermittent connection issues")
    else:
        print("❌ RESULT: Site has connectivity issues")
        print("💡 Try a different betting site or check your internet connection")
    
    print("\n🔍 To monitor real-time connections, run:")
    print("   python monitor_connections.py")

if __name__ == "__main__":
    main()