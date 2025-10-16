#!/usr/bin/env python3
"""
Real Network Inspector - Actual browser automation for network traffic analysis
Analyzes real betting sites to discover WebSocket and API endpoints
Author: MiniMax Agent
"""

import asyncio
import json
import os
import re
from typing import Dict, List, Any, Optional
from urllib.parse import urlparse, urljoin
import requests
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.common.exceptions import TimeoutException, WebDriverException
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
import time
import logging

logger = logging.getLogger(__name__)

class RealNetworkInspector:
    """
    Real network traffic inspector using browser automation
    Discovers actual WebSocket and API endpoints from betting sites
    """
    
    def __init__(self):
        self.driver = None
        self.network_logs = []
        self.websocket_urls = set()
        self.api_endpoints = set()
        self.discovered_data = {}
        
    def initialize(self):
        """Initialize the network inspector"""
        logger.info("Real Network Inspector initialized")
        
    def _setup_chrome_driver(self, headless=True):
        """Setup Chrome driver with network logging enabled"""
        try:
            # Try to find available browser
            chrome_paths = [
                '/usr/bin/google-chrome',
                '/usr/bin/chromium-browser',
                '/usr/bin/chromium',
                '/snap/bin/chromium'
            ]
            
            chrome_path = None
            for path in chrome_paths:
                if os.path.exists(path):
                    chrome_path = path
                    break
            
            if not chrome_path:
                logger.warning("No Chrome/Chromium browser found")
                return False
            
            chrome_options = Options()
            chrome_options.binary_location = chrome_path
            
            if headless:
                chrome_options.add_argument('--headless')
            chrome_options.add_argument('--no-sandbox')
            chrome_options.add_argument('--disable-dev-shm-usage')
            chrome_options.add_argument('--disable-gpu')
            chrome_options.add_argument('--window-size=1920,1080')
            chrome_options.add_argument('--user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36')
            
            # Enable logging
            chrome_options.add_argument('--enable-logging')
            chrome_options.add_argument('--log-level=0')
            chrome_options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})
            
            self.driver = webdriver.Chrome(options=chrome_options)
            logger.info("Chrome driver initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup Chrome driver: {e}")
            return False
    
    def analyze_site_traffic(self, site_url: str, analysis_time: int = 30) -> Dict[str, Any]:
        """
        Analyze network traffic for a given site URL
        
        Args:
            site_url: URL of the betting site to analyze
            analysis_time: Time in seconds to monitor traffic
            
        Returns:
            Dictionary containing discovered endpoints and analysis data
        """
        logger.info(f"Starting real network analysis for {site_url}")
        
        if not SELENIUM_AVAILABLE:
            logger.warning("Selenium not available, using fallback analysis")
            return self._fallback_analysis(site_url)
        
        if not self._setup_chrome_driver():
            return self._fallback_analysis(site_url)
            
        try:
            # Clear previous data
            self.network_logs.clear()
            self.websocket_urls.clear()
            self.api_endpoints.clear()
            
            # Navigate to the site
            logger.info(f"Navigating to {site_url}")
            self.driver.get(site_url)
            
            # Wait for initial page load
            WebDriverWait(self.driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, "body"))
            )
            
            # Start monitoring network traffic
            start_time = time.time()
            while time.time() - start_time < analysis_time:
                # Get network logs
                logs = self.driver.get_log('performance')
                self.network_logs.extend(logs)
                
                # Look for game-related elements and interactions
                self._analyze_page_elements()
                
                time.sleep(1)
            
            # Process collected network logs
            self._process_network_logs()
            
            # Additional endpoint discovery
            self._discover_additional_endpoints(site_url)
            
            result = {
                'websocket_urls': list(self.websocket_urls),
                'api_endpoints': list(self.api_endpoints),
                'details': {
                    'analysis_time': analysis_time,
                    'total_requests': len(self.network_logs),
                    'site_structure': self._analyze_site_structure(),
                    'javascript_apis': self._extract_javascript_apis()
                }
            }
            
            logger.info(f"Analysis complete: {len(self.websocket_urls)} WebSockets, {len(self.api_endpoints)} APIs found")
            return result
            
        except Exception as e:
            logger.error(f"Network analysis failed: {e}")
            return self._fallback_analysis(site_url)
            
        finally:
            if self.driver:
                self.driver.quit()
    
    def _process_network_logs(self):
        """Process collected network logs to find endpoints"""
        for log_entry in self.network_logs:
            try:
                message = json.loads(log_entry['message'])
                
                if message.get('message', {}).get('method') == 'Network.requestWillBeSent':
                    request_data = message['message']['params']['request']
                    url = request_data.get('url', '')
                    
                    # Detect WebSocket upgrades
                    headers = request_data.get('headers', {})
                    if headers.get('Upgrade') == 'websocket':
                        self.websocket_urls.add(url)
                        logger.info(f"Found WebSocket: {url}")
                    
                    # Detect API endpoints
                    elif self._is_api_endpoint(url):
                        self.api_endpoints.add(url)
                        logger.info(f"Found API endpoint: {url}")
                        
            except (json.JSONDecodeError, KeyError) as e:
                continue
    
    def _is_api_endpoint(self, url: str) -> bool:
        """Determine if a URL is likely an API endpoint"""
        api_patterns = [
            r'/api/',
            r'/ws/',
            r'/socket',
            r'/realtime',
            r'/live',
            r'/game',
            r'/aviator',
            r'/crash',
            r'/multiplier',
            r'/bet',
            r'/odds',
            r'\.json',
            r'/graphql',
            r'/rpc'
        ]
        
        url_lower = url.lower()
        return any(re.search(pattern, url_lower) for pattern in api_patterns)
    
    def _analyze_page_elements(self):
        """Analyze page elements for game-related components"""
        try:
            # Look for common game elements
            game_selectors = [
                '[class*="aviator"]',
                '[class*="crash"]',
                '[class*="multiplier"]',
                '[class*="game"]',
                '[id*="aviator"]',
                '[id*="game"]',
                'canvas',
                '[data-testid*="game"]'
            ]
            
            for selector in game_selectors:
                try:
                    elements = self.driver.find_elements(By.CSS_SELECTOR, selector)
                    if elements:
                        logger.info(f"Found {len(elements)} game elements with selector: {selector}")
                except:
                    continue
                    
        except Exception as e:
            logger.debug(f"Element analysis failed: {e}")
    
    def _analyze_site_structure(self) -> Dict[str, Any]:
        """Analyze the overall site structure"""
        try:
            title = self.driver.title
            current_url = self.driver.current_url
            
            # Get meta information
            meta_elements = self.driver.find_elements(By.TAG_NAME, "meta")
            meta_info = {}
            for meta in meta_elements:
                name = meta.get_attribute("name") or meta.get_attribute("property")
                content = meta.get_attribute("content")
                if name and content:
                    meta_info[name] = content
            
            return {
                'title': title,
                'final_url': current_url,
                'meta_info': meta_info,
                'has_canvas': len(self.driver.find_elements(By.TAG_NAME, "canvas")) > 0
            }
            
        except Exception as e:
            logger.error(f"Site structure analysis failed: {e}")
            return {}
    
    def _extract_javascript_apis(self) -> List[str]:
        """Extract potential API endpoints from JavaScript code"""
        try:
            # Get all script elements
            scripts = self.driver.find_elements(By.TAG_NAME, "script")
            js_apis = set()
            
            api_patterns = [
                r'["\']([^"\']*\/api\/[^"\']*)["\']',
                r'["\']([^"\']*\/ws\/[^"\']*)["\']',
                r'["\']([^"\']*\/socket[^"\']*)["\']',
                r'WebSocket\(["\']([^"\']*)["\']',
                r'fetch\(["\']([^"\']*)["\']',
                r'axios\.[a-z]+\(["\']([^"\']*)["\']'
            ]
            
            for script in scripts:
                try:
                    script_content = script.get_attribute("innerHTML")
                    if script_content:
                        for pattern in api_patterns:
                            matches = re.findall(pattern, script_content)
                            for match in matches:
                                if self._is_api_endpoint(match):
                                    js_apis.add(match)
                except:
                    continue
            
            return list(js_apis)
            
        except Exception as e:
            logger.error(f"JavaScript API extraction failed: {e}")
            return []
    
    def _discover_additional_endpoints(self, base_url: str):
        """Try to discover additional endpoints using common patterns"""
        parsed_url = urlparse(base_url)
        base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        # Common endpoint patterns for aviator games
        common_endpoints = [
            '/api/aviator/history',
            '/api/aviator/current',
            '/api/aviator/bet',
            '/api/game/aviator',
            '/api/crash/data',
            '/api/live/aviator',
            '/ws/aviator',
            '/socket.io/',
            '/websocket/aviator',
            '/realtime/aviator',
            '/graphql'
        ]
        
        for endpoint in common_endpoints:
            full_url = urljoin(base_domain, endpoint)
            try:
                # Quick HEAD request to check if endpoint exists
                response = requests.head(full_url, timeout=5, allow_redirects=True)
                if response.status_code < 400:
                    self.api_endpoints.add(full_url)
                    logger.info(f"Discovered endpoint: {full_url}")
            except:
                continue
    
    def _fallback_analysis(self, site_url: str) -> Dict[str, Any]:
        """Fallback analysis when browser automation fails"""
        logger.warning("Using fallback analysis due to browser automation failure")
        
        parsed_url = urlparse(site_url)
        base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        # Try basic HTTP analysis
        try:
            response = requests.get(site_url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
            })
            
            # Simple pattern matching on HTML content
            content = response.text.lower()
            potential_endpoints = set()
            
            # Look for common patterns in HTML
            ws_patterns = [r'ws://[^\s"\'">]+', r'wss://[^\s"\'">]+']
            api_patterns = [r'/api/[^\s"\'">]+', r'/ws/[^\s"\'">]+']
            
            for pattern in ws_patterns + api_patterns:
                matches = re.findall(pattern, content)
                for match in matches:
                    if not match.startswith(('http://', 'https://', 'ws://', 'wss://')):
                        match = urljoin(base_domain, match)
                    potential_endpoints.add(match)
            
            return {
                'websocket_urls': [url for url in potential_endpoints if url.startswith(('ws://', 'wss://'))],
                'api_endpoints': [url for url in potential_endpoints if not url.startswith(('ws://', 'wss://'))],
                'details': {
                    'analysis_method': 'fallback_http',
                    'response_status': response.status_code,
                    'content_length': len(content)
                }
            }
            
        except Exception as e:
            logger.error(f"Fallback analysis also failed: {e}")
            return {
                'websocket_urls': [],
                'api_endpoints': [],
                'details': {'error': str(e), 'analysis_method': 'failed'}
            }
