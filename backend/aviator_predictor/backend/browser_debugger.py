"""
Feature 7: Browser Debugging
Advanced browser debugging and JavaScript injection capabilities
"""

import json
import time
import asyncio
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains
import base64

logger = logging.getLogger(__name__)

class BrowserDebugger:
    def __init__(self):
        self.active = False
        self.debug_sessions = {}
        self.injected_scripts = []
        self.captured_data = []
        self.performance_logs = []
        self.console_logs = []
        self.network_logs = []
        
    def initialize(self):
        """Initialize the browser debugger"""
        self.active = True
        logger.info("Browser Debugger initialized")
        
    def is_active(self):
        return self.active
        
    def start_debug_session(self, url: str, session_id: str = None) -> Dict:
        """Start a browser debugging session"""
        try:
            if not session_id:
                session_id = f"session_{int(time.time())}"
                
            # Setup Chrome options for debugging
            chrome_options = Options()
            chrome_options.add_argument('--enable-logging')
            chrome_options.add_argument('--log-level=0')
            chrome_options.add_argument('--enable-network-service-logging')
            chrome_options.add_argument('--disable-web-security')
            chrome_options.add_argument('--disable-features=VizDisplayCompositor')
            chrome_options.add_experimental_option('useAutomationExtension', False)
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])
            chrome_options.add_experimental_option('loggingPrefs', {
                'browser': 'ALL',
                'driver': 'ALL',
                'performance': 'ALL'
            })
            
            # Start WebDriver
            driver = webdriver.Chrome(options=chrome_options)
            
            # Enable performance logging
            driver.execute_cdp_cmd('Performance.enable', {})
            driver.execute_cdp_cmd('Runtime.enable', {})
            driver.execute_cdp_cmd('Network.enable', {})
            
            # Navigate to URL
            driver.get(url)
            
            # Wait for page to load
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.TAG_NAME, 'body'))
            )
            
            # Store session
            self.debug_sessions[session_id] = {
                'driver': driver,
                'url': url,
                'start_time': datetime.utcnow(),
                'status': 'active',
                'injected_scripts': [],
                'captured_elements': [],
                'console_logs': [],
                'network_requests': []
            }
            
            # Start collecting logs
            self._start_log_collection(session_id)
            
            return {
                'success': True,
                'session_id': session_id,
                'url': url,
                'status': 'active',
                'start_time': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error starting debug session: {e}")
            return {'success': False, 'error': str(e)}
            
    def inject_script(self, script_code: str, session_id: str = None) -> Dict:
        """Inject JavaScript code into browser session"""
        try:
            if session_id:
                if session_id not in self.debug_sessions:
                    return {'success': False, 'error': 'Session not found'}
                sessions_to_inject = [session_id]
            else:
                sessions_to_inject = list(self.debug_sessions.keys())
                
            results = []
            
            for sid in sessions_to_inject:
                session = self.debug_sessions[sid]
                driver = session['driver']
                
                try:
                    # Execute the script
                    result = driver.execute_script(script_code)
                    
                    # Store injection record
                    injection_record = {
                        'timestamp': datetime.utcnow().isoformat(),
                        'script': script_code,
                        'result': result,
                        'session_id': sid
                    }
                    
                    session['injected_scripts'].append(injection_record)
                    self.injected_scripts.append(injection_record)
                    
                    results.append({
                        'session_id': sid,
                        'success': True,
                        'result': result
                    })
                    
                except Exception as e:
                    results.append({
                        'session_id': sid,
                        'success': False,
                        'error': str(e)
                    })
                    
            return {
                'success': True,
                'injection_results': results,
                'total_sessions': len(sessions_to_inject)
            }
            
        except Exception as e:
            logger.error(f"Error injecting script: {e}")
            return {'success': False, 'error': str(e)}
            
    def capture_elements(self, selectors: List[str], session_id: str) -> Dict:
        """Capture specific DOM elements"""
        try:
            if session_id not in self.debug_sessions:
                return {'success': False, 'error': 'Session not found'}
                
            session = self.debug_sessions[session_id]
            driver = session['driver']
            
            captured_elements = []
            
            for selector in selectors:
                try:
                    elements = driver.find_elements(By.CSS_SELECTOR, selector)
                    
                    for i, element in enumerate(elements):
                        element_data = {
                            'selector': selector,
                            'index': i,
                            'tag_name': element.tag_name,
                            'text': element.text,
                            'attributes': {},
                            'location': element.location,
                            'size': element.size,
                            'visible': element.is_displayed(),
                            'enabled': element.is_enabled()
                        }
                        
                        # Get all attributes
                        try:
                            attrs = driver.execute_script(
                                "var items = {}; "
                                "for (index = 0; index < arguments[0].attributes.length; ++index) { "
                                "items[arguments[0].attributes[index].name] = arguments[0].attributes[index].value; "
                                "} return items;", element
                            )
                            element_data['attributes'] = attrs
                        except Exception:
                            pass
                            
                        captured_elements.append(element_data)
                        
                except Exception as e:
                    captured_elements.append({
                        'selector': selector,
                        'error': str(e)
                    })
                    
            # Store captured elements
            capture_record = {
                'timestamp': datetime.utcnow().isoformat(),
                'elements': captured_elements,
                'selectors': selectors
            }
            
            session['captured_elements'].append(capture_record)
            
            return {
                'success': True,
                'captured_elements': captured_elements,
                'total_elements': len(captured_elements)
            }
            
        except Exception as e:
            logger.error(f"Error capturing elements: {e}")
            return {'success': False, 'error': str(e)}
            
    def monitor_console(self, session_id: str) -> Dict:
        """Monitor browser console logs"""
        try:
            if session_id not in self.debug_sessions:
                return {'success': False, 'error': 'Session not found'}
                
            session = self.debug_sessions[session_id]
            driver = session['driver']
            
            # Get console logs
            logs = driver.get_log('browser')
            
            console_entries = []
            for log_entry in logs:
                console_entries.append({
                    'timestamp': log_entry['timestamp'],
                    'level': log_entry['level'],
                    'message': log_entry['message'],
                    'source': log_entry.get('source', 'unknown')
                })
                
            # Store console logs
            session['console_logs'].extend(console_entries)
            
            return {
                'success': True,
                'console_logs': console_entries,
                'total_entries': len(console_entries)
            }
            
        except Exception as e:
            logger.error(f"Error monitoring console: {e}")
            return {'success': False, 'error': str(e)}
            
    def capture_network_traffic(self, session_id: str) -> Dict:
        """Capture network requests and responses"""
        try:
            if session_id not in self.debug_sessions:
                return {'success': False, 'error': 'Session not found'}
                
            session = self.debug_sessions[session_id]
            driver = session['driver']
            
            # Get performance logs (contains network data)
            logs = driver.get_log('performance')
            
            network_requests = []
            for log_entry in logs:
                try:
                    message = json.loads(log_entry['message'])
                    if message['message']['method'].startswith('Network.'):
                        network_requests.append({
                            'timestamp': log_entry['timestamp'],
                            'method': message['message']['method'],
                            'params': message['message'].get('params', {})
                        })
                except Exception:
                    continue
                    
            # Store network requests
            session['network_requests'].extend(network_requests)
            
            return {
                'success': True,
                'network_requests': network_requests,
                'total_requests': len(network_requests)
            }
            
        except Exception as e:
            logger.error(f"Error capturing network traffic: {e}")
            return {'success': False, 'error': str(e)}
            
    def take_screenshot(self, session_id: str, filename: str = None) -> Dict:
        """Take screenshot of current page"""
        try:
            if session_id not in self.debug_sessions:
                return {'success': False, 'error': 'Session not found'}
                
            session = self.debug_sessions[session_id]
            driver = session['driver']
            
            if not filename:
                filename = f"screenshot_{session_id}_{int(time.time())}.png"
                
            # Take screenshot
            screenshot_data = driver.get_screenshot_as_base64()
            
            # Save screenshot
            with open(filename, 'wb') as f:
                f.write(base64.b64decode(screenshot_data))
                
            return {
                'success': True,
                'filename': filename,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error taking screenshot: {e}")
            return {'success': False, 'error': str(e)}
            
    def simulate_user_interaction(self, session_id: str, interactions: List[Dict]) -> Dict:
        """Simulate user interactions (clicks, typing, etc.)"""
        try:
            if session_id not in self.debug_sessions:
                return {'success': False, 'error': 'Session not found'}
                
            session = self.debug_sessions[session_id]
            driver = session['driver']
            
            interaction_results = []
            
            for interaction in interactions:
                try:
                    action_type = interaction.get('type')
                    
                    if action_type == 'click':
                        element = driver.find_element(By.CSS_SELECTOR, interaction['selector'])
                        element.click()
                        interaction_results.append({
                            'type': 'click',
                            'selector': interaction['selector'],
                            'success': True
                        })
                        
                    elif action_type == 'type':
                        element = driver.find_element(By.CSS_SELECTOR, interaction['selector'])
                        element.clear()
                        element.send_keys(interaction['text'])
                        interaction_results.append({
                            'type': 'type',
                            'selector': interaction['selector'],
                            'text': interaction['text'],
                            'success': True
                        })
                        
                    elif action_type == 'scroll':
                        driver.execute_script(f"window.scrollTo(0, {interaction.get('y', 0)});")
                        interaction_results.append({
                            'type': 'scroll',
                            'y': interaction.get('y', 0),
                            'success': True
                        })
                        
                    elif action_type == 'wait':
                        time.sleep(interaction.get('seconds', 1))
                        interaction_results.append({
                            'type': 'wait',
                            'seconds': interaction.get('seconds', 1),
                            'success': True
                        })
                        
                    time.sleep(0.5)  # Small delay between interactions
                    
                except Exception as e:
                    interaction_results.append({
                        'type': action_type,
                        'success': False,
                        'error': str(e)
                    })
                    
            return {
                'success': True,
                'interactions': interaction_results,
                'total_interactions': len(interactions)
            }
            
        except Exception as e:
            logger.error(f"Error simulating interactions: {e}")
            return {'success': False, 'error': str(e)}
            
    def _start_log_collection(self, session_id: str):
        """Start collecting logs for a session"""
        def collect_logs():
            while session_id in self.debug_sessions:
                try:
                    # Collect console logs
                    self.monitor_console(session_id)
                    
                    # Collect network traffic
                    self.capture_network_traffic(session_id)
                    
                    time.sleep(2)  # Collect every 2 seconds
                except Exception as e:
                    logger.error(f"Error in log collection: {e}")
                    time.sleep(5)
                    
        import threading
        log_thread = threading.Thread(target=collect_logs, daemon=True)
        log_thread.start()
        
    def get_session_info(self, session_id: str = None) -> Dict:
        """Get information about debug sessions"""
        try:
            if session_id:
                if session_id not in self.debug_sessions:
                    return {'error': 'Session not found'}
                    
                session = self.debug_sessions[session_id]
                return {
                    'session_id': session_id,
                    'url': session['url'],
                    'start_time': session['start_time'].isoformat(),
                    'status': session['status'],
                    'injected_scripts_count': len(session['injected_scripts']),
                    'captured_elements_count': len(session['captured_elements']),
                    'console_logs_count': len(session['console_logs']),
                    'network_requests_count': len(session['network_requests'])
                }
            else:
                # Return info for all sessions
                sessions_info = {}
                for sid, session in self.debug_sessions.items():
                    sessions_info[sid] = {
                        'url': session['url'],
                        'start_time': session['start_time'].isoformat(),
                        'status': session['status'],
                        'injected_scripts_count': len(session['injected_scripts']),
                        'captured_elements_count': len(session['captured_elements'])
                    }
                    
                return {
                    'total_sessions': len(self.debug_sessions),
                    'active_sessions': len([s for s in self.debug_sessions.values() if s['status'] == 'active']),
                    'sessions': sessions_info
                }
                
        except Exception as e:
            logger.error(f"Error getting session info: {e}")
            return {'error': str(e)}
            
    def close_session(self, session_id: str) -> Dict:
        """Close a debug session"""
        try:
            if session_id not in self.debug_sessions:
                return {'success': False, 'error': 'Session not found'}
                
            session = self.debug_sessions[session_id]
            
            # Close the browser
            try:
                session['driver'].quit()
            except Exception as e:
                logger.warning(f"Error closing browser: {e}")
                
            # Mark session as closed
            session['status'] = 'closed'
            session['end_time'] = datetime.utcnow()
            
            # Remove from active sessions
            del self.debug_sessions[session_id]
            
            return {
                'success': True,
                'session_id': session_id,
                'end_time': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error closing session: {e}")
            return {'success': False, 'error': str(e)}
            
    def get_debug_analytics(self) -> Dict:
        """Get analytics about debugging activities"""
        try:
            total_injections = len(self.injected_scripts)
            total_captures = sum(len(session['captured_elements']) for session in self.debug_sessions.values())
            total_console_logs = sum(len(session['console_logs']) for session in self.debug_sessions.values())
            
            return {
                'total_sessions': len(self.debug_sessions),
                'total_script_injections': total_injections,
                'total_element_captures': total_captures,
                'total_console_logs': total_console_logs,
                'active_sessions': len([s for s in self.debug_sessions.values() if s['status'] == 'active']),
                'last_activity': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error getting debug analytics: {e}")
            return {'error': str(e)}
            
    def __del__(self):
        """Cleanup when object is destroyed"""
        try:
            for session_id in list(self.debug_sessions.keys()):
                self.close_session(session_id)
        except Exception:
            pass