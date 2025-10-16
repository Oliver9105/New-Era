#!/usr/bin/env python3
"""
Real Data Collector - Actual WebSocket and API data collection
Connects to real betting sites and collects live game data
Author: MiniMax Agent
"""

import asyncio
import json
import time
import threading
from typing import Dict, List, Any, Optional, Callable
from urllib.parse import urlparse
import requests
try:
    import websocket
    from websocket import WebSocketApp
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
import logging
from datetime import datetime, timedelta
import queue
import ssl

logger = logging.getLogger(__name__)

class RealDataCollector:
    """
    Real-time data collector for aviator games
    Handles WebSocket connections and API polling
    """
    
    def __init__(self):
        self.active_connections = {}
        self.data_queue = queue.Queue()
        self.capture_active = False
        self.current_site = None
        self.websocket_threads = {}
        self.api_poll_threads = {}
        self.collected_data = []
        self.connection_callbacks = []
        
    def initialize(self):
        """Initialize the data collector"""
        logger.info("Real Data Collector initialized")
        
    def start_capture(self, site_url: str) -> Dict[str, Any]:
        """
        Start data capture from a betting site
        
        Args:
            site_url: URL of the betting site
            
        Returns:
            Dictionary with capture status and details
        """
        try:
            logger.info(f"Starting real data capture for {site_url}")
            
            self.current_site = site_url
            self.capture_active = True
            
            # Get discovered endpoints (would normally come from network inspector)
            endpoints = self._discover_endpoints(site_url)
            
            capture_id = f"capture_{int(time.time())}"
            connections_started = 0
            
            # Start WebSocket connections
            for ws_url in endpoints.get('websocket_urls', []):
                if self._start_websocket_connection(ws_url, capture_id):
                    connections_started += 1
            
            # Start API polling
            for api_url in endpoints.get('api_endpoints', []):
                if self._start_api_polling(api_url, capture_id):
                    connections_started += 1
            
            if connections_started > 0:
                logger.info(f"Started {connections_started} connections for {site_url}")
                return {
                    'success': True,
                    'capture_id': capture_id,
                    'connections_started': connections_started,
                    'endpoints': endpoints
                }
            else:
                logger.warning(f"No successful connections for {site_url}")
                return {
                    'success': False,
                    'error': 'No valid endpoints found or connections failed',
                    'attempted_endpoints': endpoints
                }
                
        except Exception as e:
            logger.error(f"Failed to start capture: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def stop_capture(self):
        """Stop all active data capture"""
        logger.info("Stopping data capture")
        self.capture_active = False
        
        # Close WebSocket connections
        for ws_id, ws_connection in list(self.active_connections.items()):
            try:
                if hasattr(ws_connection, 'close'):
                    ws_connection.close()
                logger.info(f"Closed connection: {ws_id}")
            except Exception as e:
                logger.error(f"Error closing connection {ws_id}: {e}")
        
        self.active_connections.clear()
        
        # Stop polling threads
        for thread_id, thread in list(self.api_poll_threads.items()):
            try:
                # Threads will stop when capture_active becomes False
                logger.info(f"Signaled stop for polling thread: {thread_id}")
            except Exception as e:
                logger.error(f"Error stopping thread {thread_id}: {e}")
    
    def connect_websocket(self, url: str) -> Dict[str, Any]:
        """
        Connect to a specific WebSocket URL
        
        Args:
            url: WebSocket URL to connect to
            
        Returns:
            Connection result
        """
        try:
            logger.info(f"Attempting WebSocket connection to {url}")
            
            connection_id = f"ws_{int(time.time())}_{hash(url) % 10000}"
            
            if self._start_websocket_connection(url, connection_id):
                return {
                    'success': True,
                    'connection_id': connection_id,
                    'url': url
                }
            else:
                return {
                    'success': False,
                    'error': 'Failed to establish WebSocket connection'
                }
                
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def connect_api(self, url: str) -> Dict[str, Any]:
        """
        Connect to a specific API endpoint
        
        Args:
            url: API URL to connect to
            
        Returns:
            Connection result
        """
        try:
            logger.info(f"Testing API connection to {url}")
            
            # Test the API endpoint
            response = requests.get(url, timeout=10, headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Accept': 'application/json, */*',
                'Accept-Language': 'en-US,en;q=0.9',
                'Referer': self.current_site or url
            })
            
            if response.status_code == 200:
                # Start polling this endpoint
                connection_id = f"api_{int(time.time())}_{hash(url) % 10000}"
                
                if self._start_api_polling(url, connection_id):
                    return {
                        'success': True,
                        'connection_id': connection_id,
                        'url': url,
                        'status_code': response.status_code
                    }
            
            return {
                'success': False,
                'error': f'API returned status code: {response.status_code}',
                'status_code': response.status_code
            }
            
        except Exception as e:
            logger.error(f"API connection failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def get_realtime_data(self) -> Optional[Dict[str, Any]]:
        """
        Get the most recent real-time data
        
        Returns:
            Latest game data or None if no data available
        """
        try:
            if not self.data_queue.empty():
                latest_data = None
                # Get the most recent data from queue
                while not self.data_queue.empty():
                    latest_data = self.data_queue.get_nowait()
                
                if latest_data:
                    # Process and standardize the data
                    processed_data = self._process_raw_data(latest_data)
                    self.collected_data.append(processed_data)
                    return processed_data
            
            # Return None if no new data
            return None
            
        except Exception as e:
            logger.error(f"Error getting realtime data: {e}")
            return None
    
    def get_collected_data(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get collected historical data
        
        Args:
            limit: Maximum number of records to return
            
        Returns:
            List of collected game data
        """
        return self.collected_data[-limit:] if self.collected_data else []
    
    def _discover_endpoints(self, site_url: str) -> Dict[str, List[str]]:
        """
        Discover endpoints for a site (simplified version)
        In a real implementation, this would use the network inspector
        """
        parsed_url = urlparse(site_url)
        base_domain = f"{parsed_url.scheme}://{parsed_url.netloc}"
        
        # Common patterns for aviator games
        potential_websockets = [
            f"wss://{parsed_url.netloc}/ws/aviator",
            f"wss://{parsed_url.netloc}/socket.io/?EIO=4&transport=websocket",
            f"wss://{parsed_url.netloc}/websocket",
            f"ws://{parsed_url.netloc}/ws/game"
        ]
        
        potential_apis = [
            f"{base_domain}/api/aviator/current",
            f"{base_domain}/api/game/aviator",
            f"{base_domain}/api/aviator/history",
            f"{base_domain}/api/live/aviator",
            f"{base_domain}/graphql"
        ]
        
        # Test which endpoints are actually available
        working_websockets = []
        working_apis = []
        
        # Quick test for API endpoints
        for api_url in potential_apis:
            try:
                response = requests.head(api_url, timeout=5, headers={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                })
                if response.status_code < 500:  # Accept even 4xx as potentially valid
                    working_apis.append(api_url)
                    logger.info(f"Found API endpoint: {api_url}")
            except:
                continue
        
        # For WebSockets, we'll try to connect during actual capture
        working_websockets = potential_websockets
        
        return {
            'websocket_urls': working_websockets,
            'api_endpoints': working_apis
        }
    
    def _start_websocket_connection(self, ws_url: str, connection_id: str) -> bool:
        """
        Start a WebSocket connection in a separate thread
        
        Args:
            ws_url: WebSocket URL
            connection_id: Unique identifier for this connection
            
        Returns:
            True if connection started successfully
        """
        try:
            if not WEBSOCKET_AVAILABLE:
                logger.warning("WebSocket library not available, skipping WebSocket connection")
                return False
            def on_message(ws, message):
                try:
                    # Parse message and add to queue
                    data = {
                        'source': 'websocket',
                        'url': ws_url,
                        'timestamp': datetime.now().isoformat(),
                        'raw_message': message,
                        'connection_id': connection_id
                    }
                    
                    # Try to parse JSON
                    try:
                        parsed_message = json.loads(message)
                        data['parsed_message'] = parsed_message
                    except json.JSONDecodeError:
                        data['parsed_message'] = None
                    
                    self.data_queue.put(data)
                    logger.debug(f"Received WebSocket message from {ws_url}")
                    
                except Exception as e:
                    logger.error(f"Error processing WebSocket message: {e}")
            
            def on_error(ws, error):
                logger.error(f"WebSocket error for {ws_url}: {error}")
            
            def on_close(ws, close_status_code, close_msg):
                logger.info(f"WebSocket connection closed for {ws_url}")
                if connection_id in self.active_connections:
                    del self.active_connections[connection_id]
            
            def on_open(ws):
                logger.info(f"WebSocket connection opened for {ws_url}")
                # Send initial messages if needed
                self._send_initial_websocket_messages(ws, ws_url)
            
            # Create WebSocket connection
            ws = WebSocketApp(
                ws_url,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close,
                on_open=on_open,
                header={
                    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                    'Origin': self.current_site or ws_url
                }
            )
            
            # Start connection in separate thread
            def run_websocket():
                try:
                    ws.run_forever(
                        sslopt={"cert_reqs": ssl.CERT_NONE},
                        ping_interval=30,
                        ping_timeout=10
                    )
                except Exception as e:
                    logger.error(f"WebSocket thread error: {e}")
            
            thread = threading.Thread(target=run_websocket, daemon=True)
            thread.start()
            
            self.active_connections[connection_id] = ws
            self.websocket_threads[connection_id] = thread
            
            logger.info(f"Started WebSocket connection: {connection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start WebSocket connection: {e}")
            return False
    
    def _start_api_polling(self, api_url: str, connection_id: str, interval: int = 5) -> bool:
        """
        Start API polling in a separate thread
        
        Args:
            api_url: API URL to poll
            connection_id: Unique identifier for this connection
            interval: Polling interval in seconds
            
        Returns:
            True if polling started successfully
        """
        try:
            def poll_api():
                logger.info(f"Starting API polling for {api_url}")
                
                while self.capture_active:
                    try:
                        response = requests.get(api_url, timeout=10, headers={
                            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                            'Accept': 'application/json, */*',
                            'Referer': self.current_site or api_url
                        })
                        
                        if response.status_code == 200:
                            data = {
                                'source': 'api',
                                'url': api_url,
                                'timestamp': datetime.now().isoformat(),
                                'status_code': response.status_code,
                                'connection_id': connection_id
                            }
                            
                            # Try to parse JSON response
                            try:
                                data['response_data'] = response.json()
                            except json.JSONDecodeError:
                                data['response_text'] = response.text[:1000]  # Limit text length
                            
                            self.data_queue.put(data)
                            logger.debug(f"Polled API data from {api_url}")
                        else:
                            logger.warning(f"API {api_url} returned status {response.status_code}")
                        
                        time.sleep(interval)
                        
                    except Exception as e:
                        logger.error(f"API polling error for {api_url}: {e}")
                        time.sleep(interval * 2)  # Back off on errors
                
                logger.info(f"Stopped API polling for {api_url}")
            
            thread = threading.Thread(target=poll_api, daemon=True)
            thread.start()
            
            self.api_poll_threads[connection_id] = thread
            logger.info(f"Started API polling: {connection_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start API polling: {e}")
            return False
    
    def _send_initial_websocket_messages(self, ws: WebSocketApp, ws_url: str):
        """
        Send initial messages that might be required by the WebSocket
        """
        try:
            # Common initial messages for aviator games
            initial_messages = [
                '{"type":"subscribe","channel":"aviator"}',
                '{"action":"join","room":"aviator"}',
                '{"event":"subscribe","data":{"channel":"game"}}',
                '{"type":"ping"}'
            ]
            
            for message in initial_messages:
                try:
                    ws.send(message)
                    time.sleep(0.1)
                except Exception as e:
                    logger.debug(f"Failed to send initial message: {e}")
                    
        except Exception as e:
            logger.error(f"Error sending initial WebSocket messages: {e}")
    
    def _process_raw_data(self, raw_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process and standardize raw data from various sources
        
        Args:
            raw_data: Raw data from WebSocket or API
            
        Returns:
            Processed and standardized data
        """
        try:
            processed = {
                'timestamp': raw_data.get('timestamp'),
                'source': raw_data.get('source'),
                'url': raw_data.get('url'),
                'connection_id': raw_data.get('connection_id'),
                'game_data': None,
                'raw_data': raw_data
            }
            
            # Extract game-specific data based on source
            if raw_data['source'] == 'websocket':
                processed['game_data'] = self._extract_game_data_from_websocket(raw_data)
            elif raw_data['source'] == 'api':
                processed['game_data'] = self._extract_game_data_from_api(raw_data)
            
            return processed
            
        except Exception as e:
            logger.error(f"Error processing raw data: {e}")
            return raw_data
    
    def _extract_game_data_from_websocket(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract aviator game data from WebSocket messages
        """
        try:
            parsed_message = data.get('parsed_message')
            if not parsed_message:
                return None
            
            # Common patterns for aviator game data
            game_data = {}
            
            # Look for multiplier data
            for key in ['multiplier', 'crash_multiplier', 'coefficient', 'odds']:
                if key in parsed_message:
                    game_data['multiplier'] = parsed_message[key]
                    break
            
            # Look for round/game ID
            for key in ['round_id', 'game_id', 'id', 'roundId']:
                if key in parsed_message:
                    game_data['round_id'] = parsed_message[key]
                    break
            
            # Look for game status
            for key in ['status', 'state', 'phase', 'game_state']:
                if key in parsed_message:
                    game_data['status'] = parsed_message[key]
                    break
            
            # Look for timestamp
            for key in ['timestamp', 'time', 'created_at', 'start_time']:
                if key in parsed_message:
                    game_data['game_timestamp'] = parsed_message[key]
                    break
            
            return game_data if game_data else None
            
        except Exception as e:
            logger.error(f"Error extracting WebSocket game data: {e}")
            return None
    
    def _extract_game_data_from_api(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract aviator game data from API responses
        """
        try:
            response_data = data.get('response_data')
            if not response_data:
                return None
            
            # Handle different API response structures
            game_data = {}
            
            # Check if response has direct game data
            if isinstance(response_data, dict):
                # Extract multiplier
                for key in ['multiplier', 'crash_multiplier', 'coefficient', 'current_multiplier']:
                    if key in response_data:
                        game_data['multiplier'] = response_data[key]
                        break
                
                # Extract round data
                for key in ['round_id', 'game_id', 'current_round']:
                    if key in response_data:
                        game_data['round_id'] = response_data[key]
                        break
                
                # Check for nested data structures
                for nested_key in ['data', 'result', 'game', 'aviator']:
                    if nested_key in response_data and isinstance(response_data[nested_key], dict):
                        nested_data = response_data[nested_key]
                        for key in ['multiplier', 'crash_multiplier', 'coefficient']:
                            if key in nested_data:
                                game_data['multiplier'] = nested_data[key]
                                break
            
            return game_data if game_data else None
            
        except Exception as e:
            logger.error(f"Error extracting API game data: {e}")
            return None
