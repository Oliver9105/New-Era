"""
Feature 11: SDK Integration
Multiple platform SDK support and cross-platform compatibility
"""

import requests
import json
import time
import asyncio
import aiohttp
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import hashlib
import hmac
import base64
from urllib.parse import urlencode
import websocket
import threading

logger = logging.getLogger(__name__)

class SDKManager:
    def __init__(self):
        self.active = False
        self.connected_platforms = {}
        self.api_configurations = {
            'spribe': {
                'base_url': 'https://api.spribe.co/v1/',
                'websocket_url': 'wss://api.spribe.co/ws/',
                'auth_type': 'api_key',
                'rate_limit': 100,  # requests per minute
                'endpoints': {
                    'game_data': 'games/aviator/data',
                    'history': 'games/aviator/history',
                    'stats': 'games/aviator/stats'
                }
            },
            'evolution': {
                'base_url': 'https://api.evolution.com/v2/',
                'websocket_url': 'wss://api.evolution.com/live/',
                'auth_type': 'oauth2',
                'rate_limit': 200,
                'endpoints': {
                    'game_data': 'aviator/current',
                    'history': 'aviator/rounds',
                    'config': 'aviator/config'
                }
            },
            'pragmatic': {
                'base_url': 'https://api.pragmaticplay.com/v1/',
                'websocket_url': 'wss://live.pragmaticplay.com/aviator/',
                'auth_type': 'signature',
                'rate_limit': 150,
                'endpoints': {
                    'game_data': 'aviator/live',
                    'history': 'aviator/results',
                    'multipliers': 'aviator/multipliers'
                }
            },
            'betgames': {
                'base_url': 'https://api.betgames.tv/v3/',
                'websocket_url': 'wss://live.betgames.tv/aviator/',
                'auth_type': 'token',
                'rate_limit': 300,
                'endpoints': {
                    'game_data': 'aviator/current',
                    'history': 'aviator/history',
                    'live': 'aviator/live'
                }
            }
        }
        self.websocket_connections = {}
        self.rate_limiters = {}
        self.data_streams = {}
        self.connection_status = {}
        
    def initialize(self):
        """Initialize the SDK manager"""
        self.active = True
        self._initialize_rate_limiters()
        logger.info("SDK Manager initialized")
        
    def is_active(self):
        return self.active
        
    def _initialize_rate_limiters(self):
        """Initialize rate limiters for each platform"""
        for platform, config in self.api_configurations.items():
            self.rate_limiters[platform] = {
                'requests': [],
                'limit': config['rate_limit'],
                'window': 60  # seconds
            }
            
    def get_platforms(self) -> Dict:
        """Get available SDK platforms"""
        platforms_info = {}
        
        for platform, config in self.api_configurations.items():
            platforms_info[platform] = {
                'name': platform,
                'base_url': config['base_url'],
                'auth_type': config['auth_type'],
                'rate_limit': config['rate_limit'],
                'endpoints': list(config['endpoints'].keys()),
                'connected': platform in self.connected_platforms,
                'websocket_available': 'websocket_url' in config
            }
            
        return {
            'available_platforms': platforms_info,
            'total_platforms': len(self.api_configurations),
            'connected_platforms': len(self.connected_platforms)
        }
        
    def connect_platform(self, platform: str, credentials: Dict) -> Dict:
        """Connect to a platform SDK"""
        try:
            if platform not in self.api_configurations:
                return {'success': False, 'error': f'Unknown platform: {platform}'}
                
            config = self.api_configurations[platform]
            
            # Validate credentials based on auth type
            validation_result = self._validate_credentials(platform, credentials)
            if not validation_result['valid']:
                return {'success': False, 'error': validation_result['error']}
                
            # Test connection
            connection_test = self._test_platform_connection(platform, credentials)
            if not connection_test['success']:
                return {'success': False, 'error': f'Connection test failed: {connection_test["error"]}'}
                
            # Store connection
            self.connected_platforms[platform] = {
                'credentials': credentials,
                'connected_at': datetime.utcnow(),
                'last_used': datetime.utcnow(),
                'config': config,
                'connection_id': hashlib.md5(f"{platform}_{time.time()}".encode()).hexdigest()
            }
            
            # Initialize connection status
            self.connection_status[platform] = {
                'status': 'connected',
                'last_request': None,
                'request_count': 0,
                'error_count': 0,
                'last_error': None
            }
            
            logger.info(f"Connected to {platform} SDK")
            
            return {
                'success': True,
                'platform': platform,
                'connected_at': self.connected_platforms[platform]['connected_at'].isoformat(),
                'connection_id': self.connected_platforms[platform]['connection_id'],
                'auth_type': config['auth_type']
            }
            
        except Exception as e:
            logger.error(f"Error connecting to {platform}: {e}")
            return {'success': False, 'error': str(e)}
            
    def _validate_credentials(self, platform: str, credentials: Dict) -> Dict:
        """Validate credentials for platform"""
        try:
            config = self.api_configurations[platform]
            auth_type = config['auth_type']
            
            if auth_type == 'api_key':
                if 'api_key' not in credentials:
                    return {'valid': False, 'error': 'API key required'}
                if not isinstance(credentials['api_key'], str) or len(credentials['api_key']) < 10:
                    return {'valid': False, 'error': 'Invalid API key format'}
                    
            elif auth_type == 'oauth2':
                required_fields = ['client_id', 'client_secret']
                for field in required_fields:
                    if field not in credentials:
                        return {'valid': False, 'error': f'{field} required for OAuth2'}
                        
            elif auth_type == 'signature':
                required_fields = ['api_key', 'secret_key']
                for field in required_fields:
                    if field not in credentials:
                        return {'valid': False, 'error': f'{field} required for signature auth'}
                        
            elif auth_type == 'token':
                if 'token' not in credentials:
                    return {'valid': False, 'error': 'Token required'}
                    
            return {'valid': True}
            
        except Exception as e:
            return {'valid': False, 'error': str(e)}
            
    def _test_platform_connection(self, platform: str, credentials: Dict) -> Dict:
        """Test connection to platform"""
        try:
            config = self.api_configurations[platform]
            
            # Prepare authentication headers
            headers = self._prepare_auth_headers(platform, credentials)
            
            # Test endpoint (usually a simple status or info endpoint)
            test_url = config['base_url'] + 'status'
            
            response = requests.get(test_url, headers=headers, timeout=10)
            
            if response.status_code in [200, 401, 403]:  # 401/403 means endpoint exists but needs auth
                return {'success': True, 'status_code': response.status_code}
            else:
                return {'success': False, 'error': f'HTTP {response.status_code}'}
                
        except requests.exceptions.RequestException as e:
            return {'success': False, 'error': f'Connection error: {e}'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
            
    def _prepare_auth_headers(self, platform: str, credentials: Dict) -> Dict:
        """Prepare authentication headers for platform"""
        config = self.api_configurations[platform]
        auth_type = config['auth_type']
        headers = {'Content-Type': 'application/json'}
        
        if auth_type == 'api_key':
            headers['X-API-Key'] = credentials['api_key']
            
        elif auth_type == 'oauth2':
            # In real implementation, would handle OAuth2 flow
            headers['Authorization'] = f'Bearer {credentials.get("access_token", "")}'
            
        elif auth_type == 'signature':
            # Create signature
            timestamp = str(int(time.time()))
            signature = self._create_signature(platform, credentials, timestamp)
            headers['X-API-Key'] = credentials['api_key']
            headers['X-Timestamp'] = timestamp
            headers['X-Signature'] = signature
            
        elif auth_type == 'token':
            headers['Authorization'] = f'Token {credentials["token"]}'
            
        return headers
        
    def _create_signature(self, platform: str, credentials: Dict, timestamp: str) -> str:
        """Create signature for signature-based authentication"""
        try:
            api_key = credentials['api_key']
            secret_key = credentials['secret_key']
            
            # Create message to sign
            message = f"{api_key}{timestamp}"
            
            # Create HMAC signature
            signature = hmac.new(
                secret_key.encode('utf-8'),
                message.encode('utf-8'),
                hashlib.sha256
            ).hexdigest()
            
            return signature
            
        except Exception as e:
            logger.error(f"Error creating signature: {e}")
            return ""
            
    def _check_rate_limit(self, platform: str) -> bool:
        """Check if request is within rate limit"""
        try:
            rate_limiter = self.rate_limiters[platform]
            current_time = time.time()
            
            # Remove old requests outside the window
            rate_limiter['requests'] = [
                req_time for req_time in rate_limiter['requests']
                if current_time - req_time < rate_limiter['window']
            ]
            
            # Check if under limit
            if len(rate_limiter['requests']) < rate_limiter['limit']:
                rate_limiter['requests'].append(current_time)
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Error checking rate limit: {e}")
            return False
            
    def make_api_request(self, platform: str, endpoint: str, method: str = 'GET', params: Dict = None, data: Dict = None) -> Dict:
        """Make API request to platform"""
        try:
            if platform not in self.connected_platforms:
                return {'success': False, 'error': f'Not connected to {platform}'}
                
            # Check rate limit
            if not self._check_rate_limit(platform):
                return {'success': False, 'error': 'Rate limit exceeded'}
                
            connection = self.connected_platforms[platform]
            config = connection['config']
            
            # Build URL
            if endpoint in config['endpoints']:
                endpoint_path = config['endpoints'][endpoint]
            else:
                endpoint_path = endpoint
                
            url = config['base_url'] + endpoint_path
            
            # Prepare headers
            headers = self._prepare_auth_headers(platform, connection['credentials'])
            
            # Make request
            start_time = time.time()
            
            if method.upper() == 'GET':
                response = requests.get(url, headers=headers, params=params, timeout=30)
            elif method.upper() == 'POST':
                response = requests.post(url, headers=headers, params=params, json=data, timeout=30)
            else:
                return {'success': False, 'error': f'Unsupported method: {method}'}
                
            request_time = time.time() - start_time
            
            # Update connection status
            status = self.connection_status[platform]
            status['last_request'] = datetime.utcnow().isoformat()
            status['request_count'] += 1
            
            # Update last used time
            connection['last_used'] = datetime.utcnow()
            
            if response.status_code == 200:
                try:
                    response_data = response.json()
                except json.JSONDecodeError:
                    response_data = response.text
                    
                return {
                    'success': True,
                    'data': response_data,
                    'status_code': response.status_code,
                    'request_time': request_time,
                    'headers': dict(response.headers)
                }
            else:
                status['error_count'] += 1
                status['last_error'] = f'HTTP {response.status_code}: {response.text}'
                
                return {
                    'success': False,
                    'error': f'HTTP {response.status_code}: {response.text}',
                    'status_code': response.status_code,
                    'request_time': request_time
                }
                
        except requests.exceptions.RequestException as e:
            # Update error status
            if platform in self.connection_status:
                status = self.connection_status[platform]
                status['error_count'] += 1
                status['last_error'] = str(e)
                
            return {'success': False, 'error': f'Request error: {e}'}
        except Exception as e:
            logger.error(f"Error making API request to {platform}: {e}")
            return {'success': False, 'error': str(e)}
            
    def start_websocket_stream(self, platform: str, stream_type: str = 'live_data') -> Dict:
        """Start WebSocket stream for real-time data"""
        try:
            if platform not in self.connected_platforms:
                return {'success': False, 'error': f'Not connected to {platform}'}
                
            config = self.api_configurations[platform]
            
            if 'websocket_url' not in config:
                return {'success': False, 'error': f'WebSocket not supported for {platform}'}
                
            # Create WebSocket connection
            ws_url = config['websocket_url']
            
            def on_message(ws, message):
                try:
                    data = json.loads(message)
                    self._handle_websocket_message(platform, stream_type, data)
                except Exception as e:
                    logger.error(f"Error handling WebSocket message: {e}")
                    
            def on_error(ws, error):
                logger.error(f"WebSocket error for {platform}: {error}")
                
            def on_close(ws, close_status_code, close_msg):
                logger.info(f"WebSocket closed for {platform}")
                if platform in self.websocket_connections:
                    del self.websocket_connections[platform]
                    
            def on_open(ws):
                logger.info(f"WebSocket connected to {platform}")
                # Send authentication or subscription message
                auth_message = self._prepare_websocket_auth(platform)
                if auth_message:
                    ws.send(json.dumps(auth_message))
                    
            # Create WebSocket connection
            ws = websocket.WebSocketApp(
                ws_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            # Start WebSocket in separate thread
            ws_thread = threading.Thread(target=ws.run_forever, daemon=True)
            ws_thread.start()
            
            # Store connection
            self.websocket_connections[platform] = {
                'websocket': ws,
                'thread': ws_thread,
                'stream_type': stream_type,
                'started_at': datetime.utcnow(),
                'message_count': 0
            }
            
            # Initialize data stream
            self.data_streams[platform] = {
                'stream_type': stream_type,
                'messages': [],
                'last_message': None
            }
            
            return {
                'success': True,
                'platform': platform,
                'stream_type': stream_type,
                'websocket_url': ws_url,
                'started_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error starting WebSocket stream: {e}")
            return {'success': False, 'error': str(e)}
            
    def _prepare_websocket_auth(self, platform: str) -> Optional[Dict]:
        """Prepare WebSocket authentication message"""
        try:
            connection = self.connected_platforms[platform]
            credentials = connection['credentials']
            config = connection['config']
            
            if config['auth_type'] == 'api_key':
                return {
                    'type': 'auth',
                    'api_key': credentials['api_key']
                }
            elif config['auth_type'] == 'token':
                return {
                    'type': 'auth',
                    'token': credentials['token']
                }
            # Add other auth types as needed
            
            return None
            
        except Exception as e:
            logger.error(f"Error preparing WebSocket auth: {e}")
            return None
            
    def _handle_websocket_message(self, platform: str, stream_type: str, data: Dict):
        """Handle incoming WebSocket message"""
        try:
            stream = self.data_streams[platform]
            
            # Process message
            processed_data = {
                'timestamp': datetime.utcnow().isoformat(),
                'platform': platform,
                'stream_type': stream_type,
                'data': data
            }
            
            # Store message
            stream['messages'].append(processed_data)
            stream['last_message'] = processed_data
            
            # Keep only last 1000 messages
            if len(stream['messages']) > 1000:
                stream['messages'] = stream['messages'][-1000:]
                
            # Update connection stats
            if platform in self.websocket_connections:
                self.websocket_connections[platform]['message_count'] += 1
                
        except Exception as e:
            logger.error(f"Error handling WebSocket message: {e}")
            
    def stop_websocket_stream(self, platform: str) -> Dict:
        """Stop WebSocket stream"""
        try:
            if platform not in self.websocket_connections:
                return {'success': False, 'error': f'No WebSocket connection for {platform}'}
                
            connection = self.websocket_connections[platform]
            
            # Close WebSocket
            connection['websocket'].close()
            
            # Wait for thread to finish
            connection['thread'].join(timeout=5)
            
            # Remove from connections
            del self.websocket_connections[platform]
            
            return {
                'success': True,
                'platform': platform,
                'stopped_at': datetime.utcnow().isoformat(),
                'message_count': connection['message_count']
            }
            
        except Exception as e:
            logger.error(f"Error stopping WebSocket stream: {e}")
            return {'success': False, 'error': str(e)}
            
    def get_stream_data(self, platform: str, limit: int = 100) -> Dict:
        """Get recent stream data"""
        try:
            if platform not in self.data_streams:
                return {'success': False, 'error': f'No data stream for {platform}'}
                
            stream = self.data_streams[platform]
            recent_messages = stream['messages'][-limit:] if stream['messages'] else []
            
            return {
                'success': True,
                'platform': platform,
                'stream_type': stream['stream_type'],
                'message_count': len(recent_messages),
                'total_messages': len(stream['messages']),
                'messages': recent_messages,
                'last_message': stream['last_message']
            }
            
        except Exception as e:
            logger.error(f"Error getting stream data: {e}")
            return {'success': False, 'error': str(e)}
            
    def disconnect_platform(self, platform: str) -> Dict:
        """Disconnect from platform"""
        try:
            if platform not in self.connected_platforms:
                return {'success': False, 'error': f'Not connected to {platform}'}
                
            # Stop WebSocket if running
            if platform in self.websocket_connections:
                self.stop_websocket_stream(platform)
                
            # Remove connection
            connection_info = self.connected_platforms[platform]
            del self.connected_platforms[platform]
            
            # Clean up status
            if platform in self.connection_status:
                del self.connection_status[platform]
                
            # Clean up data stream
            if platform in self.data_streams:
                del self.data_streams[platform]
                
            return {
                'success': True,
                'platform': platform,
                'disconnected_at': datetime.utcnow().isoformat(),
                'connection_duration': (datetime.utcnow() - connection_info['connected_at']).total_seconds()
            }
            
        except Exception as e:
            logger.error(f"Error disconnecting from {platform}: {e}")
            return {'success': False, 'error': str(e)}
            
    def get_connection_status(self) -> Dict:
        """Get status of all platform connections"""
        try:
            status_info = {}
            
            for platform in self.api_configurations:
                if platform in self.connected_platforms:
                    connection = self.connected_platforms[platform]
                    status = self.connection_status.get(platform, {})
                    
                    status_info[platform] = {
                        'connected': True,
                        'connected_at': connection['connected_at'].isoformat(),
                        'last_used': connection['last_used'].isoformat(),
                        'connection_id': connection['connection_id'],
                        'request_count': status.get('request_count', 0),
                        'error_count': status.get('error_count', 0),
                        'last_request': status.get('last_request'),
                        'last_error': status.get('last_error'),
                        'websocket_active': platform in self.websocket_connections
                    }
                    
                    if platform in self.websocket_connections:
                        ws_info = self.websocket_connections[platform]
                        status_info[platform]['websocket_info'] = {
                            'started_at': ws_info['started_at'].isoformat(),
                            'message_count': ws_info['message_count'],
                            'stream_type': ws_info['stream_type']
                        }
                else:
                    status_info[platform] = {
                        'connected': False,
                        'websocket_active': False
                    }
                    
            return {
                'platform_status': status_info,
                'total_platforms': len(self.api_configurations),
                'connected_platforms': len(self.connected_platforms),
                'active_websockets': len(self.websocket_connections)
            }
            
        except Exception as e:
            logger.error(f"Error getting connection status: {e}")
            return {'error': str(e)}