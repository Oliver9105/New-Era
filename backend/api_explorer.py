"""
Feature 1: API Exploration
Advanced API discovery and testing capabilities for aviator platforms
"""

import requests
import json
import time
import asyncio
import aiohttp
from datetime import datetime
from typing import Dict, List, Any
import logging

logger = logging.getLogger(__name__)

class APIExplorer:
    def __init__(self):
        self.active = False
        self.discovered_apis = {}
        self.api_endpoints = {
            'spribe': {
                'base_url': 'https://api.spribe.io/v1/',
                'endpoints': {
                    'game_history': 'aviator/history',
                    'current_game': 'aviator/current',
                    'stats': 'aviator/stats'
                },
                'auth_required': True
            },
            'betgames': {
                'base_url': 'https://api.betgames.tv/v2/',
                'endpoints': {
                    'aviator_data': 'games/aviator/results',
                    'live_data': 'games/aviator/live'
                },
                'auth_required': False
            },
            'evolution': {
                'base_url': 'https://api.evolution.com/v1/',
                'endpoints': {
                    'game_results': 'aviator/results',
                    'game_config': 'aviator/config'
                },
                'auth_required': True
            },
            'pragmatic': {
                'base_url': 'https://api.pragmaticplay.com/v1/',
                'endpoints': {
                    'aviator_rounds': 'aviator/rounds',
                    'multipliers': 'aviator/multipliers'
                },
                'auth_required': True
            }
        }
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'AviatorPredictor/1.0',
            'Accept': 'application/json',
            'Content-Type': 'application/json'
        })

    def initialize(self):
        """Initialize the API Explorer"""
        self.active = True
        logger.info("API Explorer initialized")
        self._discover_apis()

    def is_active(self):
        return self.active

    def _discover_apis(self):
        """Discover available APIs automatically"""
        logger.info("Starting API discovery process...")
        
        for platform, config in self.api_endpoints.items():
            try:
                status = self._test_platform_connectivity(platform, config)
                self.discovered_apis[platform] = {
                    'config': config,
                    'status': status,
                    'last_tested': datetime.utcnow().isoformat(),
                    'response_time': 0
                }
            except Exception as e:
                logger.error(f"Error discovering {platform} API: {e}")

    def _test_platform_connectivity(self, platform: str, config: Dict) -> str:
        """Test connectivity to a platform API"""
        try:
            test_url = config['base_url']
            start_time = time.time()
            response = self.session.get(test_url, timeout=5)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                return 'available'
            elif response.status_code == 401:
                return 'auth_required'
            else:
                return 'limited'
        except requests.exceptions.Timeout:
            return 'timeout'
        except requests.exceptions.ConnectionError:
            return 'unavailable'
        except Exception:
            return 'error'

    def explore_endpoint(self, data: Dict) -> Dict:
        """Explore a specific API endpoint"""
        platform = data.get('platform')
        endpoint = data.get('endpoint')
        method = data.get('method', 'GET')
        params = data.get('params', {})
        headers = data.get('headers', {})

        if platform not in self.api_endpoints:
            return {'error': f'Platform {platform} not found'}

        config = self.api_endpoints[platform]
        if endpoint not in config['endpoints']:
            return {'error': f'Endpoint {endpoint} not found for {platform}'}

        full_url = config['base_url'] + config['endpoints'][endpoint]
        
        try:
            # Merge custom headers with session headers
            request_headers = {**self.session.headers, **headers}
            
            if method == 'GET':
                response = self.session.get(full_url, params=params, headers=request_headers, timeout=10)
            elif method == 'POST':
                response = self.session.post(full_url, json=params, headers=request_headers, timeout=10)
            else:
                return {'error': f'Method {method} not supported'}

            result = {
                'platform': platform,
                'endpoint': endpoint,
                'url': full_url,
                'status_code': response.status_code,
                'response_time': response.elapsed.total_seconds(),
                'headers': dict(response.headers),
                'data': None,
                'error': None
            }

            if response.status_code == 200:
                try:
                    result['data'] = response.json()
                except json.JSONDecodeError:
                    result['data'] = response.text
            else:
                result['error'] = f'HTTP {response.status_code}: {response.text}'

            return result

        except Exception as e:
            return {
                'platform': platform,
                'endpoint': endpoint,
                'error': str(e)
            }

    def get_available_apis(self) -> Dict:
        """Get all discovered APIs and their status"""
        return {
            'discovered_apis': self.discovered_apis,
            'total_platforms': len(self.discovered_apis),
            'active_platforms': len([p for p in self.discovered_apis.values() if p['status'] == 'available']),
            'last_discovery': datetime.utcnow().isoformat()
        }

    def test_platform_api(self, platform: str) -> Dict:
        """Test specific platform API comprehensively"""
        if platform not in self.api_endpoints:
            return {'error': f'Platform {platform} not found'}

        config = self.api_endpoints[platform]
        results = {}

        for endpoint_name, endpoint_path in config['endpoints'].items():
            full_url = config['base_url'] + endpoint_path
            
            try:
                start_time = time.time()
                response = self.session.get(full_url, timeout=5)
                response_time = time.time() - start_time

                results[endpoint_name] = {
                    'url': full_url,
                    'status_code': response.status_code,
                    'response_time': response_time,
                    'available': response.status_code in [200, 401, 403],  # 401/403 means endpoint exists
                    'requires_auth': response.status_code == 401
                }
            except Exception as e:
                results[endpoint_name] = {
                    'url': full_url,
                    'error': str(e),
                    'available': False
                }

        return {
            'platform': platform,
            'base_url': config['base_url'],
            'endpoints': results,
            'overall_status': 'available' if any(r.get('available') for r in results.values()) else 'unavailable'
        }

    async def async_api_test(self, urls: List[str]) -> List[Dict]:
        """Perform asynchronous API testing for multiple URLs"""
        async def test_url(session, url):
            try:
                async with session.get(url, timeout=5) as response:
                    return {
                        'url': url,
                        'status': response.status,
                        'response_time': time.time(),
                        'available': response.status < 400
                    }
            except Exception as e:
                return {
                    'url': url,
                    'error': str(e),
                    'available': False
                }

        async with aiohttp.ClientSession() as session:
            tasks = [test_url(session, url) for url in urls]
            results = await asyncio.gather(*tasks)
            return results

    def batch_api_exploration(self, targets: List[Dict]) -> List[Dict]:
        """Explore multiple APIs in batch"""
        results = []
        for target in targets:
            result = self.explore_endpoint(target)
            results.append(result)
            time.sleep(0.1)  # Rate limiting
        return results

    def get_api_documentation(self, platform: str) -> Dict:
        """Generate API documentation for discovered endpoints"""
        if platform not in self.discovered_apis:
            return {'error': f'Platform {platform} not found'}

        config = self.api_endpoints[platform]
        documentation = {
            'platform': platform,
            'base_url': config['base_url'],
            'authentication': 'required' if config['auth_required'] else 'not_required',
            'endpoints': {}
        }

        for endpoint_name, endpoint_path in config['endpoints'].items():
            documentation['endpoints'][endpoint_name] = {
                'path': endpoint_path,
                'full_url': config['base_url'] + endpoint_path,
                'methods': ['GET', 'POST'],  # Default methods
                'description': f'API endpoint for {endpoint_name.replace("_", " ")}',
                'parameters': 'To be discovered through exploration'
            }

        return documentation

    def monitor_api_health(self) -> Dict:
        """Monitor health of all discovered APIs"""
        health_status = {}
        
        for platform in self.discovered_apis:
            try:
                status = self.test_platform_api(platform)
                available_endpoints = sum(1 for ep in status['endpoints'].values() if ep.get('available', False))
                total_endpoints = len(status['endpoints'])
                
                health_status[platform] = {
                    'status': status['overall_status'],
                    'available_endpoints': available_endpoints,
                    'total_endpoints': total_endpoints,
                    'health_percentage': (available_endpoints / total_endpoints) * 100 if total_endpoints > 0 else 0,
                    'last_checked': datetime.utcnow().isoformat()
                }
            except Exception as e:
                health_status[platform] = {
                    'status': 'error',
                    'error': str(e),
                    'last_checked': datetime.utcnow().isoformat()
                }

        return {
            'overall_health': health_status,
            'healthy_platforms': len([p for p in health_status.values() if p.get('status') == 'available']),
            'total_platforms': len(health_status)
        }
