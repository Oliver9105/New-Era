"""
Feature 5: Network Inspection
Advanced network traffic analysis and monitoring capabilities
"""

import socket
import requests
import psutil
import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import subprocess
import re
from collections import defaultdict, deque
import asyncio
import aiohttp
from urllib.parse import urlparse
import ssl
import dns.resolver

logger = logging.getLogger(__name__)

class NetworkInspector:
    def __init__(self):
        self.active = False
        self.monitoring_active = False
        self.traffic_data = deque(maxlen=1000)
        self.monitored_targets = {}
        self.network_stats = {
            'total_requests': 0,
            'successful_requests': 0,
            'failed_requests': 0,
            'average_response_time': 0,
            'bandwidth_usage': 0
        }
        self.connection_pool = {}
        self.monitoring_thread = None

    def initialize(self):
        """Initialize the network inspector"""
        self.active = True
        self._start_background_monitoring()
        logger.info("Network Inspector initialized")

    def is_active(self):
        return self.active

    def _start_background_monitoring(self):
        """Start background network monitoring"""
        def monitor_loop():
            while self.monitoring_active:
                try:
                    self._collect_network_metrics()
                    time.sleep(5)  # Collect metrics every 5 seconds
                except Exception as e:
                    logger.error(f"Network monitoring error: {e}")
                    time.sleep(10)

        self.monitoring_active = True
        self.monitoring_thread = threading.Thread(target=monitor_loop, daemon=True)
        self.monitoring_thread.start()

    def inspect_target(self, target: str) -> Dict:
        """Inspect a specific target (URL or IP)"""
        try:
            inspection_result = {
                'target': target,
                'timestamp': datetime.utcnow().isoformat(),
                'dns_resolution': {},
                'connectivity_test': {},
                'ssl_analysis': {},
                'port_scan': {},
                'traceroute': {},
                'geolocation': {},
                'performance_metrics': {}
            }

            # Parse target
            parsed_url = self._parse_target(target)
            hostname = parsed_url.get('hostname', target)
            
            # DNS Resolution
            inspection_result['dns_resolution'] = self._perform_dns_lookup(hostname)
            
            # Connectivity Test
            inspection_result['connectivity_test'] = self._test_connectivity(target, parsed_url)
            
            # SSL Analysis (if HTTPS)
            if parsed_url.get('scheme') == 'https':
                inspection_result['ssl_analysis'] = self._analyze_ssl_certificate(hostname)
            
            # Port Scan
            inspection_result['port_scan'] = self._scan_common_ports(hostname)
            
            # Traceroute
            inspection_result['traceroute'] = self._perform_traceroute(hostname)
            
            # Performance Metrics
            inspection_result['performance_metrics'] = self._measure_performance(target)
            
            # Store for monitoring
            self.monitored_targets[target] = {
                'last_inspection': datetime.utcnow(),
                'result': inspection_result
            }
            
            return inspection_result

        except Exception as e:
            logger.error(f"Error inspecting target {target}: {e}")
            return {'error': str(e), 'target': target}

    def _parse_target(self, target: str) -> Dict:
        """Parse target URL or IP address"""
        try:
            if target.startswith(('http://', 'https://')):
                parsed = urlparse(target)
                return {
                    'scheme': parsed.scheme,
                    'hostname': parsed.hostname,
                    'port': parsed.port or (443 if parsed.scheme == 'https' else 80),
                    'path': parsed.path,
                    'full_url': target
                }
            else:
                # Assume it's a hostname or IP
                return {
                    'hostname': target,
                    'port': 80,
                    'scheme': 'http'
                }
        except Exception as e:
            logger.error(f"Error parsing target {target}: {e}")
            return {'hostname': target}

    def _perform_dns_lookup(self, hostname: str) -> Dict:
        """Perform DNS resolution and analysis"""
        try:
            dns_info = {
                'hostname': hostname,
                'a_records': [],
                'aaaa_records': [],
                'mx_records': [],
                'ns_records': [],
                'txt_records': [],
                'resolution_time': 0
            }

            start_time = time.time()
            
            # A records (IPv4)
            try:
                a_records = dns.resolver.resolve(hostname, 'A')
                dns_info['a_records'] = [str(record) for record in a_records]
            except Exception:
                pass

            # AAAA records (IPv6)
            try:
                aaaa_records = dns.resolver.resolve(hostname, 'AAAA')
                dns_info['aaaa_records'] = [str(record) for record in aaaa_records]
            except Exception:
                pass

            # MX records
            try:
                mx_records = dns.resolver.resolve(hostname, 'MX')
                dns_info['mx_records'] = [str(record) for record in mx_records]
            except Exception:
                pass

            # NS records
            try:
                ns_records = dns.resolver.resolve(hostname, 'NS')
                dns_info['ns_records'] = [str(record) for record in ns_records]
            except Exception:
                pass

            # TXT records
            try:
                txt_records = dns.resolver.resolve(hostname, 'TXT')
                dns_info['txt_records'] = [str(record) for record in txt_records]
            except Exception:
                pass

            dns_info['resolution_time'] = time.time() - start_time
            
            return dns_info

        except Exception as e:
            return {'error': f'DNS lookup failed: {e}', 'hostname': hostname}

    def _test_connectivity(self, target: str, parsed_url: Dict) -> Dict:
        """Test connectivity to target"""
        try:
            connectivity_result = {
                'tcp_test': {},
                'http_test': {},
                'ping_test': {}
            }

            hostname = parsed_url.get('hostname', target)
            port = parsed_url.get('port', 80)

            # TCP connection test
            connectivity_result['tcp_test'] = self._test_tcp_connection(hostname, port)
            
            # HTTP request test
            if parsed_url.get('full_url'):
                connectivity_result['http_test'] = self._test_http_request(parsed_url['full_url'])
            
            # Ping test
            connectivity_result['ping_test'] = self._test_ping(hostname)
            
            return connectivity_result

        except Exception as e:
            return {'error': f'Connectivity test failed: {e}'}

    def _test_tcp_connection(self, hostname: str, port: int) -> Dict:
        """Test TCP connection to hostname:port"""
        try:
            start_time = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(10)
            
            result = sock.connect_ex((hostname, port))
            connection_time = time.time() - start_time
            
            sock.close()
            
            return {
                'success': result == 0,
                'connection_time': connection_time,
                'port': port,
                'error_code': result if result != 0 else None
            }
        except Exception as e:
            return {'success': False, 'error': str(e), 'port': port}

    def _test_http_request(self, url: str) -> Dict:
        """Test HTTP request to URL"""
        try:
            start_time = time.time()
            
            response = requests.get(url, timeout=10, allow_redirects=True)
            request_time = time.time() - start_time
            
            return {
                'success': True,
                'status_code': response.status_code,
                'response_time': request_time,
                'content_length': len(response.content),
                'headers': dict(response.headers),
                'final_url': response.url,
                'redirects': len(response.history)
            }
        except requests.exceptions.RequestException as e:
            return {
                'success': False,
                'error': str(e),
                'response_time': time.time() - start_time
            }

    def _test_ping(self, hostname: str) -> Dict:
        """Test ping to hostname"""
        try:
            # Use system ping command
            if psutil.WINDOWS:
                cmd = ['ping', '-n', '4', hostname]
            else:
                cmd = ['ping', '-c', '4', hostname]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            
            ping_stats = self._parse_ping_output(result.stdout)
            
            return {
                'success': result.returncode == 0,
                'statistics': ping_stats,
                'raw_output': result.stdout
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _parse_ping_output(self, output: str) -> Dict:
        """Parse ping command output"""
        try:
            stats = {
                'packets_sent': 0,
                'packets_received': 0,
                'packet_loss_percent': 0,
                'min_time': 0,
                'max_time': 0,
                'avg_time': 0
            }

            # Parse packet statistics
            if 'packets transmitted' in output:
                # Linux/Mac format
                match = re.search(r'(\d+) packets transmitted, (\d+) (?:packets )?received', output)
                if match:
                    stats['packets_sent'] = int(match.group(1))
                    stats['packets_received'] = int(match.group(2))
            elif 'Packets:' in output:
                # Windows format
                sent_match = re.search(r'Sent = (\d+)', output)
                received_match = re.search(r'Received = (\d+)', output)
                if sent_match and received_match:
                    stats['packets_sent'] = int(sent_match.group(1))
                    stats['packets_received'] = int(received_match.group(1))

            # Calculate packet loss
            if stats['packets_sent'] > 0:
                stats['packet_loss_percent'] = ((stats['packets_sent'] - stats['packets_received']) / stats['packets_sent']) * 100

            # Parse timing statistics
            time_match = re.search(r'min/avg/max.*?=\s*([\d.]+)/([\d.]+)/([\d.]+)', output)
            if time_match:
                stats['min_time'] = float(time_match.group(1))
                stats['avg_time'] = float(time_match.group(2))
                stats['max_time'] = float(time_match.group(3))

            return stats
        except Exception:
            return {}

    def _analyze_ssl_certificate(self, hostname: str) -> Dict:
        """Analyze SSL certificate for hostname"""
        try:
            context = ssl.create_default_context()
            
            with socket.create_connection((hostname, 443), timeout=10) as sock:
                with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                    cert = ssock.getpeercert()
                    
                    ssl_info = {
                        'subject': dict(x[0] for x in cert['subject']),
                        'issuer': dict(x[0] for x in cert['issuer']),
                        'version': cert['version'],
                        'serial_number': cert['serialNumber'],
                        'not_before': cert['notBefore'],
                        'not_after': cert['notAfter'],
                        'signature_algorithm': cert.get('signatureAlgorithm'),
                        'san': cert.get('subjectAltName', []),
                        'cipher': ssock.cipher(),
                        'tls_version': ssock.version()
                    }
                    
                    # Check if certificate is valid
                    not_after = datetime.strptime(cert['notAfter'], '%b %d %H:%M:%S %Y %Z')
                    ssl_info['days_until_expiry'] = (not_after - datetime.utcnow()).days
                    ssl_info['is_valid'] = ssl_info['days_until_expiry'] > 0
                    
                    return ssl_info
        except Exception as e:
            return {'error': f'SSL analysis failed: {e}'}

    def _scan_common_ports(self, hostname: str) -> Dict:
        """Scan common ports on hostname"""
        common_ports = [21, 22, 23, 25, 53, 80, 110, 143, 443, 993, 995, 3389, 5432, 3306]
        
        scan_results = {
            'hostname': hostname,
            'open_ports': [],
            'closed_ports': [],
            'scan_time': 0
        }
        
        start_time = time.time()
        
        for port in common_ports:
            try:
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(2)
                result = sock.connect_ex((hostname, port))
                sock.close()
                
                if result == 0:
                    scan_results['open_ports'].append(port)
                else:
                    scan_results['closed_ports'].append(port)
            except Exception:
                scan_results['closed_ports'].append(port)
        
        scan_results['scan_time'] = time.time() - start_time
        return scan_results

    def _perform_traceroute(self, hostname: str) -> Dict:
        """Perform traceroute to hostname"""
        try:
            if psutil.WINDOWS:
                cmd = ['tracert', '-h', '15', hostname]
            else:
                cmd = ['traceroute', '-m', '15', hostname]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            hops = self._parse_traceroute_output(result.stdout)
            
            return {
                'success': result.returncode == 0,
                'hops': hops,
                'total_hops': len(hops),
                'raw_output': result.stdout
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _parse_traceroute_output(self, output: str) -> List[Dict]:
        """Parse traceroute output"""
        hops = []
        lines = output.split('\n')
        
        for line in lines:
            if re.match(r'^\s*\d+', line):
                # Extract hop information
                parts = line.split()
                if len(parts) >= 2:
                    hop_num = parts[0]
                    
                    # Extract IP addresses and timing
                    ips = re.findall(r'\b(?:\d{1,3}\.){3}\d{1,3}\b', line)
                    times = re.findall(r'(\d+(?:\.\d+)?)\s*ms', line)
                    
                    hops.append({
                        'hop_number': int(hop_num),
                        'ip_addresses': ips,
                        'response_times': [float(t) for t in times],
                        'raw_line': line.strip()
                    })
        
        return hops

    def _measure_performance(self, target: str) -> Dict:
        """Measure performance metrics for target"""
        try:
            performance_metrics = {
                'dns_resolution_time': 0,
                'tcp_connection_time': 0,
                'tls_handshake_time': 0,
                'http_request_time': 0,
                'total_time': 0,
                'bandwidth_test': {}
            }

            start_total = time.time()
            
            # DNS resolution time
            start_dns = time.time()
            try:
                parsed = urlparse(target) if target.startswith('http') else None
                hostname = parsed.hostname if parsed else target
                socket.gethostbyname(hostname)
                performance_metrics['dns_resolution_time'] = time.time() - start_dns
            except Exception:
                pass

            # HTTP request timing
            if target.startswith('http'):
                start_http = time.time()
                try:
                    response = requests.get(target, timeout=10)
                    performance_metrics['http_request_time'] = time.time() - start_http
                    performance_metrics['bandwidth_test'] = {
                        'content_size': len(response.content),
                        'transfer_rate': len(response.content) / performance_metrics['http_request_time']
                    }
                except Exception:
                    pass

            performance_metrics['total_time'] = time.time() - start_total
            
            return performance_metrics
        except Exception as e:
            return {'error': f'Performance measurement failed: {e}'}

    def _collect_network_metrics(self):
        """Collect system-wide network metrics"""
        try:
            # Get network I/O statistics
            net_io = psutil.net_io_counters()
            
            # Get network connections
            connections = psutil.net_connections()
            
            # Get network interfaces
            interfaces = psutil.net_if_stats()
            
            metrics = {
                'timestamp': datetime.utcnow().isoformat(),
                'io_stats': {
                    'bytes_sent': net_io.bytes_sent,
                    'bytes_recv': net_io.bytes_recv,
                    'packets_sent': net_io.packets_sent,
                    'packets_recv': net_io.packets_recv,
                    'errin': net_io.errin,
                    'errout': net_io.errout,
                    'dropin': net_io.dropin,
                    'dropout': net_io.dropout
                },
                'active_connections': len(connections),
                'connection_states': {state: len([c for c in connections if c.status == state]) 
                                   for state in set(c.status for c in connections if c.status)},
                'interfaces': {name: {'is_up': stats.isup, 'speed': stats.speed} 
                             for name, stats in interfaces.items()}
            }
            
            self.traffic_data.append(metrics)
            
        except Exception as e:
            logger.error(f"Error collecting network metrics: {e}")

    def get_traffic_analysis(self) -> Dict:
        """Get network traffic analysis"""
        try:
            if len(self.traffic_data) < 2:
                return {'error': 'Insufficient traffic data'}

            recent_data = list(self.traffic_data)[-100:]  # Last 100 samples
            
            analysis = {
                'data_points': len(recent_data),
                'time_range': {
                    'start': recent_data[0]['timestamp'],
                    'end': recent_data[-1]['timestamp']
                },
                'traffic_summary': self._analyze_traffic_patterns(recent_data),
                'connection_analysis': self._analyze_connections(recent_data),
                'bandwidth_utilization': self._calculate_bandwidth_usage(recent_data),
                'anomaly_detection': self._detect_traffic_anomalies(recent_data)
            }
            
            return analysis
        except Exception as e:
            logger.error(f"Error analyzing traffic: {e}")
            return {'error': str(e)}

    def _analyze_traffic_patterns(self, data: List[Dict]) -> Dict:
        """Analyze traffic patterns from collected data"""
        if len(data) < 2:
            return {}

        first = data[0]['io_stats']
        last = data[-1]['io_stats']
        
        return {
            'total_bytes_sent': last['bytes_sent'] - first['bytes_sent'],
            'total_bytes_received': last['bytes_recv'] - first['bytes_recv'],
            'total_packets_sent': last['packets_sent'] - first['packets_sent'],
            'total_packets_received': last['packets_recv'] - first['packets_recv'],
            'error_rate': {
                'input_errors': last['errin'] - first['errin'],
                'output_errors': last['errout'] - first['errout'],
                'dropped_input': last['dropin'] - first['dropin'],
                'dropped_output': last['dropout'] - first['dropout']
            }
        }

    def _analyze_connections(self, data: List[Dict]) -> Dict:
        """Analyze connection patterns"""
        avg_connections = sum(d['active_connections'] for d in data) / len(data)
        max_connections = max(d['active_connections'] for d in data)
        min_connections = min(d['active_connections'] for d in data)
        
        # Analyze connection states
        all_states = set()
        for d in data:
            all_states.update(d['connection_states'].keys())
        
        state_averages = {}
        for state in all_states:
            values = [d['connection_states'].get(state, 0) for d in data]
            state_averages[state] = sum(values) / len(values)
        
        return {
            'average_connections': avg_connections,
            'max_connections': max_connections,
            'min_connections': min_connections,
            'connection_variance': max_connections - min_connections,
            'state_distribution': state_averages
        }

    def _calculate_bandwidth_usage(self, data: List[Dict]) -> Dict:
        """Calculate bandwidth utilization"""
        if len(data) < 2:
            return {}

        # Calculate rates between consecutive samples
        rates = []
        for i in range(1, len(data)):
            prev = data[i-1]['io_stats']
            curr = data[i]['io_stats']
            
            # Parse timestamps
            prev_time = datetime.fromisoformat(data[i-1]['timestamp'])
            curr_time = datetime.fromisoformat(data[i]['timestamp'])
            time_diff = (curr_time - prev_time).total_seconds()
            
            if time_diff > 0:
                sent_rate = (curr['bytes_sent'] - prev['bytes_sent']) / time_diff
                recv_rate = (curr['bytes_recv'] - prev['bytes_recv']) / time_diff
                
                rates.append({
                    'sent_rate': sent_rate,
                    'recv_rate': recv_rate,
                    'total_rate': sent_rate + recv_rate
                })

        if not rates:
            return {}

        avg_sent = sum(r['sent_rate'] for r in rates) / len(rates)
        avg_recv = sum(r['recv_rate'] for r in rates) / len(rates)
        max_total = max(r['total_rate'] for r in rates)
        
        return {
            'average_sent_bps': avg_sent,
            'average_recv_bps': avg_recv,
            'average_total_bps': avg_sent + avg_recv,
            'peak_bandwidth_bps': max_total,
            'samples': len(rates)
        }

    def _detect_traffic_anomalies(self, data: List[Dict]) -> List[Dict]:
        """Detect anomalies in traffic patterns"""
        anomalies = []
        
        if len(data) < 10:
            return anomalies

        # Calculate normal ranges for different metrics
        connections = [d['active_connections'] for d in data]
        avg_connections = sum(connections) / len(connections)
        
        # Simple threshold-based anomaly detection
        threshold_multiplier = 2.0
        
        for i, sample in enumerate(data):
            # Check for connection count anomalies
            if sample['active_connections'] > avg_connections * threshold_multiplier:
                anomalies.append({
                    'type': 'high_connection_count',
                    'timestamp': sample['timestamp'],
                    'value': sample['active_connections'],
                    'threshold': avg_connections * threshold_multiplier,
                    'severity': 'medium'
                })
            
            # Check for error rate anomalies
            error_count = (sample['io_stats']['errin'] + sample['io_stats']['errout'] + 
                          sample['io_stats']['dropin'] + sample['io_stats']['dropout'])
            if error_count > 100:  # Arbitrary threshold
                anomalies.append({
                    'type': 'high_error_rate',
                    'timestamp': sample['timestamp'],
                    'value': error_count,
                    'severity': 'high'
                })

        return anomalies[-20:]  # Return last 20 anomalies

    def monitor_specific_target(self, target: str, interval: int = 60) -> Dict:
        """Start monitoring a specific target"""
        try:
            def monitor_loop():
                while target in self.monitored_targets:
                    try:
                        result = self.inspect_target(target)
                        self.monitored_targets[target]['monitoring_data'] = result
                        time.sleep(interval)
                    except Exception as e:
                        logger.error(f"Error monitoring {target}: {e}")
                        time.sleep(interval * 2)

            # Start monitoring thread
            monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
            monitor_thread.start()
            
            self.monitored_targets[target] = {
                'monitoring_active': True,
                'interval': interval,
                'thread': monitor_thread
            }
            
            return {
                'status': 'monitoring_started',
                'target': target,
                'interval': interval
            }
        except Exception as e:
            return {'error': str(e)}

    def stop_monitoring_target(self, target: str) -> Dict:
        """Stop monitoring a specific target"""
        if target in self.monitored_targets:
            del self.monitored_targets[target]
            return {'status': 'monitoring_stopped', 'target': target}
        else:
            return {'error': f'Target {target} not being monitored'}

    def __del__(self):
        """Cleanup resources"""
        self.monitoring_active = False
        if hasattr(self, 'monitoring_thread') and self.monitoring_thread:
            try:
                self.monitoring_thread.join(timeout=1)
            except Exception:
                pass
