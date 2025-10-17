"""
Feature 8: Custom Scripting
Flexible prediction algorithms and custom script execution
"""

import ast
import sys
import json
import time
import importlib
import traceback
from datetime import datetime, timedelta
from typing import Dict, List, Any, Callable, Optional
import logging
import os
import hashlib
import threading
from collections import defaultdict
import pickle
import inspect

logger = logging.getLogger(__name__)

class ScriptEngine:
    def __init__(self):
        self.active = False
        self.script_library = {}
        self.execution_history = []
        self.custom_functions = {}
        self.script_cache = {}
        self.execution_stats = {
            'total_executions': 0,
            'successful_executions': 0,
            'failed_executions': 0,
            'average_execution_time': 0
        }
        self.safe_builtins = {
            'abs', 'all', 'any', 'bin', 'bool', 'chr', 'complex', 'dict',
            'divmod', 'enumerate', 'filter', 'float', 'format', 'frozenset',
            'hex', 'id', 'int', 'isinstance', 'issubclass', 'iter', 'len',
            'list', 'map', 'max', 'min', 'oct', 'ord', 'pow', 'range',
            'repr', 'reversed', 'round', 'set', 'slice', 'sorted', 'str',
            'sum', 'tuple', 'type', 'zip'
        }
        self.allowed_modules = {
            'math', 'random', 'datetime', 'json', 'statistics', 'itertools',
            'collections', 'functools', 'operator'
        }
        
    def initialize(self):
        """Initialize the script engine"""
        self.active = True
        self._load_default_scripts()
        logger.info("Custom Script Engine initialized")
        
    def is_active(self):
        return self.active
        
    def _load_default_scripts(self):
        """Load default prediction and analysis scripts"""
        # Simple Moving Average Predictor
        sma_script = """
def simple_moving_average_predictor(data, window=10):
    import statistics
    if len(data) < window:
        return statistics.mean(data) if data else 1.0
    
    recent_data = data[-window:]
    return statistics.mean(recent_data)

result = simple_moving_average_predictor(historical_multipliers, 10)
"""
        
        # Trend Analysis Script
        trend_script = """
def analyze_trend(data, short_window=5, long_window=20):
    import statistics
    if len(data) < long_window:
        return {'trend': 'neutral', 'strength': 0}
    
    short_avg = statistics.mean(data[-short_window:])
    long_avg = statistics.mean(data[-long_window:])
    
    if short_avg > long_avg * 1.05:
        trend = 'bullish'
        strength = (short_avg / long_avg - 1) * 100
    elif short_avg < long_avg * 0.95:
        trend = 'bearish'
        strength = (1 - short_avg / long_avg) * 100
    else:
        trend = 'neutral'
        strength = 0
    
    return {'trend': trend, 'strength': min(strength, 100)}

result = analyze_trend(historical_multipliers)
"""
        
        # Pattern Recognition Script
        pattern_script = """
def find_patterns(data, pattern_length=3):
    patterns = {}
    
    for i in range(len(data) - pattern_length + 1):
        pattern = tuple(round(x, 1) for x in data[i:i + pattern_length])
        if pattern in patterns:
            patterns[pattern] += 1
        else:
            patterns[pattern] = 1
    
    # Return most common patterns
    sorted_patterns = sorted(patterns.items(), key=lambda x: x[1], reverse=True)
    return sorted_patterns[:5]

result = find_patterns(historical_multipliers)
"""
        
        # Volatility Calculator
        volatility_script = """
def calculate_volatility(data, window=20):
    import statistics
    import math
    
    if len(data) < window:
        return 0
    
    recent_data = data[-window:]
    
    # Calculate returns
    returns = []
    for i in range(1, len(recent_data)):
        return_val = (recent_data[i] - recent_data[i-1]) / recent_data[i-1]
        returns.append(return_val)
    
    if not returns:
        return 0
    
    volatility = statistics.stdev(returns) * math.sqrt(len(returns))
    return volatility

result = calculate_volatility(historical_multipliers)
"""
        
        # Risk Assessment Script
        risk_script = """
def assess_risk(current_multiplier, historical_data, risk_threshold=2.0):
    import statistics
    
    if not historical_data:
        return {'risk_level': 'unknown', 'risk_score': 0}
    
    avg = statistics.mean(historical_data)
    std_dev = statistics.stdev(historical_data) if len(historical_data) > 1 else 0
    
    z_score = (current_multiplier - avg) / std_dev if std_dev > 0 else 0
    
    if abs(z_score) > risk_threshold:
        risk_level = 'high'
        risk_score = min(abs(z_score) / risk_threshold, 1.0)
    elif abs(z_score) > risk_threshold / 2:
        risk_level = 'medium'
        risk_score = abs(z_score) / risk_threshold
    else:
        risk_level = 'low'
        risk_score = abs(z_score) / risk_threshold
    
    return {
        'risk_level': risk_level,
        'risk_score': risk_score,
        'z_score': z_score
    }

# Assume current_multiplier is passed as parameter
result = assess_risk(current_multiplier, historical_multipliers)
"""
        
        # Store default scripts
        self.script_library = {
            'simple_moving_average': {
                'code': sma_script,
                'description': 'Simple moving average predictor',
                'category': 'prediction',
                'parameters': ['window'],
                'created': datetime.utcnow().isoformat()
            },
            'trend_analysis': {
                'code': trend_script,
                'description': 'Analyze trend direction and strength',
                'category': 'analysis',
                'parameters': ['short_window', 'long_window'],
                'created': datetime.utcnow().isoformat()
            },
            'pattern_recognition': {
                'code': pattern_script,
                'description': 'Find recurring patterns in data',
                'category': 'analysis',
                'parameters': ['pattern_length'],
                'created': datetime.utcnow().isoformat()
            },
            'volatility_calculator': {
                'code': volatility_script,
                'description': 'Calculate historical volatility',
                'category': 'analysis',
                'parameters': ['window'],
                'created': datetime.utcnow().isoformat()
            },
            'risk_assessment': {
                'code': risk_script,
                'description': 'Assess risk level of current situation',
                'category': 'risk',
                'parameters': ['risk_threshold'],
                'created': datetime.utcnow().isoformat()
            }
        }
        
    def execute_script(self, script_code: str, context: Dict = None, script_name: str = None) -> Dict:
        """Execute a custom script with safety checks"""
        try:
            start_time = time.time()
            
            # Validate script safety
            safety_check = self._validate_script_safety(script_code)
            if not safety_check['safe']:
                return {
                    'success': False,
                    'error': f"Script safety check failed: {safety_check['reason']}",
                    'script_name': script_name
                }
            
            # Prepare execution context
            execution_context = self._prepare_execution_context(context)
            
            # Execute script in restricted environment
            result = self._execute_in_sandbox(script_code, execution_context)
            
            execution_time = time.time() - start_time
            
            # Update statistics
            self.execution_stats['total_executions'] += 1
            if result['success']:
                self.execution_stats['successful_executions'] += 1
            else:
                self.execution_stats['failed_executions'] += 1
                
            # Update average execution time
            total_time = (self.execution_stats['average_execution_time'] * 
                         (self.execution_stats['total_executions'] - 1) + execution_time)
            self.execution_stats['average_execution_time'] = total_time / self.execution_stats['total_executions']
            
            # Store execution history
            execution_record = {
                'timestamp': datetime.utcnow().isoformat(),
                'script_name': script_name,
                'execution_time': execution_time,
                'success': result['success'],
                'result_preview': str(result.get('result', ''))[:100] if result.get('result') else None,
                'error': result.get('error')
            }
            
            self.execution_history.append(execution_record)
            
            # Keep only last 1000 executions
            if len(self.execution_history) > 1000:
                self.execution_history = self.execution_history[-1000:]
            
            return {
                'success': result['success'],
                'result': result.get('result'),
                'error': result.get('error'),
                'execution_time': execution_time,
                'script_name': script_name,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error executing script: {e}")
            return {
                'success': False,
                'error': str(e),
                'script_name': script_name
            }
            
    def _validate_script_safety(self, script_code: str) -> Dict:
        """Validate script for safety (no dangerous operations)"""
        try:
            # Parse the script to AST
            tree = ast.parse(script_code)
            
            # Check for dangerous operations
            dangerous_patterns = [
                'import os', 'import sys', 'import subprocess', 'import socket',
                'open(', 'file(', 'execfile(', 'eval(', 'exec(',
                '__import__', 'getattr', 'setattr', 'delattr',
                'globals(', 'locals(', 'vars(',
                'exit(', 'quit('
            ]
            
            script_lower = script_code.lower()
            for pattern in dangerous_patterns:
                if pattern in script_lower:
                    return {
                        'safe': False,
                        'reason': f"Dangerous pattern detected: {pattern}"
                    }
            
            # Check AST nodes for dangerous operations
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name not in self.allowed_modules:
                            return {
                                'safe': False,
                                'reason': f"Unauthorized import: {alias.name}"
                            }
                elif isinstance(node, ast.ImportFrom):
                    if node.module not in self.allowed_modules:
                        return {
                            'safe': False,
                            'reason': f"Unauthorized import from: {node.module}"
                        }
            
            return {'safe': True}
            
        except SyntaxError as e:
            return {
                'safe': False,
                'reason': f"Syntax error: {e}"
            }
        except Exception as e:
            return {
                'safe': False,
                'reason': f"Validation error: {e}"
            }
            
    def _prepare_execution_context(self, context: Dict = None) -> Dict:
        """Prepare safe execution context with available data"""
        # Default context with sample data
        execution_context = {
            'historical_multipliers': [1.2, 1.8, 2.5, 1.1, 3.2, 1.9, 2.1, 1.5, 4.1, 1.7],
            'current_multiplier': 2.3,
            'recent_patterns': [],
            'time': time,
            'datetime': datetime,
            'math': __import__('math'),
            'random': __import__('random'),
            'statistics': __import__('statistics'),
            'json': __import__('json')
        }
        
        # Add custom context if provided
        if context:
            execution_context.update(context)
            
        return execution_context
        
    def _execute_in_sandbox(self, script_code: str, context: Dict) -> Dict:
        """Execute script in a sandboxed environment"""
        try:
            # Create restricted globals
            restricted_globals = {
                '__builtins__': {name: getattr(__builtins__, name) 
                               for name in self.safe_builtins 
                               if hasattr(__builtins__, name)}
            }
            
            # Add allowed modules and context
            restricted_globals.update(context)
            
            # Create local namespace
            local_namespace = {}
            
            # Execute the script
            exec(script_code, restricted_globals, local_namespace)
            
            # Get result (look for 'result' variable)
            result = local_namespace.get('result', 'No result variable found')
            
            return {
                'success': True,
                'result': result,
                'local_vars': {k: v for k, v in local_namespace.items() 
                             if not k.startswith('_')}
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'traceback': traceback.format_exc()
            }
            
    def add_script_to_library(self, name: str, script_data: Dict) -> Dict:
        """Add a new script to the library"""
        try:
            # Validate script
            if 'code' not in script_data:
                return {'success': False, 'error': 'Script code is required'}
                
            safety_check = self._validate_script_safety(script_data['code'])
            if not safety_check['safe']:
                return {
                    'success': False,
                    'error': f"Script safety check failed: {safety_check['reason']}"
                }
            
            # Add metadata
            script_entry = {
                'code': script_data['code'],
                'description': script_data.get('description', 'Custom script'),
                'category': script_data.get('category', 'custom'),
                'parameters': script_data.get('parameters', []),
                'created': datetime.utcnow().isoformat(),
                'author': script_data.get('author', 'user'),
                'version': script_data.get('version', '1.0')
            }
            
            self.script_library[name] = script_entry
            
            logger.info(f"Added script to library: {name}")
            
            return {
                'success': True,
                'script_name': name,
                'added_at': script_entry['created']
            }
            
        except Exception as e:
            logger.error(f"Error adding script to library: {e}")
            return {'success': False, 'error': str(e)}
            
    def execute_library_script(self, script_name: str, parameters: Dict = None) -> Dict:
        """Execute a script from the library"""
        try:
            if script_name not in self.script_library:
                return {'success': False, 'error': f'Script {script_name} not found in library'}
                
            script_entry = self.script_library[script_name]
            script_code = script_entry['code']
            
            # Prepare context with parameters
            context = self._prepare_execution_context(parameters)
            
            return self.execute_script(script_code, context, script_name)
            
        except Exception as e:
            logger.error(f"Error executing library script: {e}")
            return {'success': False, 'error': str(e)}
            
    def get_script_library(self) -> Dict:
        """Get all scripts in the library"""
        library_info = {}
        
        for name, script in self.script_library.items():
            library_info[name] = {
                'description': script['description'],
                'category': script['category'],
                'parameters': script['parameters'],
                'created': script['created'],
                'author': script.get('author', 'system'),
                'version': script.get('version', '1.0')
            }
            
        return {
            'total_scripts': len(self.script_library),
            'scripts': library_info,
            'categories': list(set(script['category'] for script in self.script_library.values())),
            'execution_stats': self.execution_stats
        }
        
    def remove_script_from_library(self, script_name: str) -> Dict:
        """Remove a script from the library"""
        try:
            if script_name not in self.script_library:
                return {'success': False, 'error': f'Script {script_name} not found'}
                
            del self.script_library[script_name]
            
            logger.info(f"Removed script from library: {script_name}")
            
            return {
                'success': True,
                'script_name': script_name,
                'removed_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error removing script: {e}")
            return {'success': False, 'error': str(e)}
            
    def batch_execute_scripts(self, script_executions: List[Dict]) -> Dict:
        """Execute multiple scripts in batch"""
        try:
            results = []
            
            for execution in script_executions:
                script_name = execution.get('script_name')
                parameters = execution.get('parameters', {})
                custom_code = execution.get('custom_code')
                
                if custom_code:
                    result = self.execute_script(custom_code, parameters, script_name)
                elif script_name:
                    result = self.execute_library_script(script_name, parameters)
                else:
                    result = {'success': False, 'error': 'No script specified'}
                    
                results.append({
                    'script_name': script_name,
                    'result': result
                })
                
            successful_executions = len([r for r in results if r['result']['success']])
            
            return {
                'success': True,
                'total_executions': len(script_executions),
                'successful_executions': successful_executions,
                'failed_executions': len(script_executions) - successful_executions,
                'results': results
            }
            
        except Exception as e:
            logger.error(f"Error in batch execution: {e}")
            return {'success': False, 'error': str(e)}
            
    def create_prediction_ensemble(self, script_names: List[str], weights: List[float] = None) -> Dict:
        """Create an ensemble predictor from multiple scripts"""
        try:
            if not weights:
                weights = [1.0 / len(script_names)] * len(script_names)
                
            if len(weights) != len(script_names):
                return {'success': False, 'error': 'Weights must match number of scripts'}
                
            ensemble_code = f"""
# Ensemble Predictor
import statistics

script_names = {script_names}
weights = {weights}

predictions = []
for script_name in script_names:
    # This would execute each script and collect predictions
    # For demo, using sample predictions
    predictions.append(2.0 + len(script_name) * 0.1)

# Weighted average
weighted_sum = sum(pred * weight for pred, weight in zip(predictions, weights))
result = weighted_sum
"""
            
            ensemble_script = {
                'code': ensemble_code,
                'description': f'Ensemble of {len(script_names)} prediction scripts',
                'category': 'ensemble',
                'parameters': ['script_names', 'weights'],
                'component_scripts': script_names,
                'weights': weights
            }
            
            ensemble_name = f"ensemble_{int(time.time())}"
            self.script_library[ensemble_name] = ensemble_script
            
            return {
                'success': True,
                'ensemble_name': ensemble_name,
                'component_scripts': script_names,
                'weights': weights
            }
            
        except Exception as e:
            logger.error(f"Error creating ensemble: {e}")
            return {'success': False, 'error': str(e)}
            
    def get_execution_analytics(self) -> Dict:
        """Get analytics about script executions"""
        try:
            # Analyze execution history
            recent_executions = self.execution_history[-100:]  # Last 100 executions
            
            if not recent_executions:
                return {'error': 'No execution history available'}
                
            # Success rate by script
            script_performance = defaultdict(lambda: {'total': 0, 'successful': 0})
            
            for execution in recent_executions:
                script_name = execution.get('script_name', 'unknown')
                script_performance[script_name]['total'] += 1
                if execution['success']:
                    script_performance[script_name]['successful'] += 1
                    
            # Calculate success rates
            performance_summary = {}
            for script, stats in script_performance.items():
                performance_summary[script] = {
                    'total_executions': stats['total'],
                    'successful_executions': stats['successful'],
                    'success_rate': stats['successful'] / stats['total'] if stats['total'] > 0 else 0,
                    'average_execution_time': 0  # Would calculate from execution times
                }
                
            return {
                'overall_stats': self.execution_stats,
                'script_performance': performance_summary,
                'recent_execution_count': len(recent_executions),
                'most_used_scripts': sorted(script_performance.items(), 
                                          key=lambda x: x[1]['total'], reverse=True)[:5]
            }
            
        except Exception as e:
            logger.error(f"Error getting execution analytics: {e}")
            return {'error': str(e)}
            
    def export_script_library(self) -> Dict:
        """Export script library for backup/sharing"""
        return {
            'export_timestamp': datetime.utcnow().isoformat(),
            'script_library': self.script_library,
            'execution_stats': self.execution_stats,
            'version': '1.0'
        }
        
    def import_script_library(self, library_data: Dict) -> Dict:
        """Import script library from exported data"""
        try:
            if 'script_library' not in library_data:
                return {'success': False, 'error': 'Invalid library data'}
                
            imported_scripts = 0
            skipped_scripts = 0
            
            for name, script in library_data['script_library'].items():
                # Validate each script
                safety_check = self._validate_script_safety(script['code'])
                if safety_check['safe']:
                    self.script_library[name] = script
                    imported_scripts += 1
                else:
                    skipped_scripts += 1
                    logger.warning(f"Skipped unsafe script: {name}")
                    
            return {
                'success': True,
                'imported_scripts': imported_scripts,
                'skipped_scripts': skipped_scripts,
                'total_scripts_in_library': len(self.script_library)
            }
            
        except Exception as e:
            logger.error(f"Error importing script library: {e}")
            return {'success': False, 'error': str(e)}