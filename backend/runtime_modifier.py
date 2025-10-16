"""
Feature 6: Runtime Modification
Dynamic algorithm adjustments and parameter tuning capabilities
"""

import json
import threading
import time
from datetime import datetime, timedelta
from typing import Dict, List, Any, Callable, Optional
import logging
import inspect
import ast
import sys
import importlib
from collections import defaultdict
import copy

logger = logging.getLogger(__name__)

class RuntimeModifier:
    def __init__(self):
        self.active = False
        self.runtime_parameters = {
            'prediction_models': {
                'ensemble_weights': [0.3, 0.3, 0.2, 0.2],  # Model weights
                'confidence_threshold': 0.75,
                'max_prediction_history': 1000,
                'model_switching_threshold': 0.6
            },
            'data_collection': {
                'collection_interval': 60,  # seconds
                'retry_attempts': 3,
                'timeout_duration': 10,
                'parallel_sources': 5
            },
            'analysis_settings': {
                'pattern_sensitivity': 0.8,
                'trend_window_size': 50,
                'volatility_threshold': 0.3,
                'anomaly_detection_threshold': 2.0
            },
            'automation': {
                'max_concurrent_tasks': 10,
                'task_timeout': 300,
                'error_retry_delay': 30,
                'health_check_interval': 180
            },
            'network_monitoring': {
                'monitoring_interval': 5,
                'connection_timeout': 10,
                'max_monitored_targets': 50,
                'alert_thresholds': {
                    'response_time': 5000,  # ms
                    'error_rate': 0.05,
                    'connection_count': 1000
                }
            },
            'risk_management': {
                'max_risk_score': 0.8,
                'stop_loss_threshold': -0.2,
                'position_size_limit': 0.1,
                'volatility_adjustment': True
            }
        }
        self.parameter_history = []
        self.modification_callbacks = {}
        self.parameter_constraints = {}
        self.adaptive_adjustments = {}
        self.performance_tracking = {}

    def initialize(self):
        """Initialize the runtime modifier"""
        self.active = True
        self._setup_parameter_constraints()
        self._start_adaptive_monitoring()
        logger.info("Runtime Modifier initialized")

    def is_active(self):
        return self.active

    def _setup_parameter_constraints(self):
        """Setup constraints for parameter modifications"""
        self.parameter_constraints = {
            'prediction_models.ensemble_weights': {
                'type': 'list',
                'min_length': 4,
                'max_length': 4,
                'element_min': 0.0,
                'element_max': 1.0,
                'sum_constraint': 1.0
            },
            'prediction_models.confidence_threshold': {
                'type': 'float',
                'min': 0.0,
                'max': 1.0
            },
            'data_collection.collection_interval': {
                'type': 'int',
                'min': 10,
                'max': 3600
            },
            'analysis_settings.pattern_sensitivity': {
                'type': 'float',
                'min': 0.1,
                'max': 1.0
            },
            'network_monitoring.alert_thresholds.response_time': {
                'type': 'int',
                'min': 100,
                'max': 30000
            }
        }

    def modify_parameters(self, modifications: Dict) -> Dict:
        """Modify runtime parameters with validation"""
        try:
            modification_result = {
                'timestamp': datetime.utcnow().isoformat(),
                'modifications': [],
                'errors': [],
                'rollback_data': {}
            }

            # Store original values for potential rollback
            original_parameters = copy.deepcopy(self.runtime_parameters)
            modification_result['rollback_data'] = original_parameters

            for parameter_path, new_value in modifications.items():
                try:
                    # Validate the modification
                    validation_result = self._validate_parameter_modification(parameter_path, new_value)
                    
                    if validation_result['valid']:
                        # Apply the modification
                        old_value = self._get_parameter_value(parameter_path)
                        self._set_parameter_value(parameter_path, new_value)
                        
                        modification_result['modifications'].append({
                            'parameter': parameter_path,
                            'old_value': old_value,
                            'new_value': new_value,
                            'timestamp': datetime.utcnow().isoformat()
                        })
                        
                        # Trigger callbacks if registered
                        self._trigger_modification_callbacks(parameter_path, old_value, new_value)
                        
                        logger.info(f"Modified parameter {parameter_path}: {old_value} -> {new_value}")
                    else:
                        modification_result['errors'].append({
                            'parameter': parameter_path,
                            'error': validation_result['error'],
                            'attempted_value': new_value
                        })
                        logger.warning(f"Failed to modify {parameter_path}: {validation_result['error']}")

                except Exception as e:
                    modification_result['errors'].append({
                        'parameter': parameter_path,
                        'error': str(e),
                        'attempted_value': new_value
                    })
                    logger.error(f"Error modifying {parameter_path}: {e}")

            # Store modification history
            self.parameter_history.append(modification_result)
            
            # Keep only last 100 modifications
            if len(self.parameter_history) > 100:
                self.parameter_history = self.parameter_history[-100:]

            return {
                'success': len(modification_result['modifications']) > 0,
                'total_modifications': len(modification_result['modifications']),
                'total_errors': len(modification_result['errors']),
                'details': modification_result
            }

        except Exception as e:
            logger.error(f"Error in modify_parameters: {e}")
            return {'success': False, 'error': str(e)}

    def _validate_parameter_modification(self, parameter_path: str, new_value: Any) -> Dict:
        """Validate a parameter modification against constraints"""
        try:
            # Check if constraint exists for this parameter
            if parameter_path not in self.parameter_constraints:
                return {'valid': True}  # No constraints defined

            constraints = self.parameter_constraints[parameter_path]
            
            # Type validation
            expected_type = constraints.get('type')
            if expected_type == 'int' and not isinstance(new_value, int):
                return {'valid': False, 'error': f'Expected int, got {type(new_value).__name__}'}
            elif expected_type == 'float' and not isinstance(new_value, (int, float)):
                return {'valid': False, 'error': f'Expected float, got {type(new_value).__name__}'}
            elif expected_type == 'list' and not isinstance(new_value, list):
                return {'valid': False, 'error': f'Expected list, got {type(new_value).__name__}'}
            elif expected_type == 'str' and not isinstance(new_value, str):
                return {'valid': False, 'error': f'Expected str, got {type(new_value).__name__}'}

            # Range validation for numbers
            if expected_type in ['int', 'float']:
                if 'min' in constraints and new_value < constraints['min']:
                    return {'valid': False, 'error': f'Value {new_value} below minimum {constraints["min"]}'}
                if 'max' in constraints and new_value > constraints['max']:
                    return {'valid': False, 'error': f'Value {new_value} above maximum {constraints["max"]}'}

            # List-specific validation
            if expected_type == 'list':
                if 'min_length' in constraints and len(new_value) < constraints['min_length']:
                    return {'valid': False, 'error': f'List length {len(new_value)} below minimum {constraints["min_length"]}'}
                if 'max_length' in constraints and len(new_value) > constraints['max_length']:
                    return {'valid': False, 'error': f'List length {len(new_value)} above maximum {constraints["max_length"]}'}
                
                # Element validation
                if 'element_min' in constraints or 'element_max' in constraints:
                    for i, element in enumerate(new_value):
                        if 'element_min' in constraints and element < constraints['element_min']:
                            return {'valid': False, 'error': f'Element {i} value {element} below minimum {constraints["element_min"]}'}
                        if 'element_max' in constraints and element > constraints['element_max']:
                            return {'valid': False, 'error': f'Element {i} value {element} above maximum {constraints["element_max"]}'}

                # Sum constraint (for weights)
                if 'sum_constraint' in constraints:
                    element_sum = sum(new_value)
                    expected_sum = constraints['sum_constraint']
                    if abs(element_sum - expected_sum) > 0.001:  # Allow small floating point errors
                        return {'valid': False, 'error': f'List sum {element_sum} does not equal required sum {expected_sum}'}

            return {'valid': True}

        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {e}'}

    def _get_parameter_value(self, parameter_path: str) -> Any:
        """Get the current value of a parameter using dot notation"""
        try:
            parts = parameter_path.split('.')
            value = self.runtime_parameters
            
            for part in parts:
                value = value[part]
            
            return value
        except (KeyError, TypeError):
            raise ValueError(f"Parameter path {parameter_path} not found")

    def _set_parameter_value(self, parameter_path: str, new_value: Any):
        """Set a parameter value using dot notation"""
        try:
            parts = parameter_path.split('.')
            target = self.runtime_parameters
            
            # Navigate to the parent of the target parameter
            for part in parts[:-1]:
                target = target[part]
            
            # Set the final value
            target[parts[-1]] = new_value
            
        except (KeyError, TypeError):
            raise ValueError(f"Cannot set parameter path {parameter_path}")

    def get_parameters(self) -> Dict:
        """Get current runtime parameters"""
        return {
            'current_parameters': copy.deepcopy(self.runtime_parameters),
            'parameter_constraints': self.parameter_constraints,
            'modification_history': self.parameter_history[-10:],  # Last 10 modifications
            'adaptive_adjustments': self.adaptive_adjustments,
            'performance_tracking': self.performance_tracking
        }

    def register_modification_callback(self, parameter_path: str, callback: Callable):
        """Register a callback function for parameter modifications"""
        try:
            if parameter_path not in self.modification_callbacks:
                self.modification_callbacks[parameter_path] = []
            
            self.modification_callbacks[parameter_path].append(callback)
            
            logger.info(f"Registered callback for parameter {parameter_path}")
            return True
        except Exception as e:
            logger.error(f"Error registering callback: {e}")
            return False

    def _trigger_modification_callbacks(self, parameter_path: str, old_value: Any, new_value: Any):
        """Trigger callbacks when a parameter is modified"""
        try:
            if parameter_path in self.modification_callbacks:
                for callback in self.modification_callbacks[parameter_path]:
                    try:
                        callback(parameter_path, old_value, new_value)
                    except Exception as e:
                        logger.error(f"Error in modification callback: {e}")
        except Exception as e:
            logger.error(f"Error triggering callbacks: {e}")

    def rollback_modification(self, modification_index: int = -1) -> Dict:
        """Rollback to a previous parameter state"""
        try:
            if not self.parameter_history:
                return {'success': False, 'error': 'No modification history available'}

            # Get the modification to rollback to
            if modification_index < 0:
                modification_index = len(self.parameter_history) + modification_index
            
            if modification_index < 0 or modification_index >= len(self.parameter_history):
                return {'success': False, 'error': 'Invalid modification index'}

            rollback_data = self.parameter_history[modification_index]['rollback_data']
            
            # Store current state for potential re-rollback
            current_state = copy.deepcopy(self.runtime_parameters)
            
            # Restore previous state
            self.runtime_parameters = copy.deepcopy(rollback_data)
            
            # Record the rollback
            rollback_record = {
                'timestamp': datetime.utcnow().isoformat(),
                'action': 'rollback',
                'target_modification_index': modification_index,
                'rollback_data': current_state
            }
            
            self.parameter_history.append(rollback_record)
            
            logger.info(f"Rolled back to modification index {modification_index}")
            
            return {
                'success': True,
                'rollback_timestamp': rollback_record['timestamp'],
                'target_index': modification_index
            }

        except Exception as e:
            logger.error(f"Error in rollback: {e}")
            return {'success': False, 'error': str(e)}

    def auto_optimize_parameters(self, performance_data: Dict) -> Dict:
        """Automatically optimize parameters based on performance data"""
        try:
            optimization_result = {
                'timestamp': datetime.utcnow().isoformat(),
                'optimizations_applied': [],
                'performance_improvement': 0
            }

            current_performance = performance_data.get('current_performance', {})
            
            # Optimize prediction model weights based on individual model performance
            if 'model_accuracies' in current_performance:
                accuracies = current_performance['model_accuracies']
                if len(accuracies) == 4:  # Ensure we have 4 model accuracies
                    # Normalize accuracies to create new weights
                    total_accuracy = sum(accuracies)
                    if total_accuracy > 0:
                        new_weights = [acc / total_accuracy for acc in accuracies]
                        
                        modification = self.modify_parameters({
                            'prediction_models.ensemble_weights': new_weights
                        })
                        
                        if modification['success']:
                            optimization_result['optimizations_applied'].append({
                                'parameter': 'ensemble_weights',
                                'optimization_type': 'accuracy_based',
                                'new_value': new_weights
                            })

            # Optimize collection interval based on data quality vs performance
            if 'data_quality_score' in current_performance and 'system_load' in current_performance:
                quality_score = current_performance['data_quality_score']
                system_load = current_performance['system_load']
                
                current_interval = self._get_parameter_value('data_collection.collection_interval')
                
                # If quality is low and system load is acceptable, decrease interval
                if quality_score < 0.7 and system_load < 0.6:
                    new_interval = max(10, int(current_interval * 0.8))
                    modification = self.modify_parameters({
                        'data_collection.collection_interval': new_interval
                    })
                    
                    if modification['success']:
                        optimization_result['optimizations_applied'].append({
                            'parameter': 'collection_interval',
                            'optimization_type': 'quality_improvement',
                            'new_value': new_interval
                        })
                
                # If system load is high, increase interval
                elif system_load > 0.8:
                    new_interval = min(300, int(current_interval * 1.2))
                    modification = self.modify_parameters({
                        'data_collection.collection_interval': new_interval
                    })
                    
                    if modification['success']:
                        optimization_result['optimizations_applied'].append({
                            'parameter': 'collection_interval',
                            'optimization_type': 'load_reduction',
                            'new_value': new_interval
                        })

            # Optimize pattern sensitivity based on false positive rate
            if 'pattern_false_positive_rate' in current_performance:
                fp_rate = current_performance['pattern_false_positive_rate']
                current_sensitivity = self._get_parameter_value('analysis_settings.pattern_sensitivity')
                
                # If too many false positives, decrease sensitivity
                if fp_rate > 0.3:
                    new_sensitivity = max(0.1, current_sensitivity * 0.9)
                    modification = self.modify_parameters({
                        'analysis_settings.pattern_sensitivity': new_sensitivity
                    })
                    
                    if modification['success']:
                        optimization_result['optimizations_applied'].append({
                            'parameter': 'pattern_sensitivity',
                            'optimization_type': 'false_positive_reduction',
                            'new_value': new_sensitivity
                        })

            # Store optimization history
            self.adaptive_adjustments[datetime.utcnow().isoformat()] = optimization_result

            return optimization_result

        except Exception as e:
            logger.error(f"Error in auto optimization: {e}")
            return {'error': str(e)}

    def _start_adaptive_monitoring(self):
        """Start adaptive parameter monitoring"""
        def adaptive_monitor():
            while self.active:
                try:
                    # Monitor parameter performance and make adjustments
                    self._monitor_parameter_performance()
                    time.sleep(300)  # Check every 5 minutes
                except Exception as e:
                    logger.error(f"Adaptive monitoring error: {e}")
                    time.sleep(600)  # Wait longer on error

        monitor_thread = threading.Thread(target=adaptive_monitor, daemon=True)
        monitor_thread.start()

    def _monitor_parameter_performance(self):
        """Monitor parameter performance and suggest adjustments"""
        try:
            # This would integrate with actual performance metrics
            # For now, we'll simulate performance tracking
            
            current_time = datetime.utcnow().isoformat()
            
            # Simulate performance metrics
            performance_metrics = {
                'prediction_accuracy': 0.75 + (time.time() % 100) / 1000,  # Simulated varying accuracy
                'data_collection_success_rate': 0.9,
                'system_response_time': 150 + (time.time() % 50),
                'error_rate': 0.02
            }
            
            self.performance_tracking[current_time] = performance_metrics
            
            # Keep only last 100 performance records
            if len(self.performance_tracking) > 100:
                oldest_keys = sorted(self.performance_tracking.keys())[:-100]
                for key in oldest_keys:
                    del self.performance_tracking[key]

        except Exception as e:
            logger.error(f"Error monitoring parameter performance: {e}")

    def export_configuration(self) -> Dict:
        """Export current configuration for backup/sharing"""
        return {
            'export_timestamp': datetime.utcnow().isoformat(),
            'runtime_parameters': copy.deepcopy(self.runtime_parameters),
            'parameter_constraints': self.parameter_constraints,
            'modification_history': self.parameter_history,
            'adaptive_adjustments': self.adaptive_adjustments,
            'version': '1.0'
        }

    def import_configuration(self, config_data: Dict) -> Dict:
        """Import configuration from exported data"""
        try:
            if 'runtime_parameters' not in config_data:
                return {'success': False, 'error': 'Invalid configuration data'}

            # Backup current state
            backup = self.export_configuration()
            
            # Import new configuration
            self.runtime_parameters = copy.deepcopy(config_data['runtime_parameters'])
            
            if 'parameter_constraints' in config_data:
                self.parameter_constraints.update(config_data['parameter_constraints'])

            # Record the import
            import_record = {
                'timestamp': datetime.utcnow().isoformat(),
                'action': 'configuration_import',
                'backup_data': backup,
                'imported_version': config_data.get('version', 'unknown')
            }
            
            self.parameter_history.append(import_record)
            
            logger.info("Configuration imported successfully")
            
            return {
                'success': True,
                'import_timestamp': import_record['timestamp'],
                'imported_version': config_data.get('version', 'unknown')
            }

        except Exception as e:
            logger.error(f"Error importing configuration: {e}")
            return {'success': False, 'error': str(e)}

    def reset_to_defaults(self) -> Dict:
        """Reset all parameters to default values"""
        try:
            # Backup current state
            backup = copy.deepcopy(self.runtime_parameters)
            
            # Reset to defaults (reinitialize)
            self.__init__()
            self.initialize()
            
            # Record the reset
            reset_record = {
                'timestamp': datetime.utcnow().isoformat(),
                'action': 'reset_to_defaults',
                'backup_data': backup
            }
            
            self.parameter_history.append(reset_record)
            
            logger.info("Parameters reset to defaults")
            
            return {
                'success': True,
                'reset_timestamp': reset_record['timestamp']
            }

        except Exception as e:
            logger.error(f"Error resetting to defaults: {e}")
            return {'success': False, 'error': str(e)}
