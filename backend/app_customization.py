"""
Feature 10: App Customization
Configurable prediction parameters and user interface customization
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
import logging
import copy
from collections import defaultdict

logger = logging.getLogger(__name__)

class CustomizationManager:
    def __init__(self):
        self.active = False
        self.settings = {
            'ui_settings': {
                'theme': 'dark',
                'language': 'en',
                'layout': 'dashboard',
                'chart_type': 'candlestick',
                'refresh_rate': 1000,  # milliseconds
                'show_notifications': True,
                'animation_enabled': True,
                'compact_mode': False
            },
            'prediction_settings': {
                'default_model': 'ensemble',
                'confidence_threshold': 0.7,
                'prediction_horizon': 1,  # Next N predictions
                'auto_retrain': True,
                'retrain_frequency': 24,  # hours
                'max_history_length': 1000,
                'feature_selection': 'auto'
            },
            'alert_settings': {
                'enable_alerts': True,
                'high_multiplier_threshold': 5.0,
                'low_multiplier_threshold': 1.2,
                'pattern_detection_alerts': True,
                'prediction_confidence_alerts': True,
                'sound_enabled': False,
                'email_notifications': False
            },
            'data_settings': {
                'data_sources': ['spribe', 'evolution'],
                'collection_frequency': 'realtime',
                'backup_enabled': True,
                'data_retention_days': 30,
                'export_format': 'json',
                'compression_enabled': True
            },
            'analysis_settings': {
                'technical_indicators': {
                    'moving_averages': [5, 10, 20],
                    'bollinger_bands': True,
                    'rsi_period': 14,
                    'macd_enabled': True
                },
                'pattern_recognition': {
                    'min_pattern_length': 3,
                    'max_pattern_length': 10,
                    'similarity_threshold': 0.8
                },
                'trend_analysis': {
                    'short_term_window': 10,
                    'long_term_window': 50,
                    'trend_strength_threshold': 0.6
                }
            },
            'display_settings': {
                'decimal_places': 2,
                'currency_symbol': '$',
                'time_format': '24h',
                'date_format': 'YYYY-MM-DD',
                'number_format': 'comma',
                'chart_colors': {
                    'bullish': '#00ff88',
                    'bearish': '#ff4444',
                    'neutral': '#ffaa00',
                    'background': '#1a1a1a',
                    'text': '#ffffff'
                }
            }
        }
        self.themes = {
            'dark': {
                'background': '#1a1a1a',
                'surface': '#2d2d2d',
                'primary': '#007bff',
                'secondary': '#6c757d',
                'success': '#28a745',
                'warning': '#ffc107',
                'danger': '#dc3545',
                'text': '#ffffff',
                'text_secondary': '#adb5bd'
            },
            'light': {
                'background': '#ffffff',
                'surface': '#f8f9fa',
                'primary': '#007bff',
                'secondary': '#6c757d',
                'success': '#28a745',
                'warning': '#ffc107',
                'danger': '#dc3545',
                'text': '#212529',
                'text_secondary': '#6c757d'
            },
            'blue': {
                'background': '#0d1421',
                'surface': '#1e2329',
                'primary': '#3461ff',
                'secondary': '#7c8db0',
                'success': '#02c076',
                'warning': '#ffa726',
                'danger': '#f44336',
                'text': '#ffffff',
                'text_secondary': '#b7bdc8'
            },
            'green': {
                'background': '#0d1b0d',
                'surface': '#1a2e1a',
                'primary': '#4caf50',
                'secondary': '#81c784',
                'success': '#66bb6a',
                'warning': '#ffb74d',
                'danger': '#e57373',
                'text': '#ffffff',
                'text_secondary': '#a5d6a7'
            }
        }
        self.layouts = {
            'dashboard': {
                'components': [
                    {'type': 'chart', 'position': {'x': 0, 'y': 0, 'w': 8, 'h': 6}},
                    {'type': 'predictions', 'position': {'x': 8, 'y': 0, 'w': 4, 'h': 6}},
                    {'type': 'history', 'position': {'x': 0, 'y': 6, 'w': 6, 'h': 4}},
                    {'type': 'analytics', 'position': {'x': 6, 'y': 6, 'w': 6, 'h': 4}}
                ]
            },
            'minimal': {
                'components': [
                    {'type': 'chart', 'position': {'x': 0, 'y': 0, 'w': 12, 'h': 8}},
                    {'type': 'predictions', 'position': {'x': 0, 'y': 8, 'w': 12, 'h': 2}}
                ]
            },
            'advanced': {
                'components': [
                    {'type': 'chart', 'position': {'x': 0, 'y': 0, 'w': 6, 'h': 6}},
                    {'type': 'predictions', 'position': {'x': 6, 'y': 0, 'w': 3, 'h': 6}},
                    {'type': 'controls', 'position': {'x': 9, 'y': 0, 'w': 3, 'h': 6}},
                    {'type': 'history', 'position': {'x': 0, 'y': 6, 'w': 4, 'h': 4}},
                    {'type': 'analytics', 'position': {'x': 4, 'y': 6, 'w': 4, 'h': 4}},
                    {'type': 'patterns', 'position': {'x': 8, 'y': 6, 'w': 4, 'h': 4}}
                ]
            }
        }
        self.user_profiles = {}
        self.customization_history = []
        
    def initialize(self):
        """Initialize the customization manager"""
        self.active = True
        self._load_settings_from_file()
        logger.info("Customization Manager initialized")
        
    def is_active(self):
        return self.active
        
    def get_settings(self) -> Dict:
        """Get current settings"""
        return {
            'settings': copy.deepcopy(self.settings),
            'available_themes': list(self.themes.keys()),
            'available_layouts': list(self.layouts.keys()),
            'last_modified': datetime.utcnow().isoformat()
        }
        
    def update_settings(self, new_settings: Dict) -> Dict:
        """Update settings with validation"""
        try:
            update_result = {
                'timestamp': datetime.utcnow().isoformat(),
                'updated_settings': [],
                'validation_errors': [],
                'previous_settings': copy.deepcopy(self.settings)
            }
            
            for category, settings in new_settings.items():
                if category not in self.settings:
                    update_result['validation_errors'].append({
                        'category': category,
                        'error': 'Unknown settings category'
                    })
                    continue
                    
                for setting_name, setting_value in settings.items():
                    validation_result = self._validate_setting(category, setting_name, setting_value)
                    
                    if validation_result['valid']:
                        old_value = self.settings[category].get(setting_name)
                        self.settings[category][setting_name] = setting_value
                        
                        update_result['updated_settings'].append({
                            'category': category,
                            'setting': setting_name,
                            'old_value': old_value,
                            'new_value': setting_value
                        })
                        
                        # Apply setting-specific logic
                        self._apply_setting_change(category, setting_name, setting_value)
                        
                    else:
                        update_result['validation_errors'].append({
                            'category': category,
                            'setting': setting_name,
                            'error': validation_result['error'],
                            'attempted_value': setting_value
                        })
            
            # Save settings to file
            if update_result['updated_settings']:
                self._save_settings_to_file()
                
            # Store in history
            self.customization_history.append(update_result)
            
            # Keep only last 100 changes
            if len(self.customization_history) > 100:
                self.customization_history = self.customization_history[-100:]
                
            return {
                'success': len(update_result['updated_settings']) > 0,
                'updated_count': len(update_result['updated_settings']),
                'error_count': len(update_result['validation_errors']),
                'details': update_result
            }
            
        except Exception as e:
            logger.error(f"Error updating settings: {e}")
            return {'success': False, 'error': str(e)}
            
    def _validate_setting(self, category: str, setting_name: str, setting_value: Any) -> Dict:
        """Validate a setting value"""
        try:
            # Theme validation
            if category == 'ui_settings' and setting_name == 'theme':
                if setting_value not in self.themes:
                    return {'valid': False, 'error': f'Unknown theme: {setting_value}'}
                    
            # Layout validation
            elif category == 'ui_settings' and setting_name == 'layout':
                if setting_value not in self.layouts:
                    return {'valid': False, 'error': f'Unknown layout: {setting_value}'}
                    
            # Refresh rate validation
            elif category == 'ui_settings' and setting_name == 'refresh_rate':
                if not isinstance(setting_value, int) or setting_value < 100 or setting_value > 10000:
                    return {'valid': False, 'error': 'Refresh rate must be between 100 and 10000 milliseconds'}
                    
            # Confidence threshold validation
            elif category == 'prediction_settings' and setting_name == 'confidence_threshold':
                if not isinstance(setting_value, (int, float)) or setting_value < 0 or setting_value > 1:
                    return {'valid': False, 'error': 'Confidence threshold must be between 0 and 1'}
                    
            # Multiplier threshold validation
            elif setting_name.endswith('_threshold') and 'multiplier' in setting_name:
                if not isinstance(setting_value, (int, float)) or setting_value < 1.0:
                    return {'valid': False, 'error': 'Multiplier threshold must be >= 1.0'}
                    
            # Data retention validation
            elif category == 'data_settings' and setting_name == 'data_retention_days':
                if not isinstance(setting_value, int) or setting_value < 1 or setting_value > 365:
                    return {'valid': False, 'error': 'Data retention must be between 1 and 365 days'}
                    
            # Moving averages validation
            elif setting_name == 'moving_averages':
                if not isinstance(setting_value, list) or not all(isinstance(x, int) and x > 0 for x in setting_value):
                    return {'valid': False, 'error': 'Moving averages must be a list of positive integers'}
                    
            return {'valid': True}
            
        except Exception as e:
            return {'valid': False, 'error': f'Validation error: {e}'}
            
    def _apply_setting_change(self, category: str, setting_name: str, setting_value: Any):
        """Apply specific logic when a setting changes"""
        try:
            # Theme change
            if category == 'ui_settings' and setting_name == 'theme':
                theme_colors = self.themes[setting_value]
                self.settings['display_settings']['chart_colors'].update({
                    'background': theme_colors['background'],
                    'text': theme_colors['text']
                })
                
            # Language change
            elif category == 'ui_settings' and setting_name == 'language':
                # Would trigger language pack loading
                logger.info(f"Language changed to: {setting_value}")
                
            # Model change
            elif category == 'prediction_settings' and setting_name == 'default_model':
                # Would update prediction engine default
                logger.info(f"Default model changed to: {setting_value}")
                
        except Exception as e:
            logger.error(f"Error applying setting change: {e}")
            
    def change_theme(self, theme_name: str) -> Dict:
        """Change application theme"""
        try:
            if theme_name not in self.themes:
                return {'success': False, 'error': f'Unknown theme: {theme_name}'}
                
            old_theme = self.settings['ui_settings']['theme']
            self.settings['ui_settings']['theme'] = theme_name
            
            # Update display colors
            theme_colors = self.themes[theme_name]
            self.settings['display_settings']['chart_colors'].update({
                'background': theme_colors['background'],
                'text': theme_colors['text']
            })
            
            self._save_settings_to_file()
            
            return {
                'success': True,
                'old_theme': old_theme,
                'new_theme': theme_name,
                'theme_colors': theme_colors,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error changing theme: {e}")
            return {'success': False, 'error': str(e)}
            
    def get_theme_info(self, theme_name: str = None) -> Dict:
        """Get theme information"""
        if theme_name:
            if theme_name in self.themes:
                return {
                    'theme_name': theme_name,
                    'colors': self.themes[theme_name],
                    'is_current': self.settings['ui_settings']['theme'] == theme_name
                }
            else:
                return {'error': f'Theme {theme_name} not found'}
        else:
            return {
                'available_themes': self.themes,
                'current_theme': self.settings['ui_settings']['theme']
            }
            
    def create_custom_theme(self, theme_name: str, colors: Dict) -> Dict:
        """Create a custom theme"""
        try:
            required_colors = ['background', 'surface', 'primary', 'secondary', 
                             'success', 'warning', 'danger', 'text', 'text_secondary']
            
            # Validate colors
            for color_key in required_colors:
                if color_key not in colors:
                    return {'success': False, 'error': f'Missing required color: {color_key}'}
                    
                # Validate hex color format
                color_value = colors[color_key]
                if not isinstance(color_value, str) or not color_value.startswith('#'):
                    return {'success': False, 'error': f'Invalid color format for {color_key}: {color_value}'}
                    
            self.themes[theme_name] = colors
            
            return {
                'success': True,
                'theme_name': theme_name,
                'created_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating custom theme: {e}")
            return {'success': False, 'error': str(e)}
            
    def change_layout(self, layout_name: str) -> Dict:
        """Change application layout"""
        try:
            if layout_name not in self.layouts:
                return {'success': False, 'error': f'Unknown layout: {layout_name}'}
                
            old_layout = self.settings['ui_settings']['layout']
            self.settings['ui_settings']['layout'] = layout_name
            
            self._save_settings_to_file()
            
            return {
                'success': True,
                'old_layout': old_layout,
                'new_layout': layout_name,
                'layout_config': self.layouts[layout_name],
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error changing layout: {e}")
            return {'success': False, 'error': str(e)}
            
    def get_layout_info(self, layout_name: str = None) -> Dict:
        """Get layout information"""
        if layout_name:
            if layout_name in self.layouts:
                return {
                    'layout_name': layout_name,
                    'config': self.layouts[layout_name],
                    'is_current': self.settings['ui_settings']['layout'] == layout_name
                }
            else:
                return {'error': f'Layout {layout_name} not found'}
        else:
            return {
                'available_layouts': self.layouts,
                'current_layout': self.settings['ui_settings']['layout']
            }
            
    def create_custom_layout(self, layout_name: str, components: List[Dict]) -> Dict:
        """Create a custom layout"""
        try:
            # Validate components
            required_fields = ['type', 'position']
            valid_types = ['chart', 'predictions', 'history', 'analytics', 'controls', 'patterns']
            
            for component in components:
                for field in required_fields:
                    if field not in component:
                        return {'success': False, 'error': f'Missing required field: {field}'}
                        
                if component['type'] not in valid_types:
                    return {'success': False, 'error': f'Invalid component type: {component["type"]}'}
                    
                position = component['position']
                required_pos_fields = ['x', 'y', 'w', 'h']
                for pos_field in required_pos_fields:
                    if pos_field not in position:
                        return {'success': False, 'error': f'Missing position field: {pos_field}'}
                        
            self.layouts[layout_name] = {'components': components}
            
            return {
                'success': True,
                'layout_name': layout_name,
                'component_count': len(components),
                'created_at': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error creating custom layout: {e}")
            return {'success': False, 'error': str(e)}
            
    def create_user_profile(self, profile_name: str, settings: Dict = None) -> Dict:
        """Create a user profile with specific settings"""
        try:
            if not settings:
                settings = copy.deepcopy(self.settings)
                
            profile = {
                'name': profile_name,
                'settings': settings,
                'created_at': datetime.utcnow().isoformat(),
                'last_used': None
            }
            
            self.user_profiles[profile_name] = profile
            
            return {
                'success': True,
                'profile_name': profile_name,
                'created_at': profile['created_at']
            }
            
        except Exception as e:
            logger.error(f"Error creating user profile: {e}")
            return {'success': False, 'error': str(e)}
            
    def load_user_profile(self, profile_name: str) -> Dict:
        """Load a user profile"""
        try:
            if profile_name not in self.user_profiles:
                return {'success': False, 'error': f'Profile {profile_name} not found'}
                
            profile = self.user_profiles[profile_name]
            
            # Backup current settings
            backup = {
                'timestamp': datetime.utcnow().isoformat(),
                'settings': copy.deepcopy(self.settings)
            }
            
            # Load profile settings
            self.settings = copy.deepcopy(profile['settings'])
            
            # Update profile last used
            profile['last_used'] = datetime.utcnow().isoformat()
            
            self._save_settings_to_file()
            
            return {
                'success': True,
                'profile_name': profile_name,
                'loaded_at': datetime.utcnow().isoformat(),
                'backup': backup
            }
            
        except Exception as e:
            logger.error(f"Error loading user profile: {e}")
            return {'success': False, 'error': str(e)}
            
    def get_user_profiles(self) -> Dict:
        """Get all user profiles"""
        profiles_info = {}
        
        for name, profile in self.user_profiles.items():
            profiles_info[name] = {
                'name': profile['name'],
                'created_at': profile['created_at'],
                'last_used': profile['last_used']
            }
            
        return {
            'user_profiles': profiles_info,
            'total_profiles': len(self.user_profiles)
        }
        
    def _save_settings_to_file(self):
        """Save settings to file"""
        try:
            settings_file = 'settings.json'
            with open(settings_file, 'w') as f:
                json.dump({
                    'settings': self.settings,
                    'themes': self.themes,
                    'layouts': self.layouts,
                    'user_profiles': self.user_profiles,
                    'last_saved': datetime.utcnow().isoformat()
                }, f, indent=2)
        except Exception as e:
            logger.error(f"Error saving settings to file: {e}")
            
    def _load_settings_from_file(self):
        """Load settings from file"""
        try:
            settings_file = 'settings.json'
            if os.path.exists(settings_file):
                with open(settings_file, 'r') as f:
                    data = json.load(f)
                    
                self.settings.update(data.get('settings', {}))
                self.themes.update(data.get('themes', {}))
                self.layouts.update(data.get('layouts', {}))
                self.user_profiles.update(data.get('user_profiles', {}))
                
                logger.info("Settings loaded from file")
        except Exception as e:
            logger.error(f"Error loading settings from file: {e}")
            
    def export_settings(self) -> Dict:
        """Export all settings for backup"""
        return {
            'export_timestamp': datetime.utcnow().isoformat(),
            'settings': self.settings,
            'themes': self.themes,
            'layouts': self.layouts,
            'user_profiles': self.user_profiles,
            'customization_history': self.customization_history[-50:],  # Last 50 changes
            'version': '1.0'
        }
        
    def import_settings(self, settings_data: Dict) -> Dict:
        """Import settings from backup"""
        try:
            # Backup current settings
            backup = self.export_settings()
            
            # Import new settings
            if 'settings' in settings_data:
                self.settings.update(settings_data['settings'])
                
            if 'themes' in settings_data:
                self.themes.update(settings_data['themes'])
                
            if 'layouts' in settings_data:
                self.layouts.update(settings_data['layouts'])
                
            if 'user_profiles' in settings_data:
                self.user_profiles.update(settings_data['user_profiles'])
                
            self._save_settings_to_file()
            
            return {
                'success': True,
                'imported_at': datetime.utcnow().isoformat(),
                'backup': backup
            }
            
        except Exception as e:
            logger.error(f"Error importing settings: {e}")
            return {'success': False, 'error': str(e)}
            
    def reset_to_defaults(self) -> Dict:
        """Reset all settings to defaults"""
        try:
            # Backup current settings
            backup = self.export_settings()
            
            # Reset to defaults
            self.__init__()
            self.initialize()
            
            return {
                'success': True,
                'reset_at': datetime.utcnow().isoformat(),
                'backup': backup
            }
            
        except Exception as e:
            logger.error(f"Error resetting to defaults: {e}")
            return {'success': False, 'error': str(e)}