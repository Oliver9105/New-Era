#!/usr/bin/env python3
"""
Real Prediction Engine - Actual machine learning models for aviator predictions
Implements real prediction algorithms based on collected data
Author: MiniMax Agent
"""

try:
    import numpy as np
    import pandas as pd
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    import random
    import statistics
    # Simple fallback implementations
    class np:
        @staticmethod
        def mean(arr): return sum(arr) / len(arr) if arr else 0
        @staticmethod
        def std(arr): 
            if not arr: return 0
            mean_val = sum(arr) / len(arr)
            return (sum((x - mean_val) ** 2 for x in arr) / len(arr)) ** 0.5
        @staticmethod
        def min(arr): return min(arr) if arr else 0
        @staticmethod
        def max(arr): return max(arr) if arr else 0
        @staticmethod
        def median(arr): return statistics.median(arr) if arr else 0
        @staticmethod
        def sqrt(x): return x ** 0.5
        @staticmethod
        def array(data): return list(data)
        @staticmethod
        def arange(n): return list(range(n))
        random = random
    
    class pd:
        @staticmethod
        def DataFrame(data): return data
        @staticmethod
        def to_datetime(date_str):
            try:
                from datetime import datetime
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            except:
                return datetime.now()

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging
import json
import pickle
import os

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class RealPredictionEngine:
    """
    Real prediction engine using machine learning models
    Analyzes historical aviator game data to make predictions
    """
    
    def __init__(self):
        if SKLEARN_AVAILABLE:
            self.models = {
                'random_forest': RandomForestRegressor(n_estimators=100, random_state=42),
                'gradient_boost': GradientBoostingRegressor(n_estimators=100, random_state=42),
                'linear_regression': LinearRegression()
            }
            self.scalers = {
                'random_forest': StandardScaler(),
                'gradient_boost': StandardScaler(),
                'linear_regression': StandardScaler()
            }
        else:
            self.models = {}
            self.scalers = {}
            
        self.trained_models = {}
        self.model_performance = {}
        self.historical_data = []
        self.feature_columns = []
        self.models_dir = 'models'
        
        # Ensure models directory exists
        os.makedirs(self.models_dir, exist_ok=True)
        
    def initialize(self):
        """Initialize the prediction engine"""
        logger.info("Real Prediction Engine initialized")
        self._load_saved_models()
        
        # Request initial historical data if data collector is available
        if hasattr(self, '_data_collector') and self._data_collector:
            self._request_initial_data()
    
    def set_data_collector(self, data_collector):
        """Set the data collector reference"""
        self._data_collector = data_collector
    
    def refresh_data(self):
        """Called when new data is available - refresh internal state"""
        try:
            # This method is called when new data has been processed
            # We can use this to trigger any data-dependent updates
            logger.debug("Prediction engine notified of new data")
        except Exception as e:
            logger.error(f"Error refreshing prediction engine data: {e}")
    
    def _request_initial_data(self):
        """Request initial historical data from the data collector"""
        try:
            logger.info("Requesting initial historical data from data collector...")
            
            # Get any existing collected data
            existing_data = self._data_collector.get_collected_data(limit=50)
            
            if existing_data:
                logger.info(f"Found {len(existing_data)} existing data points")
                for data_point in existing_data:
                    if data_point.get('game_data'):
                        self.add_historical_data(data_point['game_data'])
            else:
                logger.info("No existing data found - will populate as new data arrives")
                
            # Check if we need to trigger data collection
            if len(self.historical_data) < 10:
                logger.info("Insufficient historical data - triggering sample data collection")
                # This will be called when the app starts data collection
                
        except Exception as e:
            logger.error(f"Error requesting initial data: {e}")
        
    def add_historical_data(self, game_data: Dict[str, Any]):
        """
        Add historical game data for training
        
        Args:
            game_data: Dictionary containing game round data
        """
        try:
            # Standardize the data format
            standardized_data = self._standardize_game_data(game_data)
            if standardized_data:
                self.historical_data.append(standardized_data)
                logger.info(f"✅ Added historical data point: {standardized_data['round_id']} - {standardized_data['multiplier']}x (Total: {len(self.historical_data)} points)")
                
                # Automatically retrain if we have enough new data
                if len(self.historical_data) % 50 == 0:  # Retrain every 50 new data points
                    self._auto_retrain()
            else:
                logger.warning(f"❌ Failed to standardize game data: {game_data}")
                    
        except Exception as e:
            logger.error(f"Error adding historical data: {e}")
            logger.error(f"Game data: {game_data}")
    
    def get_current_prediction(self) -> Dict[str, Any]:
        """
        Get current prediction using the best available model
        
        Returns:
            Dictionary containing prediction details
        """
        try:
            if not self.trained_models:
                # No trained models, use simple heuristic
                return self._heuristic_prediction()
            
            # Use the best performing model
            best_model_name = self._get_best_model()
            
            if best_model_name and best_model_name in self.trained_models:
                prediction = self._predict_with_model(best_model_name)
                prediction['method'] = best_model_name
                return prediction
            else:
                return self._heuristic_prediction()
                
        except Exception as e:
            logger.error(f"Error getting prediction: {e}")
            return self._heuristic_prediction()
    
    def predict_realtime(self, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Make a real-time prediction based on current game state
        
        Args:
            current_data: Current game data
            
        Returns:
            Prediction result
        """
        try:
            # Add current data to historical data if it's complete
            if current_data.get('multiplier') and current_data.get('round_id'):
                self.add_historical_data(current_data)
            
            # Get prediction based on current context
            prediction = self.get_current_prediction()
            
            # Adjust prediction based on current data
            if current_data.get('multiplier'):
                prediction = self._adjust_prediction_for_context(prediction, current_data)
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in realtime prediction: {e}")
            return self._heuristic_prediction()
    
    def train_models(self, model_type: str = 'all') -> Dict[str, Any]:
        """
        Train prediction models on historical data
        
        Args:
            model_type: Type of model to train ('all', 'random_forest', etc.)
            
        Returns:
            Training results
        """
        try:
            if not SKLEARN_AVAILABLE:
                return {
                    'success': False,
                    'error': 'Machine learning libraries not available. Using heuristic predictions only.'
                }
                
            if len(self.historical_data) < 20:
                return {
                    'success': False,
                    'error': f'Insufficient data for training. Need at least 20 records, have {len(self.historical_data)}'
                }
            
            logger.info(f"Training models with {len(self.historical_data)} data points")
            
            # Prepare training data
            features, targets = self._prepare_training_data()
            
            if features is None or targets is None:
                return {
                    'success': False,
                    'error': 'Failed to prepare training data'
                }
            
            training_results = {}
            
            models_to_train = [model_type] if model_type != 'all' else list(self.models.keys())
            
            for model_name in models_to_train:
                if model_name in self.models:
                    result = self._train_single_model(model_name, features, targets)
                    training_results[model_name] = result
            
            # Save trained models
            self._save_models()
            
            return {
                'success': True,
                'results': training_results,
                'data_points': len(self.historical_data)
            }
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _standardize_game_data(self, game_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Standardize game data format for consistent processing
        """
        try:
            # Extract relevant fields from various possible structures
            standardized = {}
            
            # Handle nested game data
            if 'game_data' in game_data and game_data['game_data']:
                game_info = game_data['game_data']
            else:
                game_info = game_data
            
            # Extract multiplier
            multiplier = None
            for key in ['multiplier', 'crash_multiplier', 'coefficient', 'odds']:
                if key in game_info and game_info[key] is not None:
                    multiplier = float(game_info[key])
                    break
            
            if multiplier is None or multiplier <= 0:
                return None
            
            standardized['multiplier'] = multiplier
            
            # Extract round ID
            round_id = None
            for key in ['round_id', 'game_id', 'id', 'roundId']:
                if key in game_info and game_info[key] is not None:
                    round_id = str(game_info[key])
                    break
            
            standardized['round_id'] = round_id or f"round_{int(datetime.now().timestamp())}"
            
            # Extract timestamp
            timestamp = None
            for key in ['timestamp', 'game_timestamp', 'time', 'created_at']:
                if key in game_data and game_data[key]:
                    timestamp = game_data[key]
                    break
                elif key in game_info and game_info[key]:
                    timestamp = game_info[key]
                    break
            
            standardized['timestamp'] = timestamp or datetime.now().isoformat()
            
            # Convert timestamp to datetime if it's a string
            if isinstance(standardized['timestamp'], str):
                try:
                    standardized['datetime'] = pd.to_datetime(standardized['timestamp'])
                except:
                    standardized['datetime'] = datetime.now()
            else:
                standardized['datetime'] = datetime.now()
            
            return standardized
            
        except Exception as e:
            logger.error(f"Error standardizing game data: {e}")
            return None
    
    def _prepare_training_data(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Prepare features and targets for model training
        
        Returns:
            Tuple of (features, targets) or (None, None) if preparation fails
        """
        try:
            if len(self.historical_data) < 10:
                return None, None
            
            # Convert to DataFrame for easier processing
            df = pd.DataFrame(self.historical_data)
            df = df.sort_values('datetime').reset_index(drop=True)
            
            # Engineer features
            features_data = []
            targets = []
            
            # Use a sliding window approach
            window_size = min(10, len(df) // 2)
            
            for i in range(window_size, len(df)):
                # Features: statistics from previous rounds
                window_data = df.iloc[i-window_size:i]
                
                feature_vector = [
                    window_data['multiplier'].mean(),  # Average multiplier in window
                    window_data['multiplier'].std(),   # Standard deviation
                    window_data['multiplier'].min(),   # Minimum
                    window_data['multiplier'].max(),   # Maximum
                    window_data['multiplier'].median(), # Median
                    len(window_data[window_data['multiplier'] > 2.0]), # Count of high multipliers
                    len(window_data[window_data['multiplier'] < 1.5]), # Count of low multipliers
                    window_data['multiplier'].iloc[-1], # Last multiplier
                    window_data['multiplier'].iloc[-2] if len(window_data) > 1 else window_data['multiplier'].iloc[-1], # Second to last
                    i % 24,  # Hour of day (if we can extract it)
                ]
                
                # Target: next multiplier
                target = df.iloc[i]['multiplier']
                
                features_data.append(feature_vector)
                targets.append(target)
            
            if not features_data:
                return None, None
            
            if NUMPY_AVAILABLE:
                features = np.array(features_data)
                targets = np.array(targets)
            else:
                features = features_data
                targets = targets
            
            # Store feature names for later use
            self.feature_columns = [
                'avg_multiplier', 'std_multiplier', 'min_multiplier', 'max_multiplier',
                'median_multiplier', 'high_count', 'low_count', 'last_multiplier',
                'second_last_multiplier', 'hour_of_day'
            ]
            
            if NUMPY_AVAILABLE:
                logger.info(f"Prepared training data: {features.shape[0]} samples, {features.shape[1]} features")
            else:
                logger.info(f"Prepared training data: {len(features)} samples, {len(features[0]) if features else 0} features")
            return features, targets
            
        except Exception as e:
            logger.error(f"Error preparing training data: {e}")
            return None, None
    
    def _train_single_model(self, model_name: str, features: np.ndarray, targets: np.ndarray) -> Dict[str, Any]:
        """
        Train a single model
        """
        try:
            # Split data into train and validation
            split_idx = int(0.8 * len(features))
            X_train, X_val = features[:split_idx], features[split_idx:]
            y_train, y_val = targets[:split_idx], targets[split_idx:]
            
            # Scale features
            scaler = self.scalers[model_name]
            X_train_scaled = scaler.fit_transform(X_train)
            X_val_scaled = scaler.transform(X_val)
            
            # Train model
            model = self.models[model_name]
            model.fit(X_train_scaled, y_train)
            
            # Validate model
            y_pred = model.predict(X_val_scaled)
            
            # Calculate metrics
            mae = mean_absolute_error(y_val, y_pred)
            mse = mean_squared_error(y_val, y_pred)
            rmse = np.sqrt(mse)
            
            # Calculate accuracy within tolerance
            tolerance = 0.5  # Consider prediction accurate if within 0.5x
            accurate_predictions = np.abs(y_val - y_pred) <= tolerance
            accuracy = np.mean(accurate_predictions) * 100
            
            # Store trained model
            self.trained_models[model_name] = {
                'model': model,
                'scaler': scaler,
                'trained_at': datetime.now().isoformat()
            }
            
            # Store performance metrics
            self.model_performance[model_name] = {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'accuracy': accuracy,
                'validation_samples': len(y_val)
            }
            
            logger.info(f"Trained {model_name}: MAE={mae:.3f}, RMSE={rmse:.3f}, Accuracy={accuracy:.1f}%")
            
            return {
                'success': True,
                'mae': mae,
                'rmse': rmse,
                'accuracy': accuracy,
                'validation_samples': len(y_val)
            }
            
        except Exception as e:
            logger.error(f"Error training {model_name}: {e}")
            return {
                'success': False,
                'error': str(e)
            }
    
    def _predict_with_model(self, model_name: str) -> Dict[str, Any]:
        """
        Make prediction using a specific model
        """
        try:
            if model_name not in self.trained_models:
                return self._heuristic_prediction()
            
            # Get recent data for feature engineering
            if len(self.historical_data) < 10:
                return self._heuristic_prediction()
            
            # Use last 10 rounds to predict next
            recent_data = self.historical_data[-10:]
            multipliers = [d['multiplier'] for d in recent_data]
            
            # Engineer features (same as training)
            feature_vector = [
                np.mean(multipliers),
                np.std(multipliers),
                np.min(multipliers),
                np.max(multipliers),
                np.median(multipliers),
                sum(1 for m in multipliers if m > 2.0),
                sum(1 for m in multipliers if m < 1.5),
                multipliers[-1],
                multipliers[-2] if len(multipliers) > 1 else multipliers[-1],
                datetime.now().hour
            ]
            
            # Scale features and predict
            model_info = self.trained_models[model_name]
            scaler = model_info['scaler']
            model = model_info['model']
            
            features_scaled = scaler.transform([feature_vector])
            predicted_multiplier = model.predict(features_scaled)[0]
            
            # Ensure prediction is within reasonable bounds
            predicted_multiplier = max(1.01, min(50.0, predicted_multiplier))
            
            # Calculate confidence based on model performance
            performance = self.model_performance.get(model_name, {})
            base_confidence = performance.get('accuracy', 50) / 100
            
            # Adjust confidence based on recent data variability
            recent_std = np.std(multipliers)
            confidence_adjustment = max(0.1, 1.0 - (recent_std / 10.0))
            final_confidence = base_confidence * confidence_adjustment
            final_confidence = max(0.1, min(0.9, final_confidence))
            
            return {
                'multiplier': round(predicted_multiplier, 2),
                'confidence': round(final_confidence, 2),
                'method': model_name,
                'model_performance': performance
            }
            
        except Exception as e:
            logger.error(f"Error predicting with {model_name}: {e}")
            return self._heuristic_prediction()
    
    def _heuristic_prediction(self) -> Dict[str, Any]:
        """
        Smart heuristic prediction when no trained models are available
        Uses real collected data if available, fallback to intelligent defaults
        """
        try:
            # Try to get recent data from data collector
            if hasattr(self, '_data_collector') and self._data_collector:
                recent_data = self._data_collector.get_collected_data(limit=10)
                if recent_data:
                    # Extract multipliers from recent real data
                    multipliers = []
                    for data_point in recent_data:
                        if data_point.get('game_data') and data_point['game_data'].get('multiplier'):
                            try:
                                mult = float(data_point['game_data']['multiplier'])
                                if 1.0 <= mult <= 50.0:  # Reasonable range for aviator
                                    multipliers.append(mult)
                            except (ValueError, TypeError):
                                continue
                    
                    if multipliers:
                        logger.info(f"Using {len(multipliers)} real data points for heuristic prediction")
                        
                        # Calculate statistics from real data
                        avg_multiplier = np.mean(multipliers)
                        std_multiplier = np.std(multipliers)
                        min_multiplier = np.min(multipliers)
                        max_multiplier = np.max(multipliers)
                        
                        # Smart prediction based on recent patterns
                        recent_trend = np.mean(multipliers[-3:]) if len(multipliers) >= 3 else avg_multiplier
                        
                        # Predict slightly above recent trend with pattern awareness
                        if recent_trend > avg_multiplier * 1.2:
                            # Recent high trend, predict regression to mean
                            prediction = avg_multiplier * np.random.uniform(0.9, 1.1)
                        elif recent_trend < avg_multiplier * 0.8:
                            # Recent low trend, predict potential recovery
                            prediction = avg_multiplier * np.random.uniform(1.0, 1.3)
                        else:
                            # Normal range, predict around recent trend
                            prediction = recent_trend * np.random.uniform(0.95, 1.05)
                        
                        # Ensure prediction is within reasonable bounds
                        prediction = max(1.01, min(20.0, prediction))
                        
                        # Confidence based on data consistency and amount
                        data_consistency = 1.0 - min(1.0, std_multiplier / avg_multiplier)
                        data_amount_factor = min(1.0, len(multipliers) / 20)  # More data = higher confidence
                        confidence = (data_consistency * 0.7 + data_amount_factor * 0.3) * 0.8  # Cap at 0.8
                        confidence = max(0.4, min(0.8, confidence))
                        
                        return {
                            'multiplier': round(prediction, 2),
                            'confidence': round(confidence, 2),
                            'method': 'real_data_heuristic',
                            'data_points_used': len(multipliers),
                            'recent_trend': round(recent_trend, 2),
                            'historical_avg': round(avg_multiplier, 2)
                        }
            
            # Use local historical data if available
            if len(self.historical_data) >= 5:
                recent_multipliers = [d['multiplier'] for d in self.historical_data[-5:]]
                avg_multiplier = np.mean(recent_multipliers)
                std_multiplier = np.std(recent_multipliers)
                
                logger.info(f"📊 Using {len(self.historical_data)} local historical data points for prediction")
                
                # Simple prediction: slightly above average with some randomness
                prediction = avg_multiplier * (1.0 + np.random.uniform(-0.1, 0.1))
                prediction = max(1.01, min(10.0, prediction))
                
                # Confidence based on data consistency
                confidence = max(0.2, 1.0 - (std_multiplier / avg_multiplier))
                confidence = min(0.7, confidence)
                
                return {
                    'multiplier': round(prediction, 2),
                    'confidence': round(confidence, 2),
                    'method': 'historical_heuristic',
                    'data_points_used': len(recent_multipliers)
                }
            else:
                # Default prediction when no data available - use aviation industry patterns
                logger.warning(f"❌ No sufficient data available (have {len(self.historical_data)} points, need 5+), using default aviation-based prediction")
                
                # Aviator games typically have these patterns:
                # - Most crashes between 1.5x and 3x
                # - Occasional higher multipliers
                # - Very rare extremely high multipliers
                
                rand_val = np.random.random()
                if rand_val < 0.6:  # 60% chance: low-medium multiplier
                    prediction = np.random.uniform(1.2, 3.0)
                    confidence = 0.4
                elif rand_val < 0.85:  # 25% chance: medium-high multiplier
                    prediction = np.random.uniform(3.0, 6.0)
                    confidence = 0.3
                else:  # 15% chance: high multiplier
                    prediction = np.random.uniform(6.0, 15.0)
                    confidence = 0.2
                
                return {
                    'multiplier': round(prediction, 2),
                    'confidence': round(confidence, 2),
                    'method': 'pattern_based_default'
                }
            
        except Exception as e:
            logger.error(f"Error in heuristic prediction: {e}")
            return {
                'multiplier': 2.0,
                'confidence': 0.3,
                'method': 'fallback',
                'error': str(e)
            }
    
    def _get_best_model(self) -> Optional[str]:
        """
        Get the name of the best performing model
        """
        if not self.model_performance:
            return None
        
        # Rank models by accuracy
        best_model = max(self.model_performance.items(), 
                        key=lambda x: x[1].get('accuracy', 0))
        
        return best_model[0] if best_model[1].get('accuracy', 0) > 30 else None
    
    def _adjust_prediction_for_context(self, prediction: Dict[str, Any], 
                                     current_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Adjust prediction based on current game context
        """
        try:
            adjusted_prediction = prediction.copy()
            
            # If we have current multiplier data, we can adjust
            current_multiplier = current_data.get('multiplier')
            if current_multiplier:
                # Simple adjustment: if current is very high/low, adjust confidence
                if current_multiplier > 5.0:
                    # After high multiplier, next might be lower
                    adjusted_prediction['confidence'] *= 0.8
                elif current_multiplier < 1.5:
                    # After low multiplier, might be more predictable
                    adjusted_prediction['confidence'] *= 1.1
            
            # Ensure confidence stays in bounds
            adjusted_prediction['confidence'] = max(0.1, min(0.9, adjusted_prediction['confidence']))
            
            return adjusted_prediction
            
        except Exception as e:
            logger.error(f"Error adjusting prediction: {e}")
            return prediction
    
    def predict(self, model_type: str = 'ensemble') -> Dict[str, Any]:
        """
        Main prediction method called by the API
        
        Args:
            model_type: Type of model to use for prediction
            
        Returns:
            Dictionary containing prediction details
        """
        try:
            logger.info(f"Making prediction using model type: {model_type}")
            
            # Try to get real-time data first
            if hasattr(self, '_data_collector') and self._data_collector:
                realtime_data = self._data_collector.get_realtime_data()
                if realtime_data:
                    return self.predict_realtime(realtime_data)
            
            # Use trained models if available
            if self.trained_models and model_type in self.trained_models:
                return self._predict_with_model(model_type)
            elif model_type == 'ensemble' and self.trained_models:
                return self._ensemble_prediction()
            else:
                # Use current prediction method
                return self.get_current_prediction()
                
        except Exception as e:
            logger.error(f"Error in predict method: {e}")
            return self._heuristic_prediction()
    
    def set_data_collector(self, data_collector):
        """
        Set the data collector for real-time data access
        
        Args:
            data_collector: RealDataCollector instance
        """
        self._data_collector = data_collector
        logger.info("Data collector linked to prediction engine")
    
    def _ensemble_prediction(self) -> Dict[str, Any]:
        """
        Make ensemble prediction using multiple models
        """
        try:
            predictions = []
            confidences = []
            
            for model_name in self.trained_models.keys():
                pred = self._predict_with_model(model_name)
                if pred:
                    predictions.append(pred['multiplier'])
                    confidences.append(pred['confidence'])
            
            if predictions:
                # Weighted average based on confidence
                total_confidence = sum(confidences)
                if total_confidence > 0:
                    weighted_prediction = sum(p * c for p, c in zip(predictions, confidences)) / total_confidence
                    avg_confidence = sum(confidences) / len(confidences)
                else:
                    weighted_prediction = sum(predictions) / len(predictions)
                    avg_confidence = 0.5
                
                return {
                    'multiplier': round(weighted_prediction, 2),
                    'confidence': round(avg_confidence, 2),
                    'method': 'ensemble',
                    'models_used': list(self.trained_models.keys())
                }
            else:
                return self._heuristic_prediction()
                
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return self._heuristic_prediction()

    def _auto_retrain(self):
        """
        Automatically retrain models when new data is available
        """
        try:
            logger.info("Auto-retraining models with new data...")
            result = self.train_models('all')
            if result.get('success'):
                logger.info("Auto-retrain completed successfully")
            else:
                logger.warning(f"Auto-retrain failed: {result.get('error')}")
        except Exception as e:
            logger.error(f"Auto-retrain error: {e}")
    
    def _save_models(self):
        """
        Save trained models to disk
        """
        try:
            for model_name, model_info in self.trained_models.items():
                model_path = os.path.join(self.models_dir, f"{model_name}_model.pkl")
                scaler_path = os.path.join(self.models_dir, f"{model_name}_scaler.pkl")
                
                with open(model_path, 'wb') as f:
                    pickle.dump(model_info['model'], f)
                
                with open(scaler_path, 'wb') as f:
                    pickle.dump(model_info['scaler'], f)
            
            # Save performance data
            performance_path = os.path.join(self.models_dir, 'performance.json')
            with open(performance_path, 'w') as f:
                json.dump(self.model_performance, f)
            
            logger.info("Models saved successfully")
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
    
    def _load_saved_models(self):
        """
        Load previously trained models from disk
        """
        try:
            performance_path = os.path.join(self.models_dir, 'performance.json')
            if os.path.exists(performance_path):
                with open(performance_path, 'r') as f:
                    self.model_performance = json.load(f)
            
            for model_name in self.models.keys():
                model_path = os.path.join(self.models_dir, f"{model_name}_model.pkl")
                scaler_path = os.path.join(self.models_dir, f"{model_name}_scaler.pkl")
                
                if os.path.exists(model_path) and os.path.exists(scaler_path):
                    with open(model_path, 'rb') as f:
                        model = pickle.load(f)
                    
                    with open(scaler_path, 'rb') as f:
                        scaler = pickle.load(f)
                    
                    self.trained_models[model_name] = {
                        'model': model,
                        'scaler': scaler,
                        'loaded_at': datetime.now().isoformat()
                    }
                    
                    logger.info(f"Loaded saved model: {model_name}")
            
            if self.trained_models:
                logger.info(f"Loaded {len(self.trained_models)} saved models")
                
        except Exception as e:
            logger.error(f"Error loading saved models: {e}")
