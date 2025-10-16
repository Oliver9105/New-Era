"""
Feature 9: Data Modeling
Advanced machine learning models and statistical analysis
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import tensorflow as tf
from tensorflow import keras
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import logging
import joblib
import json
import os
import random
import pickle

logger = logging.getLogger(__name__)

class PredictionEngine:
    def __init__(self):
        self.active = False
        self.models = {
            'linear_regression': None,
            'random_forest': None,
            'gradient_boosting': None,
            'neural_network': None,
            'lstm': None
        }
        self.scalers = {}
        self.training_data = []
        self.model_performance = {}
        self.ensemble_weights = [0.2, 0.25, 0.25, 0.15, 0.15]
        self.feature_importance = {}
        self.prediction_history = []
        
    def initialize(self):
        """Initialize the prediction engine"""
        self.active = True
        self._generate_training_data()
        self._initialize_models()
        logger.info("Prediction Engine initialized")
        
    def is_active(self):
        return self.active
        
    def _generate_training_data(self):
        """Generate synthetic training data for demonstration"""
        np.random.seed(42)
        n_samples = 5000
        
        # Generate realistic aviator multiplier patterns
        data = []
        
        for i in range(n_samples):
            # Base features
            hour = random.randint(0, 23)
            day_of_week = random.randint(0, 6)
            players_count = random.randint(50, 500)
            total_bet = random.uniform(1000, 50000)
            
            # Historical features (last 5 multipliers)
            if i < 5:
                hist_multipliers = [random.uniform(1.0, 3.0) for _ in range(5)]
            else:
                hist_multipliers = [data[j]['multiplier'] for j in range(i-5, i)]
            
            # Technical indicators
            ma_5 = np.mean(hist_multipliers)
            volatility = np.std(hist_multipliers) if len(hist_multipliers) > 1 else 0
            trend = hist_multipliers[-1] - hist_multipliers[0] if len(hist_multipliers) >= 2 else 0
            
            # Generate target multiplier with some patterns
            base_multiplier = np.random.exponential(1.5) + 1.0
            
            # Add time-based patterns
            if hour in [20, 21, 22]:  # Peak hours
                base_multiplier *= 1.1
            if day_of_week in [5, 6]:  # Weekends
                base_multiplier *= 1.05
                
            # Add momentum patterns
            if ma_5 > 2.0:
                base_multiplier *= 1.02
            if volatility > 0.5:
                base_multiplier *= random.choice([0.95, 1.05])
                
            # Ensure realistic range
            multiplier = max(1.0, min(base_multiplier, 100.0))
            
            features = {
                'hour': hour,
                'day_of_week': day_of_week,
                'players_count': players_count,
                'total_bet': total_bet,
                'hist_1': hist_multipliers[-1],
                'hist_2': hist_multipliers[-2],
                'hist_3': hist_multipliers[-3],
                'hist_4': hist_multipliers[-4],
                'hist_5': hist_multipliers[-5],
                'ma_5': ma_5,
                'volatility': volatility,
                'trend': trend,
                'multiplier': multiplier
            }
            
            data.append(features)
            
        self.training_data = data
        logger.info(f"Generated {len(data)} training samples")
        
    def _initialize_models(self):
        """Initialize machine learning models"""
        try:
            # Linear Regression
            self.models['linear_regression'] = LinearRegression()
            
            # Random Forest
            self.models['random_forest'] = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
            
            # Gradient Boosting
            self.models['gradient_boosting'] = GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                random_state=42
            )
            
            # Neural Network
            self.models['neural_network'] = MLPRegressor(
                hidden_layer_sizes=(100, 50, 25),
                activation='relu',
                solver='adam',
                max_iter=500,
                random_state=42
            )
            
            # Initialize scalers
            self.scalers['standard'] = StandardScaler()
            self.scalers['minmax'] = MinMaxScaler()
            
            logger.info("Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            
    def train_models(self, model_type: str = 'all') -> Dict:
        """Train prediction models"""
        try:
            if not self.training_data:
                return {'error': 'No training data available'}
                
            # Prepare data
            df = pd.DataFrame(self.training_data)
            
            # Features and target
            feature_columns = ['hour', 'day_of_week', 'players_count', 'total_bet',
                             'hist_1', 'hist_2', 'hist_3', 'hist_4', 'hist_5',
                             'ma_5', 'volatility', 'trend']
            
            X = df[feature_columns]
            y = df['multiplier']
            
            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scalers['standard'].fit_transform(X_train)
            X_test_scaled = self.scalers['standard'].transform(X_test)
            
            training_results = {
                'timestamp': datetime.utcnow().isoformat(),
                'models_trained': [],
                'performance_metrics': {}
            }
            
            models_to_train = [model_type] if model_type != 'all' else list(self.models.keys())
            
            for model_name in models_to_train:
                if model_name == 'lstm':
                    # Train LSTM separately
                    lstm_results = self._train_lstm_model(X_train, y_train, X_test, y_test)
                    training_results['models_trained'].append(model_name)
                    training_results['performance_metrics'][model_name] = lstm_results
                    continue
                    
                if model_name not in self.models:
                    continue
                    
                try:
                    model = self.models[model_name]
                    
                    # Use scaled data for neural network, original for tree-based models
                    if model_name == 'neural_network':
                        model.fit(X_train_scaled, y_train)
                        y_pred = model.predict(X_test_scaled)
                    else:
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                    
                    # Calculate metrics
                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)
                    
                    # Cross-validation score
                    if model_name == 'neural_network':
                        cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='r2')
                    else:
                        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='r2')
                    
                    metrics = {
                        'mse': mse,
                        'rmse': rmse,
                        'mae': mae,
                        'r2_score': r2,
                        'cv_mean': cv_scores.mean(),
                        'cv_std': cv_scores.std()
                    }
                    
                    # Feature importance for tree-based models
                    if hasattr(model, 'feature_importances_'):
                        feature_importance = dict(zip(feature_columns, model.feature_importances_))
                        self.feature_importance[model_name] = feature_importance
                        metrics['feature_importance'] = feature_importance
                    
                    self.model_performance[model_name] = metrics
                    training_results['models_trained'].append(model_name)
                    training_results['performance_metrics'][model_name] = metrics
                    
                    logger.info(f"Trained {model_name} - R2: {r2:.4f}, RMSE: {rmse:.4f}")
                    
                except Exception as e:
                    logger.error(f"Error training {model_name}: {e}")
                    training_results['performance_metrics'][model_name] = {'error': str(e)}
            
            # Update ensemble weights based on performance
            self._update_ensemble_weights()
            
            return training_results
            
        except Exception as e:
            logger.error(f"Error in train_models: {e}")
            return {'error': str(e)}
            
    def _train_lstm_model(self, X_train, y_train, X_test, y_test) -> Dict:
        """Train LSTM model for time series prediction"""
        try:
            # Prepare data for LSTM (sequence-based)
            sequence_length = 10
            
            def create_sequences(data, seq_length):
                sequences = []
                targets = []
                for i in range(len(data) - seq_length):
                    sequences.append(data[i:(i + seq_length)])
                    targets.append(data[i + seq_length])
                return np.array(sequences), np.array(targets)
            
            # Use only the multiplier history for LSTM
            multiplier_data = y_train.values
            
            if len(multiplier_data) < sequence_length + 10:
                return {'error': 'Insufficient data for LSTM training'}
            
            X_seq, y_seq = create_sequences(multiplier_data, sequence_length)
            
            # Split sequences
            split_idx = int(len(X_seq) * 0.8)
            X_train_seq = X_seq[:split_idx]
            y_train_seq = y_seq[:split_idx]
            X_val_seq = X_seq[split_idx:]
            y_val_seq = y_seq[split_idx:]
            
            # Build LSTM model
            model = keras.Sequential([
                keras.layers.LSTM(50, return_sequences=True, input_shape=(sequence_length, 1)),
                keras.layers.Dropout(0.2),
                keras.layers.LSTM(50, return_sequences=False),
                keras.layers.Dropout(0.2),
                keras.layers.Dense(25),
                keras.layers.Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse', metrics=['mae'])
            
            # Reshape data for LSTM
            X_train_seq = X_train_seq.reshape((X_train_seq.shape[0], X_train_seq.shape[1], 1))
            X_val_seq = X_val_seq.reshape((X_val_seq.shape[0], X_val_seq.shape[1], 1))
            
            # Train model
            history = model.fit(
                X_train_seq, y_train_seq,
                epochs=50,
                batch_size=32,
                validation_data=(X_val_seq, y_val_seq),
                verbose=0
            )
            
            # Evaluate
            y_pred = model.predict(X_val_seq)
            mse = mean_squared_error(y_val_seq, y_pred)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_val_seq, y_pred)
            r2 = r2_score(y_val_seq, y_pred)
            
            self.models['lstm'] = model
            
            return {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2_score': r2,
                'training_loss': history.history['loss'][-1],
                'validation_loss': history.history['val_loss'][-1]
            }
            
        except Exception as e:
            logger.error(f"Error training LSTM: {e}")
            return {'error': str(e)}
            
    def predict(self, model_type: str = 'ensemble') -> Dict:
        """Generate prediction using specified model or ensemble"""
        try:
            # Generate current features (simulated)
            current_features = self._get_current_features()
            
            if model_type == 'ensemble':
                return self._ensemble_prediction(current_features)
            elif model_type in self.models and self.models[model_type] is not None:
                return self._single_model_prediction(model_type, current_features)
            else:
                return {'error': f'Model {model_type} not available'}
                
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            return {'error': str(e)}
            
    def _get_current_features(self) -> Dict:
        """Get current features for prediction (simulated)"""
        # In real implementation, this would get actual current data
        current_time = datetime.utcnow()
        
        # Simulate recent multiplier history
        recent_multipliers = [random.uniform(1.0, 5.0) for _ in range(5)]
        
        features = {
            'hour': current_time.hour,
            'day_of_week': current_time.weekday(),
            'players_count': random.randint(100, 400),
            'total_bet': random.uniform(5000, 30000),
            'hist_1': recent_multipliers[4],
            'hist_2': recent_multipliers[3],
            'hist_3': recent_multipliers[2],
            'hist_4': recent_multipliers[1],
            'hist_5': recent_multipliers[0],
            'ma_5': np.mean(recent_multipliers),
            'volatility': np.std(recent_multipliers),
            'trend': recent_multipliers[4] - recent_multipliers[0]
        }
        
        return features
        
    def _single_model_prediction(self, model_type: str, features: Dict) -> Dict:
        """Make prediction using a single model"""
        try:
            model = self.models[model_type]
            
            if model is None:
                return {'error': f'Model {model_type} not trained'}
            
            # Prepare features
            feature_columns = ['hour', 'day_of_week', 'players_count', 'total_bet',
                             'hist_1', 'hist_2', 'hist_3', 'hist_4', 'hist_5',
                             'ma_5', 'volatility', 'trend']
            
            feature_array = np.array([[features[col] for col in feature_columns]])
            
            # Handle LSTM separately
            if model_type == 'lstm':
                # For LSTM, use recent multiplier sequence
                sequence = np.array([features[f'hist_{i}'] for i in range(5, 0, -1)])
                sequence = sequence.reshape(1, 5, 1)
                prediction = model.predict(sequence, verbose=0)[0][0]
            else:
                # Scale features if needed
                if model_type == 'neural_network':
                    feature_array = self.scalers['standard'].transform(feature_array)
                
                prediction = model.predict(feature_array)[0]
            
            # Calculate confidence based on model performance
            performance = self.model_performance.get(model_type, {})
            r2_score = performance.get('r2_score', 0)
            confidence = max(0.1, min(0.95, r2_score))
            
            return {
                'model_type': model_type,
                'predicted_multiplier': float(prediction),
                'confidence': confidence,
                'timestamp': datetime.utcnow().isoformat(),
                'features_used': features
            }
            
        except Exception as e:
            logger.error(f"Error in single model prediction: {e}")
            return {'error': str(e)}
            
    def _ensemble_prediction(self, features: Dict) -> Dict:
        """Make ensemble prediction using multiple models"""
        try:
            predictions = []
            weights = []
            model_results = {}
            
            for i, model_name in enumerate(self.models.keys()):
                if self.models[model_name] is not None:
                    try:
                        result = self._single_model_prediction(model_name, features)
                        if 'predicted_multiplier' in result:
                            predictions.append(result['predicted_multiplier'])
                            weights.append(self.ensemble_weights[i] if i < len(self.ensemble_weights) else 0.2)
                            model_results[model_name] = result
                    except Exception as e:
                        logger.warning(f"Error in {model_name} prediction: {e}")
            
            if not predictions:
                return {'error': 'No models available for prediction'}
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                weights = [w / total_weight for w in weights]
            
            # Weighted average
            ensemble_prediction = sum(p * w for p, w in zip(predictions, weights))
            
            # Calculate ensemble confidence
            prediction_variance = np.var(predictions)
            ensemble_confidence = max(0.1, min(0.95, 1.0 - prediction_variance / 10))
            
            result = {
                'model_type': 'ensemble',
                'predicted_multiplier': float(ensemble_prediction),
                'confidence': ensemble_confidence,
                'timestamp': datetime.utcnow().isoformat(),
                'individual_predictions': model_results,
                'ensemble_weights': dict(zip(model_results.keys(), weights)),
                'prediction_variance': prediction_variance
            }
            
            # Store prediction history
            self.prediction_history.append(result)
            
            # Keep only last 1000 predictions
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-1000:]
            
            return result
            
        except Exception as e:
            logger.error(f"Error in ensemble prediction: {e}")
            return {'error': str(e)}
            
    def _update_ensemble_weights(self):
        """Update ensemble weights based on model performance"""
        try:
            weights = []
            
            for model_name in self.models.keys():
                performance = self.model_performance.get(model_name, {})
                r2_score = performance.get('r2_score', 0)
                
                # Weight based on R2 score
                weight = max(0.05, r2_score) if r2_score > 0 else 0.05
                weights.append(weight)
            
            # Normalize weights
            total_weight = sum(weights)
            if total_weight > 0:
                self.ensemble_weights = [w / total_weight for w in weights]
            
            logger.info(f"Updated ensemble weights: {dict(zip(self.models.keys(), self.ensemble_weights))}")
            
        except Exception as e:
            logger.error(f"Error updating ensemble weights: {e}")
            
    def predict_realtime(self, realtime_data: Dict) -> Dict:
        """Make real-time prediction based on current data"""
        try:
            # Extract features from real-time data
            features = {
                'hour': datetime.utcnow().hour,
                'day_of_week': datetime.utcnow().weekday(),
                'players_count': realtime_data.get('players_count', 200),
                'total_bet': realtime_data.get('total_bet', 15000),
                'hist_1': realtime_data.get('current_multiplier', 2.0),
                'hist_2': 1.8,  # Would get from actual history
                'hist_3': 2.2,
                'hist_4': 1.5,
                'hist_5': 3.1,
                'ma_5': 2.12,
                'volatility': 0.6,
                'trend': 0.5
            }
            
            # Update features with real data if available
            if 'historical_multipliers' in realtime_data:
                hist = realtime_data['historical_multipliers'][-5:]
                for i, mult in enumerate(hist):
                    features[f'hist_{len(hist)-i}'] = mult
                
                features['ma_5'] = np.mean(hist)
                features['volatility'] = np.std(hist) if len(hist) > 1 else 0
                features['trend'] = hist[-1] - hist[0] if len(hist) >= 2 else 0
            
            # Make ensemble prediction
            prediction = self._ensemble_prediction(features)
            
            # Add real-time specific information
            prediction['realtime_features'] = features
            prediction['data_source'] = 'realtime'
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in real-time prediction: {e}")
            return {'error': str(e)}
            
    def get_model_analytics(self) -> Dict:
        """Get analytics about model performance"""
        try:
            analytics = {
                'model_performance': self.model_performance,
                'ensemble_weights': dict(zip(self.models.keys(), self.ensemble_weights)),
                'feature_importance': self.feature_importance,
                'prediction_history_count': len(self.prediction_history),
                'training_data_size': len(self.training_data),
                'models_trained': [name for name, model in self.models.items() if model is not None]
            }
            
            # Recent prediction accuracy (simulated)
            if self.prediction_history:
                recent_predictions = self.prediction_history[-50:]
                avg_confidence = np.mean([p['confidence'] for p in recent_predictions])
                analytics['recent_average_confidence'] = avg_confidence
            
            return analytics
            
        except Exception as e:
            logger.error(f"Error getting model analytics: {e}")
            return {'error': str(e)}
            
    def save_models(self, directory: str = 'models') -> Dict:
        """Save trained models to disk"""
        try:
            os.makedirs(directory, exist_ok=True)
            
            saved_models = []
            
            for model_name, model in self.models.items():
                if model is not None:
                    try:
                        if model_name == 'lstm':
                            model.save(os.path.join(directory, f'{model_name}_model.h5'))
                        else:
                            joblib.dump(model, os.path.join(directory, f'{model_name}_model.pkl'))
                        saved_models.append(model_name)
                    except Exception as e:
                        logger.error(f"Error saving {model_name}: {e}")
            
            # Save scalers
            joblib.dump(self.scalers, os.path.join(directory, 'scalers.pkl'))
            
            # Save metadata
            metadata = {
                'ensemble_weights': self.ensemble_weights,
                'model_performance': self.model_performance,
                'feature_importance': self.feature_importance,
                'save_timestamp': datetime.utcnow().isoformat()
            }
            
            with open(os.path.join(directory, 'metadata.json'), 'w') as f:
                json.dump(metadata, f, indent=2)
            
            return {
                'success': True,
                'saved_models': saved_models,
                'save_directory': directory,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error saving models: {e}")
            return {'success': False, 'error': str(e)}
            
    def load_models(self, directory: str = 'models') -> Dict:
        """Load trained models from disk"""
        try:
            loaded_models = []
            
            for model_name in self.models.keys():
                try:
                    if model_name == 'lstm':
                        model_path = os.path.join(directory, f'{model_name}_model.h5')
                        if os.path.exists(model_path):
                            self.models[model_name] = keras.models.load_model(model_path)
                            loaded_models.append(model_name)
                    else:
                        model_path = os.path.join(directory, f'{model_name}_model.pkl')
                        if os.path.exists(model_path):
                            self.models[model_name] = joblib.load(model_path)
                            loaded_models.append(model_name)
                except Exception as e:
                    logger.error(f"Error loading {model_name}: {e}")
            
            # Load scalers
            scalers_path = os.path.join(directory, 'scalers.pkl')
            if os.path.exists(scalers_path):
                self.scalers = joblib.load(scalers_path)
            
            # Load metadata
            metadata_path = os.path.join(directory, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                
                self.ensemble_weights = metadata.get('ensemble_weights', self.ensemble_weights)
                self.model_performance = metadata.get('model_performance', {})
                self.feature_importance = metadata.get('feature_importance', {})
            
            return {
                'success': True,
                'loaded_models': loaded_models,
                'load_directory': directory,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return {'success': False, 'error': str(e)}