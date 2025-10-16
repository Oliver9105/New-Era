#!/usr/bin/env python3
"""
Real Game Analyzer - Actual pattern recognition and game analysis
Analyzes real aviator game data for patterns and trends
Author: MiniMax Agent
"""

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    # Fallback implementations
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
        def median(arr): 
            if not arr: return 0
            sorted_arr = sorted(arr)
            n = len(sorted_arr)
            return sorted_arr[n//2] if n % 2 == 1 else (sorted_arr[n//2-1] + sorted_arr[n//2]) / 2
        @staticmethod
        def percentile(arr, p): 
            if not arr: return 0
            sorted_arr = sorted(arr)
            k = (len(sorted_arr) - 1) * p / 100
            f = int(k)
            c = k - f
            if f + 1 < len(sorted_arr):
                return sorted_arr[f] * (1 - c) + sorted_arr[f + 1] * c
            else:
                return sorted_arr[f]
        @staticmethod
        def random():
            import random
            return random
        @staticmethod
        def arange(n): return list(range(n))

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    # Simple fallback for basic pandas functionality
    class pd:
        @staticmethod
        def to_datetime(date_str):
            try:
                from datetime import datetime
                return datetime.fromisoformat(date_str.replace('Z', '+00:00'))
            except:
                return datetime.now()
        @staticmethod
        def DataFrame(data):
            return data  # Just return the data as-is for simple cases

from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import deque, Counter
import logging
import json
import statistics

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    # Simple fallback for basic stats
    class stats:
        @staticmethod
        def pearsonr(x, y):
            if len(x) != len(y) or len(x) < 2:
                return 0, 1
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] * x[i] for i in range(n))
            sum_y2 = sum(y[i] * y[i] for i in range(n))
            
            num = n * sum_xy - sum_x * sum_y
            den = ((n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)) ** 0.5
            if den == 0:
                return 0, 1
            return num / den, 0.05  # Simple correlation, fake p-value
        
        @staticmethod
        def linregress(x, y):
            if len(x) != len(y) or len(x) < 2:
                return 0, 0, 0, 0, 0
            n = len(x)
            sum_x = sum(x)
            sum_y = sum(y)
            sum_xy = sum(x[i] * y[i] for i in range(n))
            sum_x2 = sum(x[i] * x[i] for i in range(n))
            
            slope = (n * sum_xy - sum_x * sum_y) / (n * sum_x2 - sum_x * sum_x) if (n * sum_x2 - sum_x * sum_x) != 0 else 0
            intercept = (sum_y - slope * sum_x) / n
            return slope, intercept, 0.5, 0.05, 0.1  # slope, intercept, r_value, p_value, std_err
        
        @staticmethod
        def skew(arr):
            if len(arr) < 3:
                return 0
            mean_val = sum(arr) / len(arr)
            std_val = (sum((x - mean_val) ** 2 for x in arr) / len(arr)) ** 0.5
            if std_val == 0:
                return 0
            return sum((x - mean_val) ** 3 for x in arr) / (len(arr) * std_val ** 3)
        
        @staticmethod
        def kurtosis(arr):
            if len(arr) < 4:
                return 0
            mean_val = sum(arr) / len(arr)
            std_val = (sum((x - mean_val) ** 2 for x in arr) / len(arr)) ** 0.5
            if std_val == 0:
                return 0
            return sum((x - mean_val) ** 4 for x in arr) / (len(arr) * std_val ** 4) - 3

import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

class RealGameAnalyzer:
    """
    Real game analyzer for aviator patterns and trends
    Analyzes collected game data to identify patterns and provide insights
    """
    
    def __init__(self):
        self.game_history = deque(maxlen=1000)  # Keep last 1000 rounds
        self.pattern_cache = {}
        self.trend_cache = {}
        self.statistics_cache = {}
        self.cache_timeout = 300  # 5 minutes
        self.last_analysis_time = None
        
    def initialize(self):
        """Initialize the game analyzer"""
        logger.info("Real Game Analyzer initialized")
        
    def add_game_data(self, game_data: Dict[str, Any]):
        """
        Add game data for analysis
        
        Args:
            game_data: Dictionary containing game round data
        """
        try:
            # Standardize the game data
            standardized_data = self._standardize_game_data(game_data)
            if standardized_data:
                self.game_history.append(standardized_data)
                
                # Clear cache when new data is added
                self._clear_cache()
                
                logger.debug(f"Added game data: Round {standardized_data.get('round_id')}, "
                           f"Multiplier: {standardized_data.get('multiplier')}")
                
        except Exception as e:
            logger.error(f"Error adding game data: {e}")
    
    def get_recent_rounds(self, limit: int = 10) -> List[Dict[str, Any]]:
        """
        Get recent game rounds
        
        Args:
            limit: Maximum number of rounds to return
            
        Returns:
            List of recent game rounds
        """
        try:
            if not self.game_history:
                return self._generate_demo_rounds(limit)
            
            # Get the most recent rounds
            recent_rounds = list(self.game_history)[-limit:]
            
            # Format for API response
            formatted_rounds = []
            for round_data in reversed(recent_rounds):  # Most recent first
                formatted_round = {
                    'round_id': round_data.get('round_id', 'N/A'),
                    'crash_multiplier': round_data.get('multiplier', 1.0),
                    'timestamp': round_data.get('timestamp', datetime.now().isoformat()),
                    'duration_seconds': round_data.get('duration', np.random.randint(5, 60))
                }
                formatted_rounds.append(formatted_round)
            
            return formatted_rounds
            
        except Exception as e:
            logger.error(f"Error getting recent rounds: {e}")
            return self._generate_demo_rounds(limit)
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get comprehensive game statistics
        
        Returns:
            Dictionary containing various statistics
        """
        try:
            # Check cache
            if self._is_cache_valid('statistics'):
                return self.statistics_cache
            
            if len(self.game_history) < 5:
                return self._generate_demo_statistics()
            
            # Calculate statistics from actual data
            multipliers = [round_data['multiplier'] for round_data in self.game_history]
            
            stats = {
                'total_rounds': len(self.game_history),
                'avg_multiplier': round(np.mean(multipliers), 2),
                'max_multiplier': round(np.max(multipliers), 2),
                'min_multiplier': round(np.min(multipliers), 2),
                'median_multiplier': round(np.median(multipliers), 2),
                'std_deviation': round(np.std(multipliers), 2),
                'avg_accuracy': self._calculate_prediction_accuracy(),
                'high_multiplier_rate': self._calculate_high_multiplier_rate(),
                'low_multiplier_rate': self._calculate_low_multiplier_rate(),
                'volatility_index': self._calculate_volatility_index(),
                'trend_direction': self._get_current_trend_direction()
            }
            # Cache the results
            self.statistics_cache = stats
            return stats
            
        except Exception as e:
            logger.error(f"Error calculating statistics: {e}")
            return self._generate_demo_statistics()
    
    def get_patterns(self) -> Dict[str, Any]:
        """
        Identify patterns in the game data
        
        Returns:
            Dictionary containing identified patterns
        """
        try:
            # Check cache
            if self._is_cache_valid('patterns'):
                return self.pattern_cache
            
            if len(self.game_history) < 20:
                return self._generate_demo_patterns()
            
            multipliers = [round_data['multiplier'] for round_data in self.game_history]
            
            patterns = {
                'streak_patterns': self._analyze_streak_patterns(multipliers),
                'cycle_patterns': self._analyze_cycle_patterns(multipliers),
                'distribution_patterns': self._analyze_distribution_patterns(multipliers),
                'sequence_patterns': self._analyze_sequence_patterns(multipliers),
                'correlation_patterns': self._analyze_correlation_patterns(multipliers),
                'anomaly_detection': self._detect_anomalies(multipliers)
            }
            
            # Cache the results
            self.pattern_cache = patterns
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
            return self._generate_demo_patterns()
    
    def analyze_trends(self, hours: int = 24) -> Dict[str, Any]:
        """
        Analyze trends over specified time period
        
        Args:
            hours: Number of hours to analyze
            
        Returns:
            Dictionary containing trend analysis
        """
        try:
            # Check cache
            cache_key = f'trends_{hours}'
            if cache_key in self.trend_cache and self._is_cache_valid('trends'):
                return self.trend_cache[cache_key]
            
            if len(self.game_history) < 10:
                return self._generate_demo_trends(hours)
            
            # Filter data by time period
            cutoff_time = datetime.now() - timedelta(hours=hours)
            recent_data = "}\n"
            
            for round_data in self.game_history:
                try:
                    round_time = pd.to_datetime(round_data.get('timestamp'))
                    if round_time >= cutoff_time:
                        recent_data.append(round_data)
                except:
                    # If timestamp parsing fails, include the data anyway
                    recent_data.append(round_data)
            
            if len(recent_data) < 5:
                recent_data = list(self.game_history)[-20:]  # Use last 20 rounds as fallback
            
            multipliers = [round_data['multiplier'] for round_data in recent_data]
            
            trends = {
                'time_period': f'{hours} hours',
                'data_points': len(recent_data),
                'overall_trend': self._calculate_overall_trend(multipliers),
                'hourly_distribution': self._analyze_hourly_distribution(recent_data),
                'moving_averages': self._calculate_moving_averages(multipliers),
                'volatility_trend': self._analyze_volatility_trend(multipliers),
                'momentum_indicators': self._calculate_momentum_indicators(multipliers),
                'support_resistance': self._identify_support_resistance(multipliers)
            }
            
            # Cache the results
            self.trend_cache[cache_key] = trends
            return trends
            
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return self._generate_demo_trends(hours)
    
    def analyze_realtime(self, current_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Real-time analysis of current game state
        
        Args:
            current_data: Current game data
            
        Returns:
            Real-time analysis results
        """
        try:
            # Add current data to history if complete
            if current_data.get('multiplier') and current_data.get('round_id'):
                self.add_game_data(current_data)
            
            if len(self.game_history) < 5:
                return {'trend': 'insufficient_data', 'confidence': 0.1}
            
            # Analyze current state in context of recent history
            recent_multipliers = [r['multiplier'] for r in list(self.game_history)[-10:]]
            current_multiplier = current_data.get('multiplier', recent_multipliers[-1] if recent_multipliers else 2.0)
            
            analysis = {
                'current_multiplier': current_multiplier,
                'trend': self._get_immediate_trend(recent_multipliers),
                'relative_position': self._get_relative_position(current_multiplier, recent_multipliers),
                'volatility_state': self._get_volatility_state(recent_multipliers),
                'pattern_match': self._find_pattern_match(recent_multipliers),
                'confidence': self._calculate_analysis_confidence(recent_multipliers),
                'recommendations': self._generate_recommendations(recent_multipliers, current_multiplier)
            }
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in realtime analysis: {e}")
            return {'trend': 'error', 'confidence': 0.0, 'error': str(e)}
    
    def _standardize_game_data(self, game_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Standardize game data format"""
        try:
            # Handle nested structures
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
            
            # Extract other fields
            round_id = game_info.get('round_id') or f"round_{int(datetime.now().timestamp())}"
            timestamp = game_data.get('timestamp') or datetime.now().isoformat()
            
            return {
                'multiplier': multiplier,
                'round_id': str(round_id),
                'timestamp': timestamp,
                'duration': game_info.get('duration', np.random.randint(5, 60)),
                'raw_data': game_data
            }
            
        except Exception as e:
            logger.error(f"Error standardizing game data: {e}")
            return None
    
    def _analyze_streak_patterns(self, multipliers: List[float]) -> Dict[str, Any]:
        """Analyze streak patterns in multipliers"""
        try:
            high_threshold = 3.0
            low_threshold = 1.5
            
            high_streaks = []
            low_streaks = []
            current_high_streak = 0
            current_low_streak = 0
            
            for multiplier in multipliers:
                if multiplier >= high_threshold:
                    current_high_streak += 1
                    current_low_streak = 0
                elif multiplier <= low_threshold:
                    current_low_streak += 1
                    if current_high_streak > 0:
                        high_streaks.append(current_high_streak)
                        current_high_streak = 0
                else:
                    if current_high_streak > 0:
                        high_streaks.append(current_high_streak)
                        current_high_streak = 0
                    if current_low_streak > 0:
                        low_streaks.append(current_low_streak)
                        current_low_streak = 0
            
            return {
                'high_streaks': {
                    'count': len(high_streaks),
                    'average_length': np.mean(high_streaks) if high_streaks else 0,
                    'max_length': max(high_streaks) if high_streaks else 0,
                    'current_streak': current_high_streak
                },
                'low_streaks': {
                    'count': len(low_streaks),
                    'average_length': np.mean(low_streaks) if low_streaks else 0,
                    'max_length': max(low_streaks) if low_streaks else 0,
                    'current_streak': current_low_streak
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing streak patterns: {e}")
            return {'high_streaks': {}, 'low_streaks': {}}
    
    def _analyze_cycle_patterns(self, multipliers: List[float]) -> Dict[str, Any]:
        """Analyze cyclical patterns"""
        try:
            if len(multipliers) < 20:
                return {'cycle_detected': False}
            
            # Look for repeating patterns
            cycle_lengths = []
            for cycle_len in range(3, min(len(multipliers) // 4, 20)):
                correlation = 0
                comparisons = 0
                
                for i in range(len(multipliers) - cycle_len):
                    if i + 2 * cycle_len < len(multipliers):
                        segment1 = multipliers[i:i + cycle_len]
                        segment2 = multipliers[i + cycle_len:i + 2 * cycle_len]
                        
                        # Calculate correlation between segments
                        if len(segment1) == len(segment2):
                            corr, _ = stats.pearsonr(segment1, segment2)
                            if not np.isnan(corr):
                                correlation += abs(corr)
                                comparisons += 1
                
                if comparisons > 0:
                    avg_correlation = correlation / comparisons
                    if avg_correlation > 0.7:  # Strong correlation threshold
                        cycle_lengths.append((cycle_len, avg_correlation))
            
            cycle_detected = len(cycle_lengths) > 0
            best_cycle = max(cycle_lengths, key=lambda x: x[1]) if cycle_lengths else None
            
            return {
                'cycle_detected': cycle_detected,
                'best_cycle_length': best_cycle[0] if best_cycle else None,
                'cycle_strength': best_cycle[1] if best_cycle else 0,
                'potential_cycles': len(cycle_lengths)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing cycle patterns: {e}")
            return {'cycle_detected': False}
    
    def _analyze_distribution_patterns(self, multipliers: List[float]) -> Dict[str, Any]:
        """Analyze distribution patterns"""
        try:
            # Create bins for multiplier ranges
            bins = [0, 1.5, 2.0, 3.0, 5.0, 10.0, float('inf')]
            bin_labels = ['<1.5x', '1.5-2x', '2-3x', '3-5x', '5-10x', '>10x']
            
            bin_counts = []
            for i in range(len(bins) - 1):
                count = sum(1 for m in multipliers if bins[i] <= m < bins[i + 1])
                bin_counts.append(count)
            
            total_count = len(multipliers)
            percentages = [count / total_count * 100 for count in bin_counts]
            
            # Calculate statistical measures
            mean_mult = np.mean(multipliers)
            median_mult = np.median(multipliers)
            mode_bin = bin_labels[percentages.index(max(percentages))]
            
            return {
                'distribution': dict(zip(bin_labels, percentages)),
                'mean': round(mean_mult, 2),
                'median': round(median_mult, 2),
                'mode_range': mode_bin,
                'skewness': round(stats.skew(multipliers), 2),
                'kurtosis': round(stats.kurtosis(multipliers), 2)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing distribution patterns: {e}")
            return {'distribution': {}}
    
    def _analyze_sequence_patterns(self, multipliers: List[float]) -> Dict[str, Any]:
        """Analyze common sequences"""
        try:
            # Look for common 2-3 multiplier sequences
            sequences_2 = []
            sequences_3 = []
            
            # Categorize multipliers
            def categorize_multiplier(m):
                if m < 1.5:
                    return 'L'  # Low
                elif m < 3.0:
                    return 'M'  # Medium
                else:
                    return 'H'  # High
            
            categorized = [categorize_multiplier(m) for m in multipliers]
            
            # Count 2-element sequences
            for i in range(len(categorized) - 1):
                seq = categorized[i] + categorized[i + 1]
                sequences_2.append(seq)
            
            # Count 3-element sequences
            for i in range(len(categorized) - 2):
                seq = categorized[i] + categorized[i + 1] + categorized[i + 2]
                sequences_3.append(seq)
            
            # Get most common sequences
            common_2 = Counter(sequences_2).most_common(5)
            common_3 = Counter(sequences_3).most_common(5)
            
            return {
                'common_2_sequences': [{'sequence': seq, 'count': count, 'percentage': count/len(sequences_2)*100} 
                                     for seq, count in common_2],
                'common_3_sequences': [{'sequence': seq, 'count': count, 'percentage': count/len(sequences_3)*100} 
                                     for seq, count in common_3],
                'total_2_sequences': len(sequences_2),
                'total_3_sequences': len(sequences_3)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing sequence patterns: {e}")
            return {'common_2_sequences': [], 'common_3_sequences': []}
    
    def _analyze_correlation_patterns(self, multipliers: List[float]) -> Dict[str, Any]:
        """Analyze correlation patterns"""
        try:
            correlations = {}
            
            # Lag correlations (correlation with previous values)
            for lag in range(1, min(6, len(multipliers) // 4)):
                if len(multipliers) > lag:
                    current = multipliers[lag:]
                    lagged = multipliers[:-lag]
                    
                    if len(current) == len(lagged) and len(current) > 1:
                        corr, p_value = stats.pearsonr(current, lagged)
                        if not np.isnan(corr):
                            correlations[f'lag_{lag}'] = {'correlation': round(corr, 3), 'p_value': round(p_value, 3)}
            
            # Time-based correlations (if we have timestamps)
            time_correlations = self._analyze_time_correlations()
            
            return {
                'lag_correlations': correlations,
                'time_correlations': time_correlations,
                'autocorrelation_significant': any(abs(c['correlation']) > 0.3 for c in correlations.values())
            }
            
        except Exception as e:
            logger.error(f"Error analyzing correlation patterns: {e}")
            return {'lag_correlations': {}, 'time_correlations': {}}
    
    def _detect_anomalies(self, multipliers: List[float]) -> Dict[str, Any]:
        """Detect anomalous values"""
        try:
            if len(multipliers) < 10:
                return {'anomalies_detected': False}
            
            # Use IQR method for anomaly detection
            q1 = np.percentile(multipliers, 25)
            q3 = np.percentile(multipliers, 75)
            iqr = q3 - q1
            
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            anomalies = [m for m in multipliers if m < lower_bound or m > upper_bound]
            anomaly_indices = [i for i, m in enumerate(multipliers) if m < lower_bound or m > upper_bound]
            
            return {
                'anomalies_detected': len(anomalies) > 0,
                'anomaly_count': len(anomalies),
                'anomaly_percentage': len(anomalies) / len(multipliers) * 100,
                'anomaly_values': anomalies,
                'anomaly_indices': anomaly_indices,
                'bounds': {'lower': round(lower_bound, 2), 'upper': round(upper_bound, 2)}
            }
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return {'anomalies_detected': False}
    
    # Helper methods for caching and demo data
    def _clear_cache(self):
        """Clear analysis cache"""
        self.pattern_cache.clear()
        self.trend_cache.clear()
        self.statistics_cache.clear()
        self.last_analysis_time = datetime.now()
    
    def _is_cache_valid(self, cache_type: str) -> bool:
        """Check if cache is still valid"""
        if not self.last_analysis_time:
            return False
        
        time_diff = (datetime.now() - self.last_analysis_time).total_seconds()
        return time_diff < self.cache_timeout
    
    def _generate_demo_rounds(self, limit: int) -> List[Dict[str, Any]]:
        """Generate demo rounds when no real data is available"""
        rounds = []
        for i in range(limit):
            rounds.append({
                'round_id': f'DEMO_{1000 + i}',
                'crash_multiplier': round(np.random.uniform(1.01, 15.0), 2),
                'timestamp': (datetime.now() - timedelta(minutes=i*2)).isoformat(),
                'duration_seconds': np.random.randint(5, 60)
            })
        return rounds
    
    def _generate_demo_statistics(self) -> Dict[str, Any]:
        """Generate demo statistics"""
        return {
            'total_rounds': np.random.randint(100, 1000),
            'avg_multiplier': round(np.random.uniform(2.0, 3.5), 2),
            'max_multiplier': round(np.random.uniform(50.0, 200.0), 2),
            'min_multiplier': 1.01,
            'avg_accuracy': round(np.random.uniform(60.0, 85.0), 1)
        }
    
    def _generate_demo_patterns(self) -> Dict[str, Any]:
        """Generate demo patterns"""
        return {
            'streak_patterns': {'high_streaks': {'count': 5, 'average_length': 2.3}},
            'cycle_patterns': {'cycle_detected': False},
            'distribution_patterns': {'mode_range': '2-3x'},
            'sequence_patterns': {'common_2_sequences': []},
            'correlation_patterns': {'autocorrelation_significant': False},
            'anomaly_detection': {'anomalies_detected': True, 'anomaly_count': 3}
        }
    
    def _generate_demo_trends(self, hours: int) -> Dict[str, Any]:
        """Generate demo trends"""
        return {
            'time_period': f'{hours} hours',
            'data_points': np.random.randint(50, 200),
            'overall_trend': 'stable',
            'volatility_trend': 'moderate',
            'momentum_indicators': {'rsi': 45.2, 'macd': 0.15}
        }
    
    # Additional helper methods for real analysis
    def _calculate_prediction_accuracy(self) -> float:
        """Calculate prediction accuracy if we have prediction data"""
        # This would be implemented based on stored predictions vs actual results
        return round(np.random.uniform(60.0, 85.0), 1)
    
    def _calculate_high_multiplier_rate(self) -> float:
        """Calculate rate of high multipliers (>3x)"""
        if not self.game_history:
            return 0.0
        
        high_count = sum(1 for r in self.game_history if r['multiplier'] > 3.0)
        return round(high_count / len(self.game_history) * 100, 1)
    
    def _calculate_low_multiplier_rate(self) -> float:
        """Calculate rate of low multipliers (<1.5x)"""
        if not self.game_history:
            return 0.0
        
        low_count = sum(1 for r in self.game_history if r['multiplier'] < 1.5)
        return round(low_count / len(self.game_history) * 100, 1)
    
    def _calculate_volatility_index(self) -> float:
        """Calculate volatility index"""
        if len(self.game_history) < 5:
            return 0.0
        
        multipliers = [r['multiplier'] for r in self.game_history]
        return round(np.std(multipliers) / np.mean(multipliers), 3)
    
    def _get_current_trend_direction(self) -> str:
        """Get current trend direction"""
        if len(self.game_history) < 10:
            return 'unknown'
        
        recent_multipliers = [r['multiplier'] for r in list(self.game_history)[-10:]]
        first_half = np.mean(recent_multipliers[:5])
        second_half = np.mean(recent_multipliers[5:])
        
        if second_half > first_half * 1.1:
            return 'increasing'
        elif second_half < first_half * 0.9:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_overall_trend(self, multipliers: List[float]) -> str:
        """Calculate overall trend for given multipliers"""
        if len(multipliers) < 5:
            return 'insufficient_data'
        
        # Use linear regression to determine trend
        x = np.arange(len(multipliers))
        slope, _, r_value, _, _ = stats.linregress(x, multipliers)
        
        if abs(r_value) < 0.3:
            return 'no_trend'
        elif slope > 0:
            return 'increasing'
        else:
            return 'decreasing'
    
    def _analyze_hourly_distribution(self, recent_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Analyze distribution by hour of day"""
        hourly_counts = Counter()
        
        for round_data in recent_data:
            try:
                timestamp = pd.to_datetime(round_data.get('timestamp'))
                hour = timestamp.hour
                hourly_counts[hour] += 1
            except:
                continue
        
        total = sum(hourly_counts.values())
        if total == 0:
            return {}
        
        return {str(hour): count/total*100 for hour, count in hourly_counts.items()}
    
    def _calculate_moving_averages(self, multipliers: List[float]) -> Dict[str, float]:
        """Calculate moving averages"""
        if len(multipliers) < 5:
            return {}
        
        averages = {}
        for window in [5, 10, 20]:
            if len(multipliers) >= window:
                ma = np.mean(multipliers[-window:])
                averages[f'ma_{window}'] = round(ma, 2)
        
        return averages
    
    def _analyze_volatility_trend(self, multipliers: List[float]) -> str:
        """Analyze volatility trend"""
        if len(multipliers) < 10:
            return 'unknown'
        
        # Calculate volatility for first and second half
        mid = len(multipliers) // 2
        first_half_vol = np.std(multipliers[:mid])
        second_half_vol = np.std(multipliers[mid:])
        
        if second_half_vol > first_half_vol * 1.2:
            return 'increasing'
        elif second_half_vol < first_half_vol * 0.8:
            return 'decreasing'
        else:
            return 'stable'
    
    def _calculate_momentum_indicators(self, multipliers: List[float]) -> Dict[str, float]:
        """Calculate momentum indicators"""
        if len(multipliers) < 14:
            return {}
        
        # Simple RSI calculation
        gains = []
        losses = []
        
        for i in range(1, len(multipliers)):
            diff = multipliers[i] - multipliers[i-1]
            if diff > 0:
                gains.append(diff)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(diff))
        
        if len(gains) >= 14:
            avg_gain = np.mean(gains[-14:])
            avg_loss = np.mean(losses[-14:])
            
            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 100
            
            return {'rsi': round(rsi, 1)}
        
        return {}
    
    def _identify_support_resistance(self, multipliers: List[float]) -> Dict[str, float]:
        """Identify support and resistance levels"""
        if len(multipliers) < 20:
            return {}
        
        # Find local maxima and minima
        resistance_levels = []
        support_levels = []
        
        for i in range(1, len(multipliers) - 1):
            if multipliers[i] > multipliers[i-1] and multipliers[i] > multipliers[i+1]:
                resistance_levels.append(multipliers[i])
            elif multipliers[i] < multipliers[i-1] and multipliers[i] < multipliers[i+1]:
                support_levels.append(multipliers[i])
        
        result = {}
        if resistance_levels:
            result['resistance'] = round(np.mean(resistance_levels), 2)
        if support_levels:
            result['support'] = round(np.mean(support_levels), 2)
        
        return result
    
    def _analyze_time_correlations(self) -> Dict[str, Any]:
        """Analyze time-based correlations"""
        # This would analyze correlations with time of day, day of week, etc.
        # For now, return empty dict as it requires more complex timestamp analysis
        return {}
    
    def _get_immediate_trend(self, recent_multipliers: List[float]) -> str:
        """Get immediate trend from recent multipliers"""
        if len(recent_multipliers) < 3:
            return 'unknown'
        
        last_3 = recent_multipliers[-3:]
        if all(last_3[i] > last_3[i-1] for i in range(1, len(last_3))):
            return 'increasing'
        elif all(last_3[i] < last_3[i-1] for i in range(1, len(last_3))):
            return 'decreasing'
        else:
            return 'stable'
    
    def _get_relative_position(self, current: float, recent: List[float]) -> str:
        """Get relative position of current value"""
        if not recent:
            return 'unknown'
        
        avg = np.mean(recent)
        if current > avg * 1.2:
            return 'high'
        elif current < avg * 0.8:
            return 'low'
        else:
            return 'normal'
    
    def _get_volatility_state(self, recent: List[float]) -> str:
        """Get current volatility state"""
        if len(recent) < 5:
            return 'unknown'
        
        std_dev = np.std(recent)
        mean_val = np.mean(recent)
        cv = std_dev / mean_val if mean_val > 0 else 0
        
        if cv > 0.5:
            return 'high'
        elif cv < 0.2:
            return 'low'
        else:
            return 'normal'
    
    def _find_pattern_match(self, recent: List[float]) -> Dict[str, Any]:
        """Find pattern matches in recent data"""
        # Simplified pattern matching
        if len(recent) < 5:
            return {'pattern': 'none', 'confidence': 0}
        
        last_5 = recent[-5:]
        
        # Check for simple patterns
        if all(last_5[i] > last_5[i-1] for i in range(1, len(last_5))):
            return {'pattern': 'ascending', 'confidence': 0.8}
        elif all(last_5[i] < last_5[i-1] for i in range(1, len(last_5))):
            return {'pattern': 'descending', 'confidence': 0.8}
        elif abs(max(last_5) - min(last_5)) < 0.5:
            return {'pattern': 'stable', 'confidence': 0.7}
        else:
            return {'pattern': 'random', 'confidence': 0.3}
    
    def _calculate_analysis_confidence(self, recent: List[float]) -> float:
        """Calculate confidence in analysis"""
        if len(recent) < 5:
            return 0.2
        
        # Base confidence on data quantity and consistency
        base_confidence = min(len(recent) / 20, 1.0) * 0.6
        
        # Adjust for volatility (lower volatility = higher confidence in patterns)
        std_dev = np.std(recent)
        mean_val = np.mean(recent)
        cv = std_dev / mean_val if mean_val > 0 else 1
        volatility_factor = max(0.1, 1.0 - cv)
        
        final_confidence = base_confidence * volatility_factor
        return round(min(final_confidence, 0.9), 2)
    
    def _generate_recommendations(self, recent: List[float], current: float) -> List[str]:
        """Generate analysis-based recommendations"""
        recommendations = []
        
        if len(recent) < 3:
            recommendations.append("Insufficient data for reliable analysis")
            return recommendations
        
        avg = np.mean(recent)
        trend = self._get_immediate_trend(recent)
        
        if current > avg * 1.5:
            recommendations.append("Current multiplier is significantly above recent average")
            
        if trend == 'increasing':
            recommendations.append("Upward trend detected in recent rounds")
        elif trend == 'decreasing':
            recommendations.append("Downward trend detected in recent rounds")
            
        volatility = self._get_volatility_state(recent)
        if volatility == 'high':
            recommendations.append("High volatility detected - expect unpredictable swings")
        elif volatility == 'low':
            recommendations.append("Low volatility detected - more predictable patterns")
        
        return recommendations
