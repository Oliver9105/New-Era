"""
Feature 4: App Analysis
Advanced game behavior analysis and pattern recognition
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
import logging
import json
import statistics
from collections import defaultdict, Counter
import math
import scipy.stats as stats
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

logger = logging.getLogger(__name__)

class GameAnalyzer:
    def __init__(self):
        self.active = False
        self.historical_data = []
        self.patterns = {
            'sequence_patterns': [],
            'timing_patterns': [],
            'volatility_patterns': [],
            'frequency_patterns': []
        }
        self.analysis_cache = {}
        self.trend_indicators = {}

    def initialize(self):
        """Initialize the game analyzer"""
        self.active = True
        self._generate_sample_data()  # For demonstration
        logger.info("Game Analyzer initialized")

    def is_active(self):
        return self.active

    def _generate_sample_data(self):
        """Generate sample historical data for analysis"""
        import random
        
        # Generate 1000 sample multipliers
        for i in range(1000):
            timestamp = datetime.utcnow() - timedelta(minutes=i)
            
            # Simulate realistic aviator multipliers
            if random.random() < 0.7:  # 70% chance of low multipliers
                multiplier = round(random.uniform(1.0, 3.0), 2)
            elif random.random() < 0.9:  # 20% chance of medium multipliers
                multiplier = round(random.uniform(3.0, 10.0), 2)
            else:  # 10% chance of high multipliers
                multiplier = round(random.uniform(10.0, 100.0), 2)
            
            self.historical_data.append({
                'timestamp': timestamp,
                'multiplier': multiplier,
                'round_id': 1000 + i,
                'players': random.randint(50, 500),
                'total_bet': random.uniform(1000, 50000)
            })
        
        # Sort by timestamp
        self.historical_data.sort(key=lambda x: x['timestamp'])

    def get_patterns(self) -> Dict:
        """Identify and return game patterns"""
        try:
            self._analyze_sequence_patterns()
            self._analyze_timing_patterns()
            self._analyze_volatility_patterns()
            self._analyze_frequency_patterns()
            
            return {
                'patterns': self.patterns,
                'pattern_count': sum(len(patterns) for patterns in self.patterns.values()),
                'last_analysis': datetime.utcnow().isoformat(),
                'confidence_scores': self._calculate_pattern_confidence()
            }
        except Exception as e:
            logger.error(f"Error analyzing patterns: {e}")
            return {'error': str(e)}

    def _analyze_sequence_patterns(self):
        """Analyze multiplier sequence patterns"""
        if len(self.historical_data) < 10:
            return

        multipliers = [d['multiplier'] for d in self.historical_data[-1000:]]
        
        # Find consecutive patterns
        consecutive_patterns = self._find_consecutive_patterns(multipliers)
        
        # Find recurring sequences
        recurring_sequences = self._find_recurring_sequences(multipliers)
        
        # Streak analysis
        streak_analysis = self._analyze_streaks(multipliers)
        
        self.patterns['sequence_patterns'] = {
            'consecutive_patterns': consecutive_patterns,
            'recurring_sequences': recurring_sequences,
            'streak_analysis': streak_analysis
        }

    def _find_consecutive_patterns(self, multipliers: List[float]) -> List[Dict]:
        """Find patterns in consecutive multipliers"""
        patterns = []
        
        # Look for increasing/decreasing trends
        for i in range(len(multipliers) - 4):
            sequence = multipliers[i:i+5]
            
            # Check for increasing trend
            if all(sequence[j] < sequence[j+1] for j in range(4)):
                patterns.append({
                    'type': 'increasing_trend',
                    'sequence': sequence,
                    'start_index': i,
                    'confidence': 0.8
                })
            
            # Check for decreasing trend
            elif all(sequence[j] > sequence[j+1] for j in range(4)):
                patterns.append({
                    'type': 'decreasing_trend',
                    'sequence': sequence,
                    'start_index': i,
                    'confidence': 0.8
                })
            
            # Check for alternating pattern
            elif self._is_alternating(sequence):
                patterns.append({
                    'type': 'alternating_pattern',
                    'sequence': sequence,
                    'start_index': i,
                    'confidence': 0.7
                })

        return patterns[-20:]  # Return last 20 patterns

    def _is_alternating(self, sequence: List[float]) -> bool:
        """Check if sequence has alternating high/low pattern"""
        avg = sum(sequence) / len(sequence)
        categories = ['high' if x > avg else 'low' for x in sequence]
        
        # Check for alternating pattern
        alternating = True
        for i in range(len(categories) - 1):
            if categories[i] == categories[i + 1]:
                alternating = False
                break
        
        return alternating

    def _find_recurring_sequences(self, multipliers: List[float]) -> List[Dict]:
        """Find recurring sequences of multipliers"""
        sequences = {}
        sequence_length = 3
        
        for i in range(len(multipliers) - sequence_length + 1):
            sequence = tuple(multipliers[i:i + sequence_length])
            rounded_sequence = tuple(round(x, 1) for x in sequence)
            
            if rounded_sequence in sequences:
                sequences[rounded_sequence]['count'] += 1
                sequences[rounded_sequence]['positions'].append(i)
            else:
                sequences[rounded_sequence] = {
                    'count': 1,
                    'positions': [i],
                    'sequence': sequence
                }

        # Return sequences that appear more than once
        recurring = [
            {
                'sequence': list(data['sequence']),
                'count': data['count'],
                'frequency': data['count'] / len(multipliers),
                'positions': data['positions']
            }
            for seq, data in sequences.items() if data['count'] > 1
        ]
        
        return sorted(recurring, key=lambda x: x['count'], reverse=True)[:10]

    def _analyze_streaks(self, multipliers: List[float]) -> Dict:
        """Analyze winning and losing streaks"""
        # Define win/loss based on multiplier thresholds
        high_threshold = 2.0
        low_threshold = 1.5
        
        streaks = {
            'high_streaks': [],
            'low_streaks': [],
            'average_high_streak': 0,
            'average_low_streak': 0,
            'max_high_streak': 0,
            'max_low_streak': 0
        }
        
        current_high_streak = 0
        current_low_streak = 0
        high_streaks = []
        low_streaks = []
        
        for multiplier in multipliers:
            if multiplier >= high_threshold:
                current_high_streak += 1
                if current_low_streak > 0:
                    low_streaks.append(current_low_streak)
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

        streaks['high_streaks'] = high_streaks
        streaks['low_streaks'] = low_streaks
        streaks['average_high_streak'] = statistics.mean(high_streaks) if high_streaks else 0
        streaks['average_low_streak'] = statistics.mean(low_streaks) if low_streaks else 0
        streaks['max_high_streak'] = max(high_streaks) if high_streaks else 0
        streaks['max_low_streak'] = max(low_streaks) if low_streaks else 0
        
        return streaks

    def _analyze_timing_patterns(self):
        """Analyze timing-based patterns"""
        if len(self.historical_data) < 10:
            return

        # Group by hour of day
        hourly_analysis = defaultdict(list)
        for data in self.historical_data:
            hour = data['timestamp'].hour
            hourly_analysis[hour].append(data['multiplier'])

        # Calculate hourly statistics
        hourly_stats = {}
        for hour, multipliers in hourly_analysis.items():
            hourly_stats[hour] = {
                'average_multiplier': statistics.mean(multipliers),
                'median_multiplier': statistics.median(multipliers),
                'volatility': statistics.stdev(multipliers) if len(multipliers) > 1 else 0,
                'high_count': len([m for m in multipliers if m > 5.0]),
                'total_rounds': len(multipliers)
            }

        # Find time-based patterns
        peak_hours = sorted(hourly_stats.items(), key=lambda x: x[1]['average_multiplier'], reverse=True)[:3]
        volatile_hours = sorted(hourly_stats.items(), key=lambda x: x[1]['volatility'], reverse=True)[:3]

        self.patterns['timing_patterns'] = {
            'hourly_stats': hourly_stats,
            'peak_hours': [(hour, stats) for hour, stats in peak_hours],
            'volatile_hours': [(hour, stats) for hour, stats in volatile_hours],
            'optimal_play_hours': [hour for hour, _ in peak_hours]
        }

    def _analyze_volatility_patterns(self):
        """Analyze volatility and variance patterns"""
        if len(self.historical_data) < 50:
            return

        multipliers = [d['multiplier'] for d in self.historical_data]
        
        # Calculate rolling volatility
        window_size = 20
        rolling_volatility = []
        rolling_means = []
        
        for i in range(window_size, len(multipliers)):
            window = multipliers[i-window_size:i]
            volatility = statistics.stdev(window)
            mean_mult = statistics.mean(window)
            
            rolling_volatility.append(volatility)
            rolling_means.append(mean_mult)

        # Identify volatility clusters
        volatility_clusters = self._identify_volatility_clusters(rolling_volatility)
        
        # Calculate risk metrics
        risk_metrics = self._calculate_risk_metrics(multipliers)
        
        self.patterns['volatility_patterns'] = {
            'rolling_volatility': rolling_volatility[-100:],  # Last 100 values
            'volatility_clusters': volatility_clusters,
            'risk_metrics': risk_metrics,
            'volatility_trend': self._calculate_volatility_trend(rolling_volatility)
        }

    def _identify_volatility_clusters(self, volatility_data: List[float]) -> List[Dict]:
        """Identify clusters of high/low volatility periods"""
        if len(volatility_data) < 10:
            return []

        # Use K-means clustering
        volatility_array = np.array(volatility_data).reshape(-1, 1)
        scaler = StandardScaler()
        volatility_scaled = scaler.fit_transform(volatility_array)
        
        kmeans = KMeans(n_clusters=3, random_state=42)
        clusters = kmeans.fit_predict(volatility_scaled)
        
        # Identify cluster characteristics
        cluster_info = []
        for i in range(3):
            cluster_indices = np.where(clusters == i)[0]
            cluster_volatilities = [volatility_data[idx] for idx in cluster_indices]
            
            cluster_info.append({
                'cluster_id': i,
                'average_volatility': statistics.mean(cluster_volatilities),
                'size': len(cluster_indices),
                'type': 'high' if statistics.mean(cluster_volatilities) > np.mean(volatility_data) else 'low'
            })
        
        return sorted(cluster_info, key=lambda x: x['average_volatility'], reverse=True)

    def _calculate_risk_metrics(self, multipliers: List[float]) -> Dict:
        """Calculate various risk and performance metrics"""
        if len(multipliers) < 2:
            return {}

        returns = [(multipliers[i] - multipliers[i-1]) / multipliers[i-1] for i in range(1, len(multipliers))]
        
        return {
            'value_at_risk_95': np.percentile(returns, 5),  # 95% VaR
            'expected_shortfall': np.mean([r for r in returns if r <= np.percentile(returns, 5)]),
            'sharpe_ratio': np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0,
            'maximum_drawdown': self._calculate_max_drawdown(multipliers),
            'volatility': np.std(returns),
            'skewness': stats.skew(returns),
            'kurtosis': stats.kurtosis(returns)
        }

    def _calculate_max_drawdown(self, multipliers: List[float]) -> float:
        """Calculate maximum drawdown"""
        peak = multipliers[0]
        max_drawdown = 0
        
        for value in multipliers[1:]:
            if value > peak:
                peak = value
            drawdown = (peak - value) / peak
            max_drawdown = max(max_drawdown, drawdown)
        
        return max_drawdown

    def _calculate_volatility_trend(self, volatility_data: List[float]) -> str:
        """Calculate overall volatility trend"""
        if len(volatility_data) < 10:
            return 'insufficient_data'
        
        recent = volatility_data[-10:]
        older = volatility_data[-20:-10] if len(volatility_data) >= 20 else volatility_data[:-10]
        
        if not older:
            return 'insufficient_data'
        
        recent_avg = statistics.mean(recent)
        older_avg = statistics.mean(older)
        
        if recent_avg > older_avg * 1.1:
            return 'increasing'
        elif recent_avg < older_avg * 0.9:
            return 'decreasing'
        else:
            return 'stable'

    def _analyze_frequency_patterns(self):
        """Analyze frequency patterns of different multiplier ranges"""
        if len(self.historical_data) < 10:
            return

        multipliers = [d['multiplier'] for d in self.historical_data]
        
        # Define ranges
        ranges = {
            'very_low': (1.0, 1.5),
            'low': (1.5, 2.0),
            'medium': (2.0, 5.0),
            'high': (5.0, 10.0),
            'very_high': (10.0, float('inf'))
        }
        
        frequency_analysis = {}
        for range_name, (min_val, max_val) in ranges.items():
            count = len([m for m in multipliers if min_val <= m < max_val])
            frequency_analysis[range_name] = {
                'count': count,
                'frequency': count / len(multipliers),
                'range': f"{min_val}-{max_val if max_val != float('inf') else '∞'}"
            }

        # Calculate distribution statistics
        distribution_stats = {
            'mean': statistics.mean(multipliers),
            'median': statistics.median(multipliers),
            'mode': statistics.mode(multipliers) if len(set(multipliers)) < len(multipliers) else None,
            'std_dev': statistics.stdev(multipliers) if len(multipliers) > 1 else 0,
            'min': min(multipliers),
            'max': max(multipliers),
            'percentiles': {
                '25th': np.percentile(multipliers, 25),
                '50th': np.percentile(multipliers, 50),
                '75th': np.percentile(multipliers, 75),
                '90th': np.percentile(multipliers, 90),
                '95th': np.percentile(multipliers, 95)
            }
        }

        self.patterns['frequency_patterns'] = {
            'range_analysis': frequency_analysis,
            'distribution_stats': distribution_stats,
            'outliers': self._identify_outliers(multipliers)
        }

    def _identify_outliers(self, multipliers: List[float]) -> List[Dict]:
        """Identify statistical outliers in multiplier data"""
        if len(multipliers) < 10:
            return []

        Q1 = np.percentile(multipliers, 25)
        Q3 = np.percentile(multipliers, 75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = []
        for i, mult in enumerate(multipliers):
            if mult < lower_bound or mult > upper_bound:
                outliers.append({
                    'index': i,
                    'value': mult,
                    'type': 'low' if mult < lower_bound else 'high',
                    'deviation': abs(mult - statistics.mean(multipliers))
                })
        
        return sorted(outliers, key=lambda x: x['deviation'], reverse=True)[:20]

    def _calculate_pattern_confidence(self) -> Dict:
        """Calculate confidence scores for identified patterns"""
        confidence_scores = {}
        
        # Sequence pattern confidence
        if self.patterns['sequence_patterns']:
            seq_patterns = self.patterns['sequence_patterns']
            if isinstance(seq_patterns, dict):
                total_patterns = (
                    len(seq_patterns.get('consecutive_patterns', [])) +
                    len(seq_patterns.get('recurring_sequences', []))
                )
                confidence_scores['sequence_patterns'] = min(0.9, total_patterns / 10)

        # Timing pattern confidence
        if self.patterns['timing_patterns']:
            timing_patterns = self.patterns['timing_patterns']
            if isinstance(timing_patterns, dict) and 'hourly_stats' in timing_patterns:
                hourly_data = timing_patterns['hourly_stats']
                variance_scores = [stats['volatility'] for stats in hourly_data.values()]
                avg_variance = statistics.mean(variance_scores) if variance_scores else 0
                confidence_scores['timing_patterns'] = max(0.1, 1 - (avg_variance / 10))

        # Volatility pattern confidence
        if self.patterns['volatility_patterns']:
            vol_patterns = self.patterns['volatility_patterns']
            if isinstance(vol_patterns, dict) and 'risk_metrics' in vol_patterns:
                sharpe_ratio = vol_patterns['risk_metrics'].get('sharpe_ratio', 0)
                confidence_scores['volatility_patterns'] = max(0.1, min(0.9, abs(sharpe_ratio)))

        # Frequency pattern confidence
        if self.patterns['frequency_patterns']:
            freq_patterns = self.patterns['frequency_patterns']
            if isinstance(freq_patterns, dict) and 'range_analysis' in freq_patterns:
                range_data = freq_patterns['range_analysis']
                entropy = -sum(
                    freq['frequency'] * math.log(freq['frequency']) 
                    for freq in range_data.values() 
                    if freq['frequency'] > 0
                )
                confidence_scores['frequency_patterns'] = max(0.1, min(0.9, entropy / 2))

        return confidence_scores

    def analyze_trends(self, hours: int = 24) -> Dict:
        """Analyze trends over specified time period"""
        try:
            cutoff_time = datetime.utcnow() - timedelta(hours=hours)
            recent_data = [d for d in self.historical_data if d['timestamp'] > cutoff_time]
            
            if len(recent_data) < 5:
                return {'error': 'Insufficient recent data'}

            trends = {
                'time_period': f"{hours} hours",
                'data_points': len(recent_data),
                'multiplier_trend': self._calculate_multiplier_trend(recent_data),
                'volume_trend': self._calculate_volume_trend(recent_data),
                'player_trend': self._calculate_player_trend(recent_data),
                'volatility_trend': self._calculate_period_volatility_trend(recent_data)
            }

            return {
                'trends': trends,
                'analysis_timestamp': datetime.utcnow().isoformat(),
                'trend_strength': self._calculate_trend_strength(trends)
            }
        except Exception as e:
            logger.error(f"Error analyzing trends: {e}")
            return {'error': str(e)}

    def _calculate_multiplier_trend(self, data: List[Dict]) -> Dict:
        """Calculate multiplier trend direction and strength"""
        multipliers = [d['multiplier'] for d in data]
        
        # Linear regression to find trend
        x = np.arange(len(multipliers))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, multipliers)
        
        return {
            'direction': 'increasing' if slope > 0 else 'decreasing',
            'slope': slope,
            'correlation': r_value,
            'significance': p_value,
            'strength': abs(r_value)
        }

    def _calculate_volume_trend(self, data: List[Dict]) -> Dict:
        """Calculate betting volume trend"""
        volumes = [d.get('total_bet', 0) for d in data]
        
        if not any(volumes):
            return {'error': 'No volume data available'}
        
        x = np.arange(len(volumes))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, volumes)
        
        return {
            'direction': 'increasing' if slope > 0 else 'decreasing',
            'slope': slope,
            'correlation': r_value,
            'average_volume': statistics.mean(volumes)
        }

    def _calculate_player_trend(self, data: List[Dict]) -> Dict:
        """Calculate player count trend"""
        player_counts = [d.get('players', 0) for d in data]
        
        if not any(player_counts):
            return {'error': 'No player data available'}
        
        x = np.arange(len(player_counts))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, player_counts)
        
        return {
            'direction': 'increasing' if slope > 0 else 'decreasing',
            'slope': slope,
            'correlation': r_value,
            'average_players': statistics.mean(player_counts)
        }

    def _calculate_period_volatility_trend(self, data: List[Dict]) -> Dict:
        """Calculate volatility trend for the period"""
        multipliers = [d['multiplier'] for d in data]
        
        if len(multipliers) < 10:
            return {'error': 'Insufficient data for volatility calculation'}
        
        # Calculate rolling volatility
        window_size = min(10, len(multipliers) // 2)
        volatilities = []
        
        for i in range(window_size, len(multipliers)):
            window = multipliers[i-window_size:i]
            vol = statistics.stdev(window)
            volatilities.append(vol)
        
        if len(volatilities) < 3:
            return {'error': 'Insufficient volatility data'}
        
        x = np.arange(len(volatilities))
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, volatilities)
        
        return {
            'direction': 'increasing' if slope > 0 else 'decreasing',
            'slope': slope,
            'correlation': r_value,
            'current_volatility': volatilities[-1] if volatilities else 0
        }

    def _calculate_trend_strength(self, trends: Dict) -> float:
        """Calculate overall trend strength"""
        strengths = []
        
        for trend_name, trend_data in trends.items():
            if isinstance(trend_data, dict) and 'correlation' in trend_data:
                strengths.append(abs(trend_data['correlation']))
        
        return statistics.mean(strengths) if strengths else 0

    def analyze_realtime(self, realtime_data: Dict) -> Dict:
        """Analyze real-time data for immediate insights"""
        try:
            current_multiplier = realtime_data.get('current_multiplier', 0)
            
            # Compare with recent historical data
            recent_multipliers = [d['multiplier'] for d in self.historical_data[-50:]]
            
            analysis = {
                'current_vs_average': current_multiplier / statistics.mean(recent_multipliers) if recent_multipliers else 1,
                'percentile_rank': self._calculate_percentile_rank(current_multiplier, recent_multipliers),
                'volatility_signal': self._get_volatility_signal(current_multiplier, recent_multipliers),
                'trend_continuation': self._assess_trend_continuation(current_multiplier, recent_multipliers),
                'risk_level': self._assess_current_risk_level(current_multiplier, recent_multipliers)
            }
            
            return analysis
        except Exception as e:
            logger.error(f"Error in real-time analysis: {e}")
            return {'error': str(e)}

    def _calculate_percentile_rank(self, value: float, data: List[float]) -> float:
        """Calculate percentile rank of value in data"""
        if not data:
            return 50.0
        
        sorted_data = sorted(data)
        rank = sum(1 for x in sorted_data if x <= value)
        return (rank / len(sorted_data)) * 100

    def _get_volatility_signal(self, current: float, recent: List[float]) -> str:
        """Get volatility signal for current multiplier"""
        if len(recent) < 5:
            return 'unknown'
        
        recent_volatility = statistics.stdev(recent[-10:]) if len(recent) >= 10 else statistics.stdev(recent)
        historical_volatility = statistics.stdev(recent)
        
        if recent_volatility > historical_volatility * 1.2:
            return 'high_volatility'
        elif recent_volatility < historical_volatility * 0.8:
            return 'low_volatility'
        else:
            return 'normal_volatility'

    def _assess_trend_continuation(self, current: float, recent: List[float]) -> str:
        """Assess if current value continues recent trend"""
        if len(recent) < 3:
            return 'unknown'
        
        last_three = recent[-3:]
        
        # Check if trend is increasing
        if all(last_three[i] < last_three[i+1] for i in range(len(last_three)-1)):
            return 'continues_uptrend' if current > last_three[-1] else 'breaks_uptrend'
        
        # Check if trend is decreasing
        elif all(last_three[i] > last_three[i+1] for i in range(len(last_three)-1)):
            return 'continues_downtrend' if current < last_three[-1] else 'breaks_downtrend'
        
        return 'no_clear_trend'

    def _assess_current_risk_level(self, current: float, recent: List[float]) -> str:
        """Assess current risk level based on multiplier"""
        if not recent:
            return 'unknown'
        
        avg = statistics.mean(recent)
        std = statistics.stdev(recent) if len(recent) > 1 else 0
        
        if current > avg + 2 * std:
            return 'very_high'
        elif current > avg + std:
            return 'high'
        elif current < avg - 2 * std:
            return 'very_low'
        elif current < avg - std:
            return 'low'
        else:
            return 'normal'
