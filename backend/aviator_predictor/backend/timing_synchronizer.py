#!/usr/bin/env python3
"""
Timing Synchronization Module
Fixes the issue where predictions don't synchronize with real game timing
"""

import time
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

@dataclass
class GameRoundInfo:
    """Information about a game round"""
    round_id: str
    start_time: float
    estimated_end_time: float
    multiplier: Optional[float] = None
    is_active: bool = True

class TimingSynchronizer:
    """
    Synchronizes predictions with actual game round timing
    """
    
    def __init__(self):
        self.current_round: Optional[GameRoundInfo] = None
        self.last_round_end: float = 0
        self.average_round_duration: float = 30.0  # Default 30 seconds
        self.round_durations: list = []
        self.prediction_window_start: float = 5.0  # Start predictions 5 seconds before round
        self.prediction_window_end: float = 2.0   # Stop predictions 2 seconds after round starts
        
    def update_round_info(self, round_data: Dict[str, Any]) -> None:
        """
        Update current round information from real data
        
        Args:
            round_data: Dictionary containing round information
        """
        try:
            current_time = time.time()
            
            # Extract round information
            round_id = str(round_data.get('round_id', round_data.get('id', f'round_{int(current_time)}')))
            
            # If this is a new round
            if not self.current_round or self.current_round.round_id != round_id:
                
                # Record the end of the previous round
                if self.current_round and self.current_round.is_active:
                    duration = current_time - self.current_round.start_time
                    self.round_durations.append(duration)
                    self.last_round_end = current_time
                    
                    # Update average duration (keep last 20 rounds)
                    if len(self.round_durations) > 20:
                        self.round_durations = self.round_durations[-20:]
                    self.average_round_duration = sum(self.round_durations) / len(self.round_durations)
                    
                    logger.info(f"Round {self.current_round.round_id} ended. Duration: {duration:.1f}s, Avg: {self.average_round_duration:.1f}s")
                
                # Start new round
                estimated_duration = self.average_round_duration
                self.current_round = GameRoundInfo(
                    round_id=round_id,
                    start_time=current_time,
                    estimated_end_time=current_time + estimated_duration,
                    is_active=True
                )
                
                logger.info(f"New round started: {round_id}, estimated duration: {estimated_duration:.1f}s")
            
            # Update multiplier if the round is finished
            if 'multiplier' in round_data and round_data['multiplier']:
                if self.current_round:
                    self.current_round.multiplier = float(round_data['multiplier'])
                    self.current_round.is_active = False
                    logger.info(f"Round {round_id} finished with multiplier: {round_data['multiplier']}x")
                    
        except Exception as e:
            logger.error(f"Error updating round info: {e}")
    
    def should_make_prediction(self) -> bool:
        """
        Determine if we should make a prediction based on timing
        
        Returns:
            True if it's the right time to make a prediction
        """
        current_time = time.time()
        
        # If no round information, allow predictions (fallback)
        if not self.current_round:
            return True
        
        # Calculate time since round start
        time_since_start = current_time - self.current_round.start_time
        time_until_estimated_end = self.current_round.estimated_end_time - current_time
        
        # Prediction windows:
        # 1. Before round starts (preparation time)
        if time_since_start < 0 and abs(time_since_start) <= self.prediction_window_start:
            return True
        
        # 2. Early in the round (when bets are still being placed)
        if 0 <= time_since_start <= self.prediction_window_end:
            return True
        
        # 3. If round has been going much longer than expected, allow predictions for next round
        if time_since_start > self.average_round_duration * 1.5:
            return True
        
        return False
    
    def get_prediction_context(self) -> Dict[str, Any]:
        """
        Get context information for making better predictions
        
        Returns:
            Dictionary with timing context
        """
        current_time = time.time()
        
        if not self.current_round:
            return {
                'timing_available': False,
                'message': 'No round timing information available'
            }
        
        time_since_start = current_time - self.current_round.start_time
        time_until_estimated_end = self.current_round.estimated_end_time - current_time
        
        return {
            'timing_available': True,
            'current_round_id': self.current_round.round_id,
            'round_start_time': self.current_round.start_time,
            'time_since_start': time_since_start,
            'time_until_estimated_end': time_until_estimated_end,
            'average_round_duration': self.average_round_duration,
            'is_round_active': self.current_round.is_active,
            'should_predict': self.should_make_prediction(),
            'prediction_reason': self._get_prediction_reason()
        }
    
    def _get_prediction_reason(self) -> str:
        """Get reason for prediction timing decision"""
        if not self.should_make_prediction():
            return "Round in progress, waiting for better timing"
        
        current_time = time.time()
        if not self.current_round:
            return "Making prediction (no timing data)"
        
        time_since_start = current_time - self.current_round.start_time
        
        if time_since_start < 0:
            return f"Pre-round prediction (round starts in {abs(time_since_start):.1f}s)"
        elif time_since_start <= self.prediction_window_end:
            return f"Early round prediction ({time_since_start:.1f}s into round)"
        else:
            return "Making prediction for next round"
    
    def wait_for_next_prediction_window(self) -> float:
        """
        Calculate how long to wait for the next prediction window
        
        Returns:
            Seconds to wait (0 if should predict now)
        """
        if self.should_make_prediction():
            return 0
        
        current_time = time.time()
        if not self.current_round:
            return 0
        
        # Wait until the next round is expected to start
        next_round_start = self.current_round.estimated_end_time + 5  # 5 second buffer
        wait_time = max(0, next_round_start - current_time)
        
        return wait_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get synchronization statistics"""
        return {
            'rounds_tracked': len(self.round_durations),
            'average_duration': self.average_round_duration,
            'current_round': self.current_round.round_id if self.current_round else None,
            'current_round_active': self.current_round.is_active if self.current_round else False,
            'prediction_window_start': self.prediction_window_start,
            'prediction_window_end': self.prediction_window_end
        }