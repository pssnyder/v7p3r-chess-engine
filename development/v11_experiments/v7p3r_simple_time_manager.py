#!/usr/bin/env python3
"""
V7P3R v11.1 Simplified Time Management
Simplified, reliable time allocation for immediate performance recovery
Author: Pat Snyder
"""

import chess
import time
from typing import Tuple


class V7P3RSimpleTimeManager:
    """Simplified time management for reliable performance"""
    
    def __init__(self, base_time: float = 300.0, increment: float = 3.0):
        """
        Initialize simplified time manager
        
        Args:
            base_time: Total time available (seconds) 
            increment: Time increment per move (seconds)
        """
        self.base_time = base_time
        self.increment = increment
        self.moves_played = 0
        
        # Simple, conservative settings
        self.target_time_percentage = 0.70  # Use 70% of available time
        self.max_time_percentage = 0.85     # Hard limit at 85%
        self.emergency_threshold = 30.0     # Emergency mode below 30 seconds
        
        # Simple depth settings
        self.default_depth = 6
        self.emergency_depth = 4
        
    def update_time_info(self, time_remaining: float, moves_played: int):
        """Update current time information"""
        self.moves_played = moves_played
    
    def calculate_time_allocation(self, board: chess.Board, time_remaining: float) -> Tuple[float, int]:
        """
        Calculate simple, reliable time allocation
        
        Args:
            board: Current chess position (unused in simple version)
            time_remaining: Time remaining (seconds)
            
        Returns:
            Tuple of (time_to_use, target_depth)
        """
        # Emergency time management
        if time_remaining < self.emergency_threshold:
            return self._emergency_allocation(time_remaining)
        
        # Estimate remaining moves (simple heuristic)
        estimated_moves_left = max(20, 60 - self.moves_played)
        
        # Calculate base allocation per move
        available_time = time_remaining + (self.increment * estimated_moves_left)
        time_per_move = available_time / estimated_moves_left
        
        # Apply target percentage
        allocated_time = time_per_move * self.target_time_percentage
        
        # Add increment to allocation
        allocated_time += self.increment * 0.8  # Use 80% of increment
        
        # Apply bounds
        max_time = time_remaining * self.max_time_percentage
        allocated_time = min(allocated_time, max_time)
        allocated_time = max(0.5, allocated_time)  # Minimum 0.5 seconds
        
        # Simple depth calculation
        if allocated_time < 1.0:
            target_depth = 4
        elif allocated_time < 3.0:
            target_depth = 5
        elif allocated_time < 8.0:
            target_depth = 6
        else:
            target_depth = 7
            
        return allocated_time, target_depth
    
    def _emergency_allocation(self, time_remaining: float) -> Tuple[float, int]:
        """Emergency time allocation for low time situations"""
        # Use minimal time with reduced depth
        allocated_time = min(time_remaining * 0.3, 2.0)
        allocated_time = max(0.1, allocated_time)
        
        return allocated_time, self.emergency_depth
    
    def get_stats(self) -> dict:
        """Get simple statistics"""
        return {
            'moves_played': self.moves_played,
            'time_allocation_mode': 'simple'
        }