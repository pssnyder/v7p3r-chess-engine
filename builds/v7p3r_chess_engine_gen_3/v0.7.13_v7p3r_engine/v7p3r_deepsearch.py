# v7p3r_deepsearch.py

"""v7p3r Deep Search Module
This module handles iterative deepening and dynamic depth control for the v7p3r chess engine.
It works in conjunction with the time manager to determine optimal search depths.
"""

import chess
from typing import Optional, Dict, List
from v7p3r_config import v7p3rConfig
from v7p3r_time import v7p3rTime

class v7p3rDeepSearch:
    def __init__(self, time_manager: v7p3rTime, engine_config: Optional[Dict] = None):
        """Initialize the deep search module"""
        self.time_manager = time_manager
        self.config = engine_config or v7p3rConfig().get_engine_config()
        
        # Configuration settings
        self.max_depth = self.config.get('max_depth', 8)
        self.initial_depth = self.config.get('depth', 4)
        self.iterative_deepening_enabled = self.config.get('use_iterative_deepening', True)
        
        # Historical tracking
        self.historical_scores = []
        self.best_moves_history = []
        
    def should_increase_depth(self, current_depth: int, best_score: float, start_time: float) -> bool:
        """Determine if search should continue to next depth"""
        if not self.iterative_deepening_enabled:
            return False
            
        # Don't exceed max depth
        if current_depth >= self.max_depth:
            return False
            
        # Check time constraints
        elapsed_time = self.time_manager.time_elapsed()
        remaining_time = self.time_manager.time_remaining()
        
        # Stop if we've used 80% of allocated time
        if remaining_time <= (self.time_manager.allocated_time or 0) * 0.2:
            return False
            
        # If score is very good, no need to search deeper
        if abs(best_score) > 5000:  # Near checkmate
            return False
            
        # If score hasn't improved significantly in last few iterations, maybe stop
        if len(self.historical_scores) >= 3:
            recent_scores = self.historical_scores[-3:]
            if all(abs(s - recent_scores[0]) < 50 for s in recent_scores):
                return False
                
        return True
        
    def get_next_depth(self, current_depth: int) -> int:
        """Calculate the next depth to search"""
        if current_depth < self.initial_depth:
            return min(current_depth + 2, self.initial_depth)
        return current_depth + 1
        
    def update_history(self, depth: int, score: float, best_move: chess.Move):
        """Update historical data for depth control"""
        self.historical_scores.append(score)
        if len(self.historical_scores) > 10:
            self.historical_scores.pop(0)
            
        self.best_moves_history.append(str(best_move))
        if len(self.best_moves_history) > 5:
            self.best_moves_history.pop(0)
            
    def get_starting_depth(self) -> int:
        """Get initial search depth based on game state and time"""
        if not self.iterative_deepening_enabled:
            return self.initial_depth
            
        # Start with shallow depth
        return min(3, self.initial_depth)
