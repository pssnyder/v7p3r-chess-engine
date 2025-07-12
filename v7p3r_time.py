# v7p3r_time.py
"""
Time Management System for Chess Engine
Handles time allocation for moves in different time controls
"""

import os
import sys
import time
from typing import Optional, Dict, Any

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

class v7p3rTime:
    def __init__(self):
        self.start_time: Optional[float] = None
        self.allocated_time: Optional[float] = None
        self.max_time: Optional[float] = None
        self.emergency_time: Optional[float] = None
        
        # Game clock variables
        self.white_time: float = float('inf')  # White's remaining time in seconds
        self.black_time: float = float('inf')  # Black's remaining time in seconds
        self.increment: float = 0.0            # Time increment in seconds
        self.time_control_enabled: bool = False
        self.game_start_time: Optional[float] = None
        self.move_start_time: Optional[float] = None
        
        # Set default allocated time for testing
        self._test_allocated_time: float = 5.0
        
    def allocate_time(self, time_control: Dict[str, Any], board) -> float:
        """
        Allocate time for current move based on time control and position

        Args:
            time_control: Dictionary with time control parameters
            board: Current chess position

        Returns:
            Allocated time in seconds
        """
        # Handle fixed time per move
        if time_control.get('movetime'):
            allocated = time_control['movetime'] / 1000.0
        # Handle infinite time
        elif time_control.get('infinite'):
            allocated = float('inf')
        else:
            # Default for testing
            allocated = self._test_allocated_time
            
        self.allocated_time = allocated
        return allocated
        
    def get_allocated_move_time(self) -> float:
        """
        Get the currently allocated time for the move.
        Returns a default value if no time has been allocated.
        """
        if self.allocated_time is not None:
            return self.allocated_time
        return self._test_allocated_time
        
    def get_current_time(self) -> float:
        """Get current timestamp"""
        return time.time()
        
    def get_elapsed_time(self) -> float:
        """Get elapsed time since move start"""
        if self.move_start_time:
            return self.get_current_time() - self.move_start_time
        return 0.0

    def start_timer(self, allocated_time: float):
        """Start the timer for current move"""
        self.start_time = time.time()
        self.allocated_time = allocated_time
        self.max_time = allocated_time * 1.2  # 20% buffer for critical positions
        self.emergency_time = allocated_time * 0.1  # Emergency stop time

    def should_stop(self, depth: int = 0, nodes: int = 0) -> bool:
        """
        Check if search should stop based on time

        Args:
            depth: Current search depth
            nodes: Nodes searched so far

        Returns:
            True if search should stop
        """
        if self.start_time is None or self.allocated_time is None:
            return False

        elapsed = time.time() - self.start_time

        # Always stop if we exceed maximum time
        if self.max_time is not None and elapsed >= self.max_time:
            return True

        # Stop if we've used allocated time
        if elapsed >= self.allocated_time:
            return True

        # Don't stop too early (minimum search time)
        if elapsed < 0.05:  # At least 50ms
            return False

        # Depth-based stopping
        if depth >= 1:  # If we have at least one complete iteration
            # If we're close to time limit, don't start new iteration
            if elapsed >= self.allocated_time * 0.8:
                return True

        return False

    def time_remaining(self) -> float:
        """Get remaining allocated time"""
        if self.start_time is None or self.allocated_time is None:
            return float('inf')

        elapsed = time.time() - self.start_time
        return max(0, self.allocated_time - elapsed)

    def time_elapsed(self) -> float:
        """Get time elapsed since search started"""
        if self.start_time is None:
            return 0.0
        return time.time() - self.start_time

    def get_time_info(self) -> Dict[str, float]:
        """Get timing information for UCI info output"""
        return {
            'elapsed': self.time_elapsed(),
            'remaining': self.time_remaining(),
            'allocated': self.allocated_time or 0.0
        }

    def setup_game_clock(self, game_config: Dict[str, Any]):
        """Initialize game clock based on game configuration"""
        self.time_control_enabled = game_config.get('time_control', False)
        if self.time_control_enabled:
            game_time = game_config.get('game_time', 300)  # Default 5 minutes
            self.white_time = float(game_time)
            self.black_time = float(game_time)
            self.increment = float(game_config.get('time_increment', 0))
        else:
            self.white_time = float('inf')
            self.black_time = float('inf')
            self.increment = 0.0
            
        self.game_start_time = time.time()
        
    def start_move_timer(self, color: bool):
        """Start timing a move for the given color"""
        self.move_start_time = time.time()
        
    def end_move_timer(self, color: bool) -> float:
        """
        End timing a move and update remaining time
        Returns the move duration in seconds
        """
        if self.move_start_time is None:
            return 0.0
            
        move_duration = time.time() - self.move_start_time
        
        # Update remaining time if time control is enabled
        if self.time_control_enabled:
            if color:  # White
                self.white_time = max(0.0, self.white_time - move_duration + self.increment)
            else:  # Black
                self.black_time = max(0.0, self.black_time - move_duration + self.increment)
            
        self.move_start_time = None
        return move_duration
        
    def get_remaining_time(self, color: bool) -> float:
        """Get remaining time for the given color in seconds"""
        return self.white_time if color else self.black_time
        
    def is_time_up(self, color: bool) -> bool:
        """Check if a player has run out of time"""
        return self.time_control_enabled and self.get_remaining_time(color) <= 0.0
        
    def get_clock_state(self) -> Dict[str, float]:
        """Get current state of both clocks"""
        return {
            'white_time': self.white_time,
            'black_time': self.black_time,
            'increment': self.increment,
            'time_control': self.time_control_enabled,
            'game_duration': time.time() - self.game_start_time if self.game_start_time else 0.0
        }

    # Example usage and testing
    def test_time_manager(self):
        import chess

        tm = v7p3rTime()
        board = chess.Board()

        # Test different time controls
        time_controls = [
            {'wtime': 300000, 'btime': 300000, 'winc': 3000, 'binc': 3000},  # 5+3 blitz
            {'wtime': 180000, 'btime': 180000, 'winc': 0, 'binc': 0},        # 3+0 blitz
            {'wtime': 900000, 'btime': 900000, 'winc': 10000, 'binc': 10000}, # 15+10 rapid
            {'movetime': 5000},  # 5 seconds per move
            {'depth': 10},       # Fixed depth
            {'infinite': True}   # Infinite time
        ]

        print("Time Manager Test Results:")
        print("-" * 40)

        for i, tc in enumerate(time_controls):
            allocated = tm.allocate_time(tc, board)
            print(f"Time Control {i+1}: {tc}")
            print(f"Allocated Time: {allocated:.3f} seconds")
            print()
