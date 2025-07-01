# v7p3r_engine/v7p3r_time.py
"""
Time Management System for Chess Engine
Handles time allocation for moves in different time controls
"""

import os
import sys
import time
import datetime
import logging
from typing import Optional, Dict, Any

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)
def resource_path(relative_path):
    """Get absolute path to resource, works for dev and for PyInstaller"""
    base = getattr(sys, '_MEIPASS', None)
    if base:
        return os.path.join(base, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)
# =====================================
# ========== LOGGING SETUP ============
def get_timestamp():
    return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

# Create logging directory relative to project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
log_dir = os.path.join(project_root, 'logging')
if not os.path.exists(log_dir):
    os.makedirs(log_dir, exist_ok=True)

# Setup individual logger for this file
timestamp = get_timestamp()
log_filename = f"v7p3r_time_{timestamp}.log"
log_file_path = os.path.join(log_dir, log_filename)

v7p3r_time_logger = logging.getLogger(f"v7p3r_time_{timestamp}")
v7p3r_time_logger.setLevel(logging.DEBUG)

if not v7p3r_time_logger.handlers:
    from logging.handlers import RotatingFileHandler
    file_handler = RotatingFileHandler(
        log_file_path,
        maxBytes=10*1024*1024,
        backupCount=3,
        delay=True
    )
    formatter = logging.Formatter(
        '%(asctime)s | %(funcName)-15s | %(message)s',
        datefmt='%H:%M:%S'
    )
    file_handler.setFormatter(formatter)
    v7p3r_time_logger.addHandler(file_handler)
    v7p3r_time_logger.propagate = False

class v7p3rTime:
    def __init__(self):
        self.start_time = None
        self.allocated_time = None
        self.max_time = None
        self.emergency_time = None

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
            return time_control['movetime'] / 1000.0

        # Handle infinite time
        if time_control.get('infinite'):
            return float('inf')

        # Handle depth-based search
        if time_control.get('depth'):
            return float('inf')  # No time limit for depth search

        # Get time remaining for current side
        if board.turn:  # White to move
            remaining_time = time_control.get('wtime', 60000)  # Default 1 minute
            increment = time_control.get('winc', 0)
        else:  # Black to move
            remaining_time = time_control.get('btime', 60000)
            increment = time_control.get('binc', 0)

        # Convert milliseconds to seconds
        remaining_time = remaining_time / 1000.0
        increment = increment / 1000.0

        # Get moves to go (default to 30 if not specified)
        moves_to_go = time_control.get('movestogo', 30)

        # Calculate base time allocation
        if moves_to_go:
            # Tournament time control with moves to go
            base_time = remaining_time / max(moves_to_go, 1)
        else:
            # Sudden death or increment-based
            # Use more conservative approach for sudden death
            base_time = remaining_time / 40  # Assume 40 moves left

        # Add increment (but not full increment to be safe)
        allocated_time = base_time + (increment * 0.8)

        # Apply position-based time modifiers
        time_multiplier = self.get_position_time_multiplier(board)
        allocated_time *= time_multiplier

        # Safety limits
        min_time = 0.1  # Minimum 100ms
        max_time = min(
            remaining_time * 0.1,  # Never use more than 10% of remaining time
            remaining_time - 1.0   # Always leave at least 1 second
        )

        allocated_time = max(min_time, min(allocated_time, max_time))

        # Emergency time handling
        if remaining_time < 10.0:  # Less than 10 seconds
            allocated_time = min(allocated_time, remaining_time * 0.05)

        return allocated_time

    def get_position_time_multiplier(self, board) -> float:
        """
        Adjust time allocation based on position characteristics

        Args:
            board: Current chess position

        Returns:
            Time multiplier (1.0 = normal, >1.0 = use more time, <1.0 = use less time)
        """
        multiplier = 1.0

        # Check if position is complex (many legal moves)
        legal_moves = len(list(board.legal_moves))
        if legal_moves > 35:
            multiplier *= 1.3  # Complex position, think longer
        elif legal_moves < 10:
            multiplier *= 0.7  # Simple position, think less

        # Check if in check
        if board.is_check():
            multiplier *= 1.4  # Critical position, think longer

        # Opening/middlegame/endgame considerations
        piece_count = len(board.piece_map())
        if piece_count > 20:  # Opening/early middlegame
            multiplier *= 0.8  # Don't overthink opening
        elif piece_count < 10:  # Endgame
            multiplier *= 1.2  # Endgame precision important

        # Limit multiplier range
        return max(0.3, min(multiplier, 2.5))

    def get_dynamic_depth(self, depth: int, max_depth: int, time_control: Dict[str, Any], nodes: int) -> Optional[int]:
        """
        Get dynamic search depth based on time control and nodes searched

        Args:
            depth: Current search depth
            max_depth: Maximum search depth
            time_control: Time control parameters
            nodes: Nodes searched so far

        Returns:
            Dynamic depth to use for search, or None if no limit
        """
        if time_control.get('depth'):
            return time_control['depth']
        if time_control.get('movetime'):
            return None
        if time_control.get('infinite'):
            return None
        if time_control.get('wtime') or time_control.get('btime'):
            # Use a heuristic based on nodes searched
            if nodes < 1000:
                return max(1, depth)
            elif nodes < 5000:
                return max(2, depth)
            elif nodes < 10000:
                return max(3, depth)
            elif nodes < 20000:
                return max(4, depth)
            elif nodes < 50000:
                return max(5, depth)
            else:
                return max(max_depth, depth)
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

# Example usage and testing
if __name__ == "__main__":
    import chess

    # Test time manager
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
