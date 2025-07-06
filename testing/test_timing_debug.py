#!/usr/bin/env python3
"""
Debug Timing Test - Debug the timing calculation in move display
"""

import chess
import time
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(project_root)
sys.path.insert(0, parent_dir)

from v7p3r_play import v7p3rChess


def test_timing_debug():
    """Test timing calculation in detail"""
    print("=== Timing Debug Test ===")
    
    game = v7p3rChess(config_name="speed_config")
    
    print("Starting move process...")
    
    # Manually set timing for testing
    start_time = time.time()
    game.move_start_time = start_time
    
    # Simulate thinking time
    time.sleep(0.1)  # 100ms delay
    
    game.move_end_time = time.time()
    calculated_time = game.move_end_time - game.move_start_time
    
    print(f"Start time: {game.move_start_time}")
    print(f"End time: {game.move_end_time}")
    print(f"Calculated time: {calculated_time:.3f}s")
    
    # Test display with manual timing
    test_move = chess.Move.from_uci("e2e4")
    game.board.push(test_move)
    
    print(f"Calling display_move_made with time: {calculated_time:.3f}s")
    game.display_move_made(test_move, calculated_time)
    
    print("\n=== Timing Debug Complete ===")


if __name__ == "__main__":
    test_timing_debug()
