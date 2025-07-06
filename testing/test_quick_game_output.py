#!/usr/bin/env python3
"""
Quick Game Test - Test the full game flow with corrected display output
"""

import chess
import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(project_root)
sys.path.insert(0, parent_dir)

from v7p3r_play import v7p3rChess


def test_quick_game():
    """Test a quick 2-move game to verify all output"""
    print("=== Quick Game Test ===")
    
    # Create a game instance with speed config for faster moves
    game = v7p3rChess(config_name="speed_config")
    
    print(f"Verbose output enabled: {game.verbose_output_enabled}")
    print(f"Monitoring enabled: {game.monitoring_enabled}")
    
    # Play 2 moves to test output
    move_count = 0
    max_moves = 4
    
    while not game.board.is_game_over() and move_count < max_moves:
        print(f"\n--- Move {move_count + 1} ---")
        game.process_engine_move()
        move_count += 1
    
    print("\n=== Quick Game Test Complete ===")


if __name__ == "__main__":
    test_quick_game()
