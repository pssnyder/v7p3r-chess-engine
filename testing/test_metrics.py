#!/usr/bin/env python3
"""
Quick test to verify move metrics are being properly collected
"""
import sys
import os
# Add the parent directory to Python path to allow imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
from v7p3r_engine.v7p3r_play import v7p3rChess

# Test configuration with just a few moves
test_config = {
    "game_config": {
        "game_count": 1,                     # Just one game
        "starting_position": "default",
        "white_player": "v7p3r",
        "black_player": "stockfish",
    },
    "engine_config": {
        "name": "v7p3r",
        "version": "1.0.0",
        "ruleset": "tuned_ga_gen2",
        "search_algorithm": "minimax",
        "depth": 2,                          # Lower depth for faster testing
        "max_depth": 2,
        "max_moves": 4,
        "use_game_phase": True,
        "monitoring_enabled": True,
        "verbose_output": True,
        "logger": "v7p3r_engine_logger",
    },
    "stockfish_config": {
        "stockfish_path": "v7p3r_engine/external_engines/stockfish/stockfish-windows-x86-64-avx2.exe",
        "elo_rating": 100,
        "skill_level": 1,
        "debug_mode": False,
        "depth": 1,                          # Lower depth for faster testing
        "max_depth": 1,
        "movetime": 100,  # Very short time for Stockfish
    },
}

if __name__ == "__main__":
    print("Starting test game with updated metrics collection...")
    game = v7p3rChess(config=test_config)
    # Run for a few moves to test metrics
    try:
        game.run(debug_mode=False)
    except KeyboardInterrupt:
        print("\nGame stopped manually.")
    print("Test game completed. Check the database for updated metrics.")
