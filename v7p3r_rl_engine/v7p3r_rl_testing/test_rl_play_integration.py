#!/usr/bin/env python3
"""
Test the updated play_v7p3r.py with RL engine integration
"""

import sys
import os

# Add path for imports
sys.path.insert(0, os.path.abspath('.'))

def test_rl_vs_stockfish():
    """Test RL engine vs Stockfish."""
    print("Testing RL vs Stockfish integration with play_v7p3r...")
    
    try:
        # Import the updated play_v7p3r
        from v7p3r_engine.v7p3r_play import ChessGame
        
        # Configuration for RL vs Stockfish
        config = {
            "starting_position": "default",
            "white_player": "v7p3r_rl",  # Use RL engine as white
            "black_player": "stockfish",  # Use Stockfish as black
            "game_count": 1,
            "rl_config_path": "config/v7p3r_rl_config.yaml",
            "engine_config": {
                "name": "v7p3r_rl",
                "version": "1.0.0",
                "color": "white",
                "ruleset": "default_evaluation",
                "search_algorithm": "neural_rl",
                "depth": 3,
                "max_depth": 5,
                "monitoring_enabled": True,
                "verbose_output": False,
                "logger": "v7p3r_rl_logger",
                "game_count": 1,
                "starting_position": "default",
                "white_player": "v7p3r_rl",
                "black_player": "stockfish",
            },
            "stockfish_config": {
                "stockfish_path": "engine_utilities/external_engines/stockfish/stockfish-windows-x86-64-avx2.exe",
                "elo_rating": 800,  # Easy level for testing
                "skill_level": 1,
                "debug_mode": False,
                "depth": 2,
                "max_depth": 3,
                "movetime": 500,
            },
        }
        
        print("1. Creating chess game with RL engine...")
        game = ChessGame(config)
        
        print("2. Available engines:")
        for engine_name in game.engines.keys():
            print(f"   - {engine_name}")
        
        print("3. Starting a micro game (3 moves max)...")
        
        moves_played = 0
        max_test_moves = 3  # Just play a few moves for testing
        
        print(f"White: {game.white_player} vs Black: {game.black_player}")
        
        while not game.board.is_game_over() and moves_played < max_test_moves:
            print(f"Move {moves_played + 1}: {game.board.fen()}")
            
            game.process_engine_move()
            moves_played += 1
            
            if game.board.is_game_over():
                print(f"Game ended: {game.get_board_result()}")
                break
        
        print("4. Cleaning up...")
        game.cleanup_engines()
        
        print("✓ RL vs Stockfish integration test completed successfully!")
        return True
        
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_rl_vs_stockfish()
    sys.exit(0 if success else 1)
