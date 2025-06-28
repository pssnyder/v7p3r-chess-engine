#!/usr/bin/env python3
"""
Test integration of v7p3r_rl engine with play_v7p3r.py
"""

import sys
import os

# Add path for imports
sys.path.insert(0, os.path.abspath('.'))

def test_rl_integration():
    """Test RL engine integration with play_v7p3r.py."""
    print("Testing v7p3r RL Engine integration with play_v7p3r.py...")
    
    try:
        from v7p3r_engine.play_v7p3r import ChessGame
        
        # Configuration for RL vs RL test
        config = {
            "game_count": 1,
            "white_player": "v7p3r_rl",
            "black_player": "v7p3r_rl",
            "rl_config_path": "config/v7p3r_rl_config.yaml",
            "stockfish_config": {
                "stockfish_path": "engine_utilities/external_engines/stockfish/stockfish-windows-x86-64-avx2.exe",
                "elo_rating": 1500,
                "skill_level": 10,
                "debug_mode": False,
                "depth": 3,
                "max_depth": 5,
                "movetime": 1000,
            },
            "search_algorithm": "lookahead",
            "depth": 3,
            "max_depth": 5,
            "monitoring_enabled": True,
            "verbose_output": True,
            "logger": "v7p3r_engine_logger",
            "max_game_count": 1,
            "starting_position": "default"
        }
        
        print("1. Initializing game...")
        game = ChessGame(config)
        
        print("2. Testing engine availability...")
        if 'v7p3r_rl' in game.engines:
            print("   ✓ RL engine successfully loaded")
        else:
            print("   ✗ RL engine not available")
            return False
        
        print("3. Running very short game...")
        # Run a few moves only to test functionality
        move_count = 0
        max_test_moves = 6  # Very short test
        
        while not game.board.is_game_over() and move_count < max_test_moves:
            game.process_engine_move()
            move_count += 1
            print(f"   Move {move_count}: {game.board.move_stack[-1] if game.board.move_stack else 'None'}")
        
        print("4. Cleaning up...")
        game.cleanup_engines()
        
        print("✓ RL engine integration test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_engine_vs_stockfish():
    """Test RL engine vs Stockfish."""
    print("\\nTesting v7p3r_rl vs Stockfish...")
    
    try:
        from v7p3r_engine.play_v7p3r import ChessGame
        
        config = {
            "game_count": 1,
            "white_player": "v7p3r_rl", 
            "black_player": "stockfish",
            "rl_config_path": "config/v7p3r_rl_config.yaml",
            "stockfish_config": {
                "stockfish_path": "engine_utilities/external_engines/stockfish/stockfish-windows-x86-64-avx2.exe",
                "elo_rating": 800,  # Low ELO for easier win
                "skill_level": 1,
                "depth": 2,
                "movetime": 100,
            }
        }
        
        game = ChessGame(config)
        
        # Quick test
        move_count = 0
        max_test_moves = 4
        
        while not game.board.is_game_over() and move_count < max_test_moves:
            game.process_engine_move()
            move_count += 1
        
        game.cleanup_engines()
        print("✓ RL vs Stockfish test passed!")
        return True
        
    except Exception as e:
        print(f"✗ RL vs Stockfish test failed: {e}")
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("v7p3r RL Engine Integration Tests")
    print("=" * 60)
    
    success1 = test_rl_integration()
    success2 = test_engine_vs_stockfish()
    
    if success1 and success2:
        print("\\n🎉 All integration tests passed!")
        sys.exit(0)
    else:
        print("\\n❌ Some tests failed!")
        sys.exit(1)
