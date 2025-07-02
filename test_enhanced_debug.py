#!/usr/bin/env python3
"""
Test enhanced metrics with debug output
"""
import sys
import os
import logging

# Add the v7p3r_engine path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'v7p3r_engine'))

def test_enhanced_metrics_debug():
    print("ENHANCED METRICS DEBUG TEST")
    print("=" * 40)
    
    # Setup logging to see all debug output
    logging.basicConfig(level=logging.DEBUG)
    
    from v7p3r_engine.v7p3r_play import ChessGame
    
    config = {
        "engine_config": {
            "depth": 3,
            "max_moves": 5,  # Very short game
            "monitoring_enabled": True
        },
        "stockfish_config": {
            "depth": 2
        }
    }
    
    game = ChessGame(config)
    
    print(f"Enhanced metrics enabled: {game.use_enhanced_metrics}")
    
    # Play a very short game with debug output
    try:
        game.run()
        print("Game completed successfully")
    except Exception as e:
        print(f"Game failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_enhanced_metrics_debug()
