#!/usr/bin/env python3
"""
Simple test of RL engine integration
"""

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def simple_test():
    print("Simple RL engine integration test...")
    
    try:
        from v7p3r_engine.v7p3r_play import ChessGame
        
        config = {
            'game_count': 1,
            'white_player': 'v7p3r_rl',
            'black_player': 'v7p3r',
            'rl_config_path': 'config/v7p3r_rl_config.yaml',
            'stockfish_config': {
                'stockfish_path': 'engine_utilities/external_engines/stockfish/stockfish-windows-x86-64-avx2.exe',
                'elo_rating': 1500
            }
        }
        
        print("Creating game...")
        game = ChessGame(config)
        
        print(f"Available engines: {list(game.engines.keys())}")
        
        if 'v7p3r_rl' in game.engines:
            print("✓ RL engine loaded successfully!")
        else:
            print("✗ RL engine not found")
            
        game.cleanup_engines()
        print("✓ Test completed!")
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False

if __name__ == "__main__":
    simple_test()
