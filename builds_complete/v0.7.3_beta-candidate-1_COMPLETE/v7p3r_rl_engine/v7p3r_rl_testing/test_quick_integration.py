#!/usr/bin/env python3
"""Quick final integration test"""

import sys
sys.path.insert(0, '.')

from v7p3r_engine.v7p3r_play import v7p3rChess

# Test basic config
config = {
    'game_count': 1,
    'white_player': 'v7p3r_rl',
    'black_player': 'stockfish',
    'stockfish_config': {
        'stockfish_path': 'engine_utilities/external_engines/stockfish/stockfish-windows-x86-64-avx2.exe',
        'elo_rating': 100,
        'skill_level': 1,
        'depth': 2,
        'movetime': 500
    }
}

print('Creating ChessGame with RL vs Stockfish...')
game = v7p3rChess()
print(f'Available engines: {list(game.engines.keys())}')
print('Γ£ô v7p3r_rl' if 'v7p3r_rl' in game.engines else 'Γ£ù v7p3r_rl')
print('Γ£ô v7p3r_ga' if 'v7p3r_ga' in game.engines else 'Γ£ù v7p3r_ga') 
print('Γ£ô v7p3r_nn' if 'v7p3r_nn' in game.engines else 'Γ£ù v7p3r_nn')
print('Cleaning up...')
game.cleanup_engines()
print('Final integration test successful!')
