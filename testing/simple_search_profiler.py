#!/usr/bin/env python3
"""
Simple search profiler for V7P3R
"""

import cProfile
import pstats
import sys
import time
import chess

sys.path.append('../src')
from v7p3r import V7P3REngine

def profile_search():
    print('Profiling V7P3R Search...')
    engine = V7P3REngine()
    test_board = chess.Board('rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2')

    profiler = cProfile.Profile()
    profiler.enable()

    start_time = time.time()
    best_move = engine.search(test_board, time_limit=5.0, depth=4)
    elapsed = time.time() - start_time

    profiler.disable()

    print(f'Search time: {elapsed:.3f}s')
    print(f'Best move: {best_move}')
    print(f'Nodes: {engine.nodes_searched}')
    if elapsed > 0:
        print(f'NPS: {engine.nodes_searched / elapsed:.0f}')

    print('\nTOP FUNCTIONS BY CUMULATIVE TIME:')
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(15)
    
    print('\nTOP FUNCTIONS BY TOTAL TIME:')
    stats.sort_stats('tottime')
    stats.print_stats(10)

if __name__ == "__main__":
    profile_search()