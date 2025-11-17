#!/usr/bin/env python3
"""
Test V14.3 gives_check() reduction more carefully
Count exactly where the calls are happening
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine


# Patch to count calls
gives_check_counts = {
    'during_ordering': 0,
    'total': 0
}

original_gives_check = chess.Board.gives_check

def counted_gives_check(self, move):
    gives_check_counts['total'] += 1
    return original_gives_check(self, move)

chess.Board.gives_check = counted_gives_check


def test_position(fen, name):
    """Test one position"""
    global gives_check_counts
    gives_check_counts = {'during_ordering': 0, 'total': 0}
    
    board = chess.Board(fen)
    engine = V7P3REngine()
    
    import time
    start = time.perf_counter()
    best_move = engine.search(board, time_limit=1.0)
    elapsed = time.perf_counter() - start
    
    nodes = engine.nodes_searched
    nps = int(nodes / max(elapsed, 0.001))
    total_calls = gives_check_counts['total']
    
    print(f"\n{name}:")
    print(f"  Nodes: {nodes:,}")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  NPS: {nps:,}")
    print(f"  gives_check() calls: {total_calls:,}")
    print(f"  Ratio: {total_calls / max(nodes, 1):.2f} calls/node")


print("V14.3 gives_check() Call Analysis")
print("=" * 60)

test_position(chess.STARTING_FEN, "Starting Position")
test_position("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3", "Italian Game")
test_position("rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq d6 0 3", "French Defense")

print("\n" + "=" * 60)
print("Analysis:")
print("V14.2 was calling gives_check() 5.44 times per node")
print("V14.3 should be calling it much less (only top 3 captures)")
print("\nIf still high, we need to investigate WHERE the calls are coming from")
