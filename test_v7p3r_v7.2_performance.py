#!/usr/bin/env python3
"""
Test V7P3R v7.2 performance improvements
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import chess
import time
from v7p3r import V7P3RCleanEngine


def test_performance_improvements():
    """Test the performance improvements in v7.2"""
    engine = V7P3RCleanEngine()
    
    print("Testing V7P3R v7.2 Performance Improvements")
    print("=" * 50)
    
    # Test 1: Opening position with time limit
    board1 = chess.Board()
    print("Test 1: Opening position (should be fast)")
    start = time.time()
    move1 = engine.search(board1, time_limit=5.0)
    elapsed1 = time.time() - start
    print(f"Move: {move1}, Time: {elapsed1:.2f}s, Nodes: {engine.nodes_searched}")
    print()
    
    # Test 2: Complex middle game position
    board2 = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 6")
    print("Test 2: Complex middle game position")
    start = time.time()
    move2 = engine.search(board2, time_limit=8.0)
    elapsed2 = time.time() - start
    print(f"Move: {move2}, Time: {elapsed2:.2f}s, Nodes: {engine.nodes_searched}")
    print()
    
    # Test 3: Tactical position (should find tactic quickly)
    board3 = chess.Board("rnbqkb1r/pppp1ppp/5n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 2 4")
    print("Test 3: Tactical position (Scholar's mate setup)")
    start = time.time()
    move3 = engine.search(board3, time_limit=6.0)
    elapsed3 = time.time() - start
    print(f"Move: {move3}, Time: {elapsed3:.2f}s, Nodes: {engine.nodes_searched}")
    print()
    
    # Test 4: Endgame position
    board4 = chess.Board("8/8/8/8/8/8/k1K5/1R6 w - - 0 1")
    print("Test 4: Endgame position")
    start = time.time()
    move4 = engine.search(board4, time_limit=4.0)
    elapsed4 = time.time() - start
    print(f"Move: {move4}, Time: {elapsed4:.2f}s, Nodes: {engine.nodes_searched}")
    print()
    
    # Summary
    avg_time = (elapsed1 + elapsed2 + elapsed3 + elapsed4) / 4
    print("=" * 50)
    print(f"Average time per move: {avg_time:.2f}s")
    
    if avg_time < 10.0:
        print("✅ SUCCESS: Average time under 10 seconds (competitive with SlowMate)")
    elif avg_time < 15.0:
        print("⚠️  IMPROVED: Better than before, but still needs optimization")
    else:
        print("❌ NEEDS WORK: Still too slow")
        
    # Test killer moves and history
    print("\nTesting killer moves and history heuristic:")
    print(f"Killer moves stored: {len(engine.killer_moves)} plies")
    print(f"History entries: {len(engine.history_scores)}")


if __name__ == "__main__":
    test_performance_improvements()
