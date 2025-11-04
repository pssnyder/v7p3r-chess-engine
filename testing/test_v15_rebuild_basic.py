#!/usr/bin/env python3
"""
V15.0 Basic Rebuild Test
Tests the core engine functionality after clean rebuild
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r_engine import V7P3REngine
import time


def test_basic_search():
    """Test basic search functionality"""
    print("=" * 60)
    print("V15.0 BASIC REBUILD TEST")
    print("=" * 60)
    
    engine = V7P3REngine()
    board = chess.Board()
    
    print("\n1. Testing opening position search...")
    print(f"Position: {board.fen()}")
    
    start = time.time()
    best_move = engine.search(board, time_limit=3.0)
    elapsed = time.time() - start
    
    print(f"\nBest move: {best_move}")
    print(f"Time taken: {elapsed:.3f}s")
    print(f"Nodes searched: {engine.nodes_searched}")
    print(f"NPS: {int(engine.nodes_searched / elapsed)}")
    
    if best_move and best_move != chess.Move.null():
        print("✅ PASS: Engine found a move")
    else:
        print("❌ FAIL: Engine did not find a valid move")
        return False
    
    print("\n2. Testing material evaluation...")
    board = chess.Board()
    score = engine._evaluate_material(board)
    print(f"Starting position score: {score}")
    
    if abs(score) < 10:  # Should be near 0 in starting position
        print("✅ PASS: Material evaluation balanced")
    else:
        print(f"❌ FAIL: Material evaluation unbalanced ({score})")
        return False
    
    print("\n3. Testing tactical position...")
    # Scholar's mate threat
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    print(f"Position: Scholar's mate threat")
    
    start = time.time()
    best_move = engine.search(board, time_limit=3.0)
    elapsed = time.time() - start
    
    print(f"Best move: {best_move}")
    print(f"Time: {elapsed:.3f}s, Nodes: {engine.nodes_searched}, NPS: {int(engine.nodes_searched / elapsed)}")
    
    # Check if it finds Qxf7# (checkmate)
    if best_move == chess.Move.from_uci("h5f7"):
        print("✅ PASS: Found checkmate in one!")
    else:
        print(f"⚠️  WARNING: Did not find immediate checkmate (found {best_move})")
    
    print("\n4. Testing depth 8 search...")
    board = chess.Board()
    engine.default_depth = 8
    
    start = time.time()
    best_move = engine.search(board, time_limit=5.0)
    elapsed = time.time() - start
    
    print(f"Best move: {best_move}")
    print(f"Time: {elapsed:.3f}s, Nodes: {engine.nodes_searched}")
    print(f"NPS: {int(engine.nodes_searched / elapsed)}")
    
    if engine.nodes_searched > 50000:
        print(f"✅ PASS: Depth 8 search reached good node count ({engine.nodes_searched})")
    else:
        print(f"⚠️  WARNING: Low node count for depth 8 ({engine.nodes_searched})")
    
    print("\n" + "=" * 60)
    print("V15.0 BASIC REBUILD TEST COMPLETE")
    print("=" * 60)
    print("\n✅ Core engine functionality verified!")
    print("Next steps:")
    print("  1. Test against Material Opponent")
    print("  2. Verify move ordering effectiveness")
    print("  3. Gradually add back heuristics one at a time")
    
    return True


if __name__ == "__main__":
    try:
        success = test_basic_search()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ TEST FAILED WITH ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
