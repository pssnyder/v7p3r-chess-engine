#!/usr/bin/env python3
"""
Test V17.1 fixes:
1. Opening book integration
2. PV instant moves disabled
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine

def test_opening_book():
    """Test that opening book provides moves"""
    print("=" * 60)
    print("TEST 1: Opening Book Integration")
    print("=" * 60)
    
    engine = V7P3REngine()
    board = chess.Board()
    
    print("\nStarting position:")
    print(board)
    print(f"\nFEN: {board.fen()}")
    
    move = engine.search(board, time_limit=5)
    print(f"\nMove selected: {move.uci() if move else 'None'}")
    
    # Expected: e2e4, d2d4, or g1f3 from opening book
    expected_moves = ['e2e4', 'd2d4', 'g1f3']
    if move and move.uci() in expected_moves:
        print("[OK] Opening book move selected!")
        return True
    else:
        print("[FAIL] Expected opening book move, got:", move.uci() if move else "None")
        return False


def test_no_pv_instant_moves():
    """Test that PV instant moves are disabled (depth > 1 always)"""
    print("\n" + "=" * 60)
    print("TEST 2: PV Instant Moves Disabled")
    print("=" * 60)
    
    engine = V7P3REngine()
    
    # Play the critical game sequence from tournament
    moves = [
        "e2e3", "b8c6", "g1f3", "g8f6", "b1c3", "d7d5",
        "f1b5", "a7a6", "b5c6", "b7c6", "f3e5", "d8d6",
        "d2d4", "f6e4", "c3e4", "d5e4", "d1h5", "g7g6"
    ]
    
    board = chess.Board()
    for move_uci in moves:
        move = chess.Move.from_uci(move_uci)
        board.push(move)
    
    print("\nCritical position (after 9.Qh5 g6):")
    print(board)
    print(f"\nFEN: {board.fen()}")
    print("\nIn v17.0, this position produced f6 blunder with 'depth 1, 0 nodes'")
    print("In v17.1, PV instant moves are disabled - should see normal search depth\n")
    
    # Make move (should NOT be instant depth 1)
    move = engine.search(board, time_limit=5)
    
    print(f"\nMove selected: {move.uci() if move else 'None'}")
    print(f"Nodes searched: {engine.nodes_searched}")
    
    # Verify NOT a PV instant move (should have searched nodes)
    if engine.nodes_searched > 100:
        print(f"[OK] Normal search performed ({engine.nodes_searched} nodes)")
        print("[OK] PV instant moves are disabled!")
        return True
    else:
        print(f"[FAIL] Very few nodes searched ({engine.nodes_searched})")
        print("[FAIL] Possible instant move behavior!")
        return False


def main():
    print("\nV17.1 FIX VALIDATION TEST")
    print("Testing two critical fixes:")
    print("1. Opening book integration (prevents weak positions)")
    print("2. PV instant moves disabled (prevents tactical blunders)")
    print("\n")
    
    test1_pass = test_opening_book()
    test2_pass = test_no_pv_instant_moves()
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"Opening book test: {'PASS' if test1_pass else 'FAIL'}")
    print(f"PV disable test: {'PASS' if test2_pass else 'FAIL'}")
    
    if test1_pass and test2_pass:
        print("\n[SUCCESS] All V17.1 fixes validated!")
        print("Engine is ready for Arena testing")
    else:
        print("\n[WARNING] Some tests failed - review implementation")
    
    print("=" * 60)


if __name__ == "__main__":
    main()
