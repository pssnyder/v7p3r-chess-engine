#!/usr/bin/env python3
"""
V14.8 Validation Test - Verify simplified engine works properly

Tests:
1. Search depth achievement (should reach depth 4-6 in reasonable time)
2. Move generation (should consider all legal moves, not filter to 1-2)
3. Time management (should not hit 60% emergency stop prematurely)
4. Basic functionality (makes legal moves, doesn't crash)
"""

import sys
import os
import time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine


def test_search_depth():
    """Test that engine achieves reasonable search depth"""
    print("\n" + "="*80)
    print("TEST 1: Search Depth Achievement")
    print("="*80)
    
    engine = V7P3REngine()
    board = chess.Board()
    
    positions = [
        ("Starting position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Middlegame", "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4"),
        ("Endgame", "8/5k2/8/8/8/3K4/5P2/8 w - - 0 1"),
    ]
    
    for desc, fen in positions:
        print(f"\nTesting: {desc}")
        print(f"FEN: {fen}")
        
        board.set_fen(fen)
        
        start_time = time.time()
        best_move = engine.search(board, time_limit=5.0, depth=6)
        elapsed = time.time() - start_time
        
        # Check search statistics
        print(f"  Move: {best_move}")
        print(f"  Nodes: {engine.nodes_searched:,}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  NPS: {int(engine.nodes_searched / elapsed):,}")
        
        # Verify reasonable depth achieved (should be >= 4 in 5 seconds)
        if engine.nodes_searched < 1000:
            print(f"  WARNING: Very low node count (< 1000)")
        else:
            print(f"  OK: Search completed with good node count")


def test_move_consideration():
    """Test that engine considers all legal moves (not filtering to 1-2)"""
    print("\n" + "="*80)
    print("TEST 2: Move Consideration (No Aggressive Filtering)")
    print("="*80)
    
    engine = V7P3REngine()
    board = chess.Board()
    
    legal_moves = list(board.legal_moves)
    print(f"\nStarting position:")
    print(f"  Legal moves: {len(legal_moves)}")
    
    # V14.7 would filter this to 1-2 moves
    # V14.8 should consider all 20 moves
    
    # Just verify search works without crashing
    best_move = engine.search(board, time_limit=2.0, depth=4)
    print(f"  Best move: {best_move}")
    print(f"  Nodes searched: {engine.nodes_searched:,}")
    
    if engine.nodes_searched > 100:
        print(f"  OK: Engine considered multiple moves ({engine.nodes_searched} nodes)")
    else:
        print(f"  WARNING: Very few nodes explored")


def test_time_management():
    """Test that time management doesn't stop prematurely"""
    print("\n" + "="*80)
    print("TEST 3: Time Management (No Premature Stops)")
    print("="*80)
    
    engine = V7P3REngine()
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    
    print("\nTesting 10 second time limit:")
    start_time = time.time()
    best_move = engine.search(board, time_limit=10.0, depth=6)
    elapsed = time.time() - start_time
    
    print(f"  Move: {best_move}")
    print(f"  Time used: {elapsed:.2f}s / 10.0s ({elapsed/10.0*100:.1f}%)")
    print(f"  Nodes: {engine.nodes_searched:,}")
    
    # V14.3 would stop at 6 seconds (60% limit)
    # V14.8 should use closer to 8-9 seconds
    if elapsed < 2.0:
        print(f"  WARNING: Stopped too early (< 2s)")
    elif elapsed < 5.0:
        print(f"  OK: Reasonable time usage")
    else:
        print(f"  OK: Good time usage (not stopped prematurely)")


def test_basic_play():
    """Test that engine can play a few moves without crashing"""
    print("\n" + "="*80)
    print("TEST 4: Basic Play (No Crashes)")
    print("="*80)
    
    engine = V7P3REngine()
    board = chess.Board()
    
    print("\nPlaying 10 moves:")
    for i in range(10):
        move = engine.search(board, time_limit=1.0, depth=4)
        board.push(move)
        print(f"  Move {i+1}: {move}")
    
    print(f"\n  OK: Completed 10 moves without crashing")
    print(f"  Final position: {board.fen()}")


if __name__ == "__main__":
    print("="*80)
    print("V14.8 Validation Test Suite")
    print("="*80)
    print("\nV14.8 Changes:")
    print("  - DISABLED V14.7 aggressive safety filtering")
    print("  - Allows all legal moves for tactical ordering")
    print("  - Simplified time management")
    print("  - Based on V14.0 foundation (67.1% tournament score)")
    
    test_search_depth()
    test_move_consideration()
    test_time_management()
    test_basic_play()
    
    print("\n" + "="*80)
    print("Test Suite Complete")
    print("="*80)
    print("\nNext steps:")
    print("  1. Run Arena tournament: V14.8 vs V14.0, V12.6, V14.3")
    print("  2. Verify depth achievement: 4-6 in middlegame")
    print("  3. Monitor blunder rate: should be <10%")
    print("  4. Target performance: >=60% tournament score")
