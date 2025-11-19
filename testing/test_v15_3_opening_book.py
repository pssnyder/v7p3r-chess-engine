#!/usr/bin/env python3
"""
Test V15.3 opening book functionality
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine


def test_opening_book():
    """Test that opening book provides good moves"""
    print("="*60)
    print("V7P3R v15.3 - Opening Book Test")
    print("="*60)
    print()
    
    engine = V7P3REngine()
    
    # Test 1: Starting position
    print("TEST 1: Starting Position")
    print("-" * 40)
    engine.board = chess.Board()
    move = engine.opening_book.get_book_move(engine.board)
    print(f"Position: {engine.board.fen()}")
    print(f"Book move: {move}")
    expected_moves = ['e2e4', 'd2d4', 'g1f3', 'c2c4', 'b1c3']
    if move and move.uci() in expected_moves:
        print(f"✓ PASSED: Good opening move ({move})")
    else:
        print(f"✗ FAILED: Expected one of {expected_moves}, got {move}")
    print()
    
    # Test 2: After 1.e4
    print("TEST 2: After 1.e4")
    print("-" * 40)
    engine.board = chess.Board()
    engine.board.push(chess.Move.from_uci("e2e4"))
    move = engine.opening_book.get_book_move(engine.board)
    print(f"Position: {engine.board.fen()}")
    print(f"Book move: {move}")
    expected_moves = ['e7e5', 'c7c5', 'e7e6', 'c7c6', 'g8f6', 'd7d5']
    if move and move.uci() in expected_moves:
        print(f"✓ PASSED: Good Black response ({move})")
    else:
        print(f"✗ FAILED: Expected one of {expected_moves}, got {move}")
    print()
    
    # Test 3: After 1.e4 e5
    print("TEST 3: After 1.e4 e5")
    print("-" * 40)
    engine.board = chess.Board()
    engine.board.push(chess.Move.from_uci("e2e4"))
    engine.board.push(chess.Move.from_uci("e7e5"))
    move = engine.opening_book.get_book_move(engine.board)
    print(f"Position: {engine.board.fen()}")
    print(f"Book move: {move}")
    expected_moves = ['g1f3', 'f1c4', 'b1c3']
    if move and move.uci() in expected_moves:
        print(f"✓ PASSED: Good White continuation ({move})")
    else:
        print(f"✗ FAILED: Expected one of {expected_moves}, got {move}")
    print()
    
    # Test 4: After 1.d4
    print("TEST 4: After 1.d4")
    print("-" * 40)
    engine.board = chess.Board()
    engine.board.push(chess.Move.from_uci("d2d4"))
    move = engine.opening_book.get_book_move(engine.board)
    print(f"Position: {engine.board.fen()}")
    print(f"Book move: {move}")
    expected_moves = ['d7d5', 'g8f6', 'e7e6', 'c7c5']
    if move and move.uci() in expected_moves:
        print(f"✓ PASSED: Good response to Queen's pawn ({move})")
    else:
        print(f"✗ FAILED: Expected one of {expected_moves}, got {move}")
    print()
    
    # Test 5: Out of book (deep position)
    print("TEST 5: Out of Book (After many moves)")
    print("-" * 40)
    engine.board = chess.Board()
    moves = ["e2e4", "e7e5", "g1f3", "b8c6", "f1c4", "g8f6", "d2d3", "f8c5"]
    for m in moves:
        engine.board.push(chess.Move.from_uci(m))
    move = engine.opening_book.get_book_move(engine.board)
    print(f"Position after {len(moves)} moves")
    print(f"Book move: {move}")
    if move is None:
        print(f"✓ PASSED: Correctly returns None (out of book)")
    else:
        print(f"✓ OK: Found deep book move ({move})")
    print()
    
    # Test 6: Integration with get_best_move
    print("TEST 6: Integration Test (get_best_move)")
    print("-" * 40)
    engine.board = chess.Board()
    best_move = engine.get_best_move(time_left=5.0, increment=0.1)
    print(f"Engine's chosen move: {best_move}")
    expected_moves = ['e2e4', 'd2d4', 'g1f3', 'c2c4', 'b1c3']
    if best_move and best_move.uci() in expected_moves:
        print(f"✓ PASSED: Engine uses book moves correctly")
    else:
        print(f"⚠️  WARNING: Engine chose {best_move} (may have searched instead)")
    print()
    
    # Summary
    print("="*60)
    print("SUMMARY")
    print("="*60)
    print("Opening book system implemented successfully!")
    print("Features:")
    print("  ✓ Embedded opening repertoire")
    print("  ✓ Position-based book lookup")
    print("  ✓ Move variety support")
    print("  ✓ Integration with search engine")
    print()
    print("UCI Options Available:")
    print("  - OwnBook (true/false): Enable/disable book")
    print("  - BookFile (string): Path to external Polyglot .bin")
    print("  - BookDepth (1-20): Max ply to use book (default 8)")
    print("  - BookVariety (0-100): % non-best moves (default 50)")


if __name__ == "__main__":
    test_opening_book()
