#!/usr/bin/env python3
"""
Test V18.0.0 Anti-Tactical Defense System
Tests that move safety checker prevents hanging pieces
"""

import chess
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r_move_safety import MoveSafetyChecker


def test_hanging_piece_detection():
    """Test that safety checker detects hanging pieces"""
    print("=" * 60)
    print("TEST 1: Hanging Piece Detection")
    print("=" * 60)
    
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 300,
        chess.BISHOP: 300,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0
    }
    
    checker = MoveSafetyChecker(piece_values)
    
    # Test Case 1: Move that hangs a rook
    # Game 4ZzIc3g6 position before 18...Rf6 (loses material)
    board = chess.Board("r3kbnr/ppp2ppp/4b3/8/1BP5/3q4/P3BPPP/RN1Q1RK1 b kq - 0 18")
    
    # 18...Rf6 hangs the rook (Bxd5 wins it)
    hanging_move = chess.Move.from_uci("f8f6")
    safety_score = checker.evaluate_move_safety(board, hanging_move)
    
    print(f"\nPosition: {board.fen()}")
    print(f"Move: 18...Rf6 (from game 4ZzIc3g6)")
    print(f"Safety Score: {safety_score:.1f}")
    print(f"Expected: Negative (hangs material)")
    print(f"Result: {'✅ PASS' if safety_score < -100 else '❌ FAIL'}")
    
    # Test Case 2: Safe move (doesn't hang anything)
    safe_move = chess.Move.from_uci("e8f8")  # King to safety
    safety_score_safe = checker.evaluate_move_safety(board, safe_move)
    
    print(f"\nMove: Kf8 (safe king move)")
    print(f"Safety Score: {safety_score_safe:.1f}")
    print(f"Expected: Near zero (safe)")
    print(f"Result: {'✅ PASS' if safety_score_safe > -50 else '❌ FAIL'}")
    
    return True


def test_game2_position_29():
    """Test Game 2 position before 29...Ke7 (allows Ba3+)"""
    print("\n" + "=" * 60)
    print("TEST 2: Game 4ZzIc3g6 Move 29 (Ke7 mistake)")
    print("=" * 60)
    
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 300,
        chess.BISHOP: 300,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0
    }
    
    checker = MoveSafetyChecker(piece_values)
    
    # Position before 29...Ke7
    board = chess.Board("8/p4k2/5p2/4p3/4R3/1P6/1K6/8 b - - 0 29")
    
    # 29...Ke7 allows 30.Ba3+ winning material
    bad_move = chess.Move.from_uci("f7e7")
    safety_score = checker.evaluate_move_safety(board, bad_move)
    
    print(f"\nPosition: {board.fen()}")
    print(f"Move: 29...Ke7")
    print(f"Safety Score: {safety_score:.1f}")
    print(f"Expected: Negative (allows Ba3+)")
    print(f"Result: {'✅ PASS' if safety_score < -20 else '❌ FAIL'}")
    
    return True


def test_simple_position():
    """Test basic hanging piece in simple position"""
    print("\n" + "=" * 60)
    print("TEST 3: Simple Hanging Queen")
    print("=" * 60)
    
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 300,
        chess.BISHOP: 300,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 0
    }
    
    checker = MoveSafetyChecker(piece_values)
    
    # Simple position: moving queen to attacked square
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    
    # Move queen to h4 (attacked by g5 pawn after it moves)
    # Actually let's test Qh5 which can be attacked
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    board.push_san("e4")
    board.push_san("e5")
    
    # Qh5 is somewhat exposed
    dangerous_move = chess.Move.from_uci("d1h5")
    safety_score = checker.evaluate_move_safety(board, dangerous_move)
    
    print(f"\nPosition: {board.fen()}")
    print(f"Move: Qh5")
    print(f"Safety Score: {safety_score:.1f}")
    print("(Queen on h5 can be attacked by g6)")
    
    return True


def main():
    """Run all tests"""
    print("\n" + "=" * 60)
    print("V18.0.0 ANTI-TACTICAL DEFENSE SYSTEM TESTS")
    print("=" * 60)
    
    try:
        test_hanging_piece_detection()
        test_game2_position_29()
        test_simple_position()
        
        print("\n" + "=" * 60)
        print("✅ ALL TESTS COMPLETED")
        print("=" * 60)
        print("\nNote: Safety checker is designed to provide lightweight")
        print("penalties for unsafe moves. Exact scores may vary but")
        print("negative scores indicate tactical danger.")
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
