#!/usr/bin/env python3
"""
V7P3R v10.9 Perspective Fix Validation Test
Tests to ensure the critical perspective bug has been fixed
"""

import chess
import sys
import os

# Add the src directory to the path so we can import V7P3R
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from v7p3r import V7P3REngine


def test_perspective_consistency():
    """Test that evaluation is consistent regardless of side to move"""
    print("Testing V7P3R v10.9 Perspective Fix")
    print("=" * 50)
    
    engine = V7P3REngine()
    
    # Test position where White is clearly better
    # Italian Game position where White has development advantage
    test_fen = "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 4 4"
    
    # Test from White's perspective
    board_white = chess.Board(test_fen)
    eval_white = engine._evaluate_position(board_white)
    
    # Test from Black's perspective (make a null move to switch turns)
    board_black = chess.Board(test_fen)
    board_black.turn = not board_black.turn  # Flip turn
    eval_black = engine._evaluate_position(board_black)
    
    print(f"Test Position: {test_fen}")
    print(f"White to move evaluation: {eval_white:.3f}")
    print(f"Black to move evaluation: {eval_black:.3f}")
    print(f"Perspective difference: {abs(eval_white + eval_black):.3f}")
    
    # In negamax, evaluations should be exact opposites for the same position
    # with different sides to move (within small floating point tolerance)
    if abs(eval_white + eval_black) < 0.01:
        print("✅ PASS: Evaluations are properly negated")
        perspective_test_passed = True
    else:
        print("❌ FAIL: Evaluations are NOT properly negated")
        perspective_test_passed = False
    
    print()
    
    # Test multiple positions to ensure consistency
    test_positions = [
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",  # After 1.e4
        "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq e6 0 3",  # After 1.e4 e5 2.Nf3 Nf6
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",  # Italian game
        "8/8/8/4k3/4K3/8/8/8 w - - 0 1",  # King and pawn endgame
    ]
    
    all_passed = True
    
    for i, fen in enumerate(test_positions):
        board1 = chess.Board(fen)
        eval1 = engine._evaluate_position(board1)
        
        board2 = chess.Board(fen)
        board2.turn = not board2.turn
        eval2 = engine._evaluate_position(board2)
        
        diff = abs(eval1 + eval2)
        passed = diff < 0.01
        all_passed &= passed
        
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"Position {i+1}: {status} (diff: {diff:.6f})")
    
    print()
    print("=" * 50)
    if all_passed and perspective_test_passed:
        print("🎉 ALL TESTS PASSED - Perspective fix successful!")
        print("V7P3R v10.9 should now play consistently as both White and Black")
    else:
        print("💥 TESTS FAILED - Perspective issue remains")
        print("Additional debugging required")
    
    return all_passed and perspective_test_passed


def test_basic_search():
    """Test that the engine can search without errors"""
    print("\nTesting basic search functionality...")
    
    engine = V7P3REngine()
    board = chess.Board()
    
    try:
        # Test search from starting position
        move = engine.search(board, 1.0)  # 1 second search
        
        if move and move in board.legal_moves:
            print(f"✅ Search successful: Found move {move}")
            return True
        else:
            print("❌ Search failed: Invalid or no move returned")
            return False
            
    except Exception as e:
        print(f"❌ Search failed with exception: {e}")
        return False


if __name__ == "__main__":
    # Run all tests
    perspective_ok = test_perspective_consistency()
    search_ok = test_basic_search()
    
    print("\n" + "=" * 50)
    print("FINAL TEST RESULTS:")
    print(f"Perspective Fix: {'✅ PASS' if perspective_ok else '❌ FAIL'}")
    print(f"Basic Search: {'✅ PASS' if search_ok else '❌ FAIL'}")
    
    if perspective_ok and search_ok:
        print("\n🚀 V7P3R v10.9 is ready for tournament testing!")
        sys.exit(0)
    else:
        print("\n🛑 V7P3R v10.9 requires additional fixes before release")
        sys.exit(1)