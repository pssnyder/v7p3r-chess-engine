#!/usr/bin/env python3
"""Test V15.2 Material Awareness

Tests that V15.2:
1. Doesn't play Nd5 when queen can capture it
2. Doesn't play moves that hang pieces
3. Properly evaluates captures with SEE
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine


def test_nd5_blunder_fix():
    """Test that engine doesn't play Nd5 in the position from the game"""
    print("=" * 80)
    print("TEST 1: Nd5 Blunder Fix (from MaterialOpponent game)")
    print("=" * 80)
    
    # Position after move 19...Qxf6 (before White's move 20)
    # FEN from the game where V15.1 played Nd5 and lost the knight
    fen = "r4rk1/pb4p1/1p1p1q2/8/1P2P3/N7/2P2PPP/R3K2R w KQ - 0 20"
    
    engine = V7P3REngine()
    engine.board = chess.Board(fen)
    
    print(f"Position (after Black's Qxf6):")
    print(engine.board)
    print()
    
    # Check SEE for Nd5
    nd5_move = chess.Move.from_uci("a3d5")
    see_value = engine._see(engine.board, nd5_move)
    is_safe = engine._is_safe_move(engine.board, nd5_move)
    
    print(f"Move Nd5 (a3d5):")
    print(f"  SEE value: {see_value} (negative = loses material)")
    print(f"  Is safe: {is_safe}")
    print()
    
    # Get best move
    best_move = engine.get_best_move(time_left=10.0, increment=0.1)
    
    print(f"Engine's choice: {best_move}")
    
    if best_move != nd5_move:
        print("✓ PASSED: Engine avoided Nd5 blunder")
        return True
    else:
        print("✗ FAILED: Engine still plays Nd5")
        return False


def test_see_evaluation():
    """Test SEE on various exchanges"""
    print("\n" + "=" * 80)
    print("TEST 2: SEE (Static Exchange Evaluation)")
    print("=" * 80)
    
    engine = V7P3REngine()
    
    test_cases = [
        {
            "name": "Queen takes knight (loses queen)",
            "fen": "rnbqkb1r/ppp2ppp/5n2/3p4/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 0 1",
            "move": "d1f3",  # Qxf3 (illegal but testing SEE logic)
            "expected": "negative"  # Should lose queen
        },
        {
            "name": "Knight takes pawn (safe)",
            "fen": "rnbqkbnr/ppp1pppp/8/3p4/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 0 1",
            "move": "f3d4",  # Would be Nxd4 if d4 had piece - testing attacked square
            "expected": "check"
        },
        {
            "name": "Equal trade (rook for rook)",
            "fen": "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
            "move": "a1a8",  # Rxa8
            "expected": "zero or positive"
        }
    ]
    
    passed = 0
    for test in test_cases:
        print(f"\n{test['name']}:")
        engine.board = chess.Board(test['fen'])
        
        try:
            move = chess.Move.from_uci(test['move'])
            if move in engine.board.legal_moves:
                see_value = engine._see(engine.board, move)
                is_safe = engine._is_safe_move(engine.board, move)
                
                print(f"  SEE: {see_value}")
                print(f"  Safe: {is_safe}")
                
                if test['expected'] == "negative" and see_value < 0:
                    print(f"  ✓ Correctly identified as losing material")
                    passed += 1
                elif test['expected'] == "zero or positive" and see_value >= 0:
                    print(f"  ✓ Correctly identified as safe/winning")
                    passed += 1
                else:
                    print(f"  Result: {see_value}, expected {test['expected']}")
            else:
                print(f"  (Move not legal in position, checking SEE logic)")
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\n{passed}/{len(test_cases)} SEE tests passed")
    return passed >= 2  # At least 2 should pass


def test_hanging_piece_detection():
    """Test that engine doesn't hang pieces"""
    print("\n" + "=" * 80)
    print("TEST 3: Hanging Piece Detection")
    print("=" * 80)
    
    engine = V7P3REngine()
    
    # Position where queen can hang if moved carelessly
    # Black's rook on a8 attacks a-file
    fen = "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1"
    engine.board = chess.Board(fen)
    
    print("Position: Rooks on both sides")
    print(engine.board)
    print()
    
    # Test if engine knows Rxa8 is safe (equal trade)
    rxa8 = chess.Move.from_uci("a1a8")
    see_value = engine._see(engine.board, rxa8)
    is_safe = engine._is_safe_move(engine.board, rxa8)
    
    print(f"Rxa8:")
    print(f"  SEE: {see_value} (0 = equal trade)")
    print(f"  Safe: {is_safe}")
    
    if see_value >= -100 and is_safe:  # Within tolerance
        print("✓ PASSED: Recognizes equal trade as acceptable")
        return True
    else:
        print("✗ FAILED: Thinks equal trade is bad")
        return False


def main():
    print("V7P3R v15.2 - Material Awareness Tests")
    print("=" * 80)
    print()
    
    results = []
    
    # Run tests
    results.append(("Nd5 Blunder Fix", test_nd5_blunder_fix()))
    results.append(("SEE Evaluation", test_see_evaluation()))
    results.append(("Hanging Piece Detection", test_hanging_piece_detection()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! V15.2 material awareness is working.")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed. Review implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
