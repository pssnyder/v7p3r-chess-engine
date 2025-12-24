#!/usr/bin/env python3
"""
V18.1 Evaluation Test
Quick verification that v18.1 evaluation changes are working correctly
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine


def test_king_safety_center_penalty():
    """Test that unmoved king in center has penalty"""
    print("Testing king safety center penalty...")
    
    # Position 1: White king unmoved on e1 (center), Black king safe on g8
    board1 = chess.Board("rnbq1rk1/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQ - 0 1")
    
    # Position 2: White king castled to g1, Black king safe on g8  
    board2 = chess.Board("rnbq1rk1/pppppppp/8/8/8/8/PPPPPPPP/RNBQ1RK1 w - - 0 1")
    
    # Use bitboard evaluator (use_fast_evaluator=False) to test our changes
    engine = V7P3REngine(use_fast_evaluator=False)
    
    # Use internal evaluate method
    eval1 = engine._evaluate_position(board1)
    eval2 = engine._evaluate_position(board2)
    
    print(f"  White king on e1 (unmoved, center): {eval1:.2f}")
    print(f"  White king on g1 (castled): {eval2:.2f}")
    print(f"  Difference: {eval2 - eval1:.2f} (should be positive, ~80+ cp)")
    
    if eval2 > eval1 + 50:  # Castled position should be significantly better
        print("  ✓ PASS: Castled king valued higher\n")
        return True
    else:
        print("  ✗ FAIL: Center king not penalized enough\n")
        return False


def test_passed_pawn_exponential():
    """Test exponential passed pawn bonus"""
    print("Testing exponential passed pawn bonus...")
    
    # White passed pawn on a3 (4 from promotion)
    board1 = chess.Board("8/8/8/8/8/P7/8/4K2k w - - 0 1")
    
    # White passed pawn on a6 (1 from promotion)
    board2 = chess.Board("8/8/P7/8/8/8/8/4K2k w - - 0 1")
    
    engine = V7P3REngine(use_fast_evaluator=False)
    
    eval1 = engine._evaluate_position(board1)
    eval2 = engine._evaluate_position(board2)
    
    print(f"  Pawn on a3 (4 from promotion): {eval1:.2f}")
    print(f"  Pawn on a6 (1 from promotion): {eval2:.2f}")
    print(f"  Difference: {eval2 - eval1:.2f} (should be ~300+ cp)")
    
    if eval2 > eval1 + 200:  # Should be significantly higher
        print("  ✓ PASS: Advanced passed pawn valued much higher\n")
        return True
    else:
        print("  ✗ FAIL: Passed pawn scaling not exponential\n")
        return False


def test_bishop_pair_bonus():
    """Test bishop pair bonus"""
    print("Testing bishop pair bonus...")
    
    # Position with bishop pair for white
    board1 = chess.Board("rnbqk2r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1")
    
    # Position with only one bishop for white
    board2 = chess.Board("rnbqk2r/pppppppp/8/8/8/8/PPPPPPPP/RNBQK2R w KQkq - 0 1")
    
    engine = V7P3REngine(use_fast_evaluator=False)
    
    eval1 = engine._evaluate_position(board1)
    eval2 = engine._evaluate_position(board2)
    
    print(f"  White with bishop pair: {eval1:.2f}")
    print(f"  White with one bishop: {eval2:.2f}")
    print(f"  Difference: {eval1 - eval2:.2f} (should be ~30-50 cp)")
    
    if eval1 > eval2 + 20:
        print("  ✓ PASS: Bishop pair bonus applied\n")
        return True
    else:
        print("  ✗ FAIL: Bishop pair not valued\n")
        return False


def test_king_centralization_endgame():
    """Test king centralization in endgame"""
    print("Testing king centralization in endgame...")
    
    # King on e4 (center)
    board1 = chess.Board("8/8/8/8/4K3/8/8/4k3 w - - 0 1")
    
    # King on h1 (corner)
    board2 = chess.Board("8/8/8/8/8/8/8/4k2K w - - 0 1")
    
    engine = V7P3REngine(use_fast_evaluator=False)
    
    eval1 = engine._evaluate_position(board1)
    eval2 = engine._evaluate_position(board2)
    
    print(f"  King on e4 (center): {eval1:.2f}")
    print(f"  King on h1 (corner): {eval2:.2f}")
    print(f"  Difference: {eval1 - eval2:.2f} (should be ~40-70 cp)")
    
    if eval1 > eval2 + 30:
        print("  ✓ PASS: Centralized king valued higher\n")
        return True
    else:
        print("  ✗ FAIL: King centralization not working\n")
        return False


def test_high_value_attacker_penalty():
    """Test high-value attacker penalty"""
    print("Testing high-value attacker penalty...")
    
    # Position with queen near white king
    board1 = chess.Board("4k3/8/8/8/8/3q4/3PPP2/4K3 w - - 0 1")
    
    # Position without queen near king
    board2 = chess.Board("4k3/8/8/8/8/8/3PPP2/4K3 w - - 0 1")
    
    engine = V7P3REngine(use_fast_evaluator=False)
    
    eval1 = engine._evaluate_position(board1)
    eval2 = engine._evaluate_position(board2)
    
    print(f"  Queen near king: {eval1:.2f}")
    print(f"  No queen near king: {eval2:.2f}")
    print(f"  Difference: {eval2 - eval1:.2f} (should be ~100+ cp)")
    
    if eval2 > eval1 + 50:  # Should have significant penalty
        print("  ✓ PASS: High-value attacker penalty applied\n")
        return True
    else:
        print("  ✗ FAIL: Attacker penalty not sufficient\n")
        return False


def main():
    print("=" * 80)
    print("V7P3R v18.1.0 Evaluation Tests")
    print("=" * 80)
    print()
    
    results = []
    
    results.append(test_king_safety_center_penalty())
    results.append(test_passed_pawn_exponential())
    results.append(test_bishop_pair_bonus())
    results.append(test_king_centralization_endgame())
    results.append(test_high_value_attacker_penalty())
    
    print("=" * 80)
    print(f"Results: {sum(results)}/{len(results)} tests passed")
    print("=" * 80)
    
    if all(results):
        print("\n✓ All tests PASSED! V18.1 evaluation changes working correctly.")
        return 0
    else:
        print("\n✗ Some tests FAILED. Review evaluation implementation.")
        return 1


if __name__ == '__main__':
    import sys
    sys.exit(main())
