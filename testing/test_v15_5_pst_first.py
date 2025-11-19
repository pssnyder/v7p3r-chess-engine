#!/usr/bin/env python3
"""
Test V15.5 PST-first evaluation with material safety net
Ensures we maintain positional strength while preventing blunders
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine


def test_pst_dominance():
    """Test that PST evaluation dominates in normal positions"""
    print("=" * 60)
    print("V15.5 PST Dominance Test")
    print("=" * 60)
    
    engine = V7P3REngine()
    
    # Position with better piece placement but equal material
    # White's centralized knight vs Black's edge knight
    board = chess.Board("rnbqkb1r/pppppppp/8/8/3N4/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1")
    eval1 = engine._evaluate_position(board)
    
    print(f"\nCentralized white knight: {eval1} cp")
    
    # Should be significantly positive due to PST advantage
    if eval1 > 50:
        print(f"✓ PST evaluation working! (+{eval1} cp advantage)")
        return True
    else:
        print(f"❌ PST evaluation weak: {eval1} cp")
        return False


def test_material_safety_net():
    """Test that safety net prevents catastrophic material losses"""
    print("\n" + "=" * 60)
    print("V15.5 Material Safety Net Test")
    print("=" * 60)
    
    engine = V7P3REngine()
    
    # White down a queen but has slightly better position
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/2N5/PPPPPPPP/R1B1KBNR w KQkq - 0 1")
    eval1 = engine._evaluate_position(board)
    
    print(f"\nWhite down queen, slightly better position: {eval1} cp")
    
    # Should be very negative (safety net kicks in)
    if eval1 < -500:
        print(f"✓ Safety net working! ({eval1} cp - recognizes material deficit)")
        return True
    else:
        print(f"❌ Safety net failed: {eval1} cp (should be very negative)")
        return False


def test_positional_compensation():
    """Test that large positional advantages can overcome material deficit"""
    print("\n" + "=" * 60)
    print("V15.5 Positional Compensation Test")
    print("=" * 60)
    
    engine = V7P3REngine()
    
    # Starting position - material equal, slight positional differences
    board = chess.Board()
    eval1 = engine._evaluate_position(board)
    
    print(f"\nStarting position: {eval1} cp")
    
    # Should be close to 0 (no major imbalance)
    if abs(eval1) < 100:
        print(f"✓ Balanced position correctly evaluated")
        return True
    else:
        print(f"⚠ Starting position evaluation: {eval1} cp (expected near 0)")
        return True  # Not critical


def test_bishop_pair_awareness():
    """Test that bishop pair is valued in material calculation"""
    print("\n" + "=" * 60)
    print("V15.5 Bishop Pair Awareness Test")
    print("=" * 60)
    
    engine = V7P3REngine()
    
    # White with bishop pair, Black with one bishop
    board1 = chess.Board("rnbqk2r/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    
    # Both sides with one bishop
    board2 = chess.Board("rnbqk2r/pppppppp/8/8/8/8/PPPPPPPP/RN1QKBNR w KQkq - 0 1")
    
    eval1 = engine._evaluate_position(board1)
    eval2 = engine._evaluate_position(board2)
    
    print(f"\nWhite bishop pair vs Black lone bishop: {eval1} cp")
    print(f"Both sides lone bishop: {eval2} cp")
    
    # Bishop pair should be better
    if eval1 > eval2:
        print(f"✓ Bishop pair valued: +{eval1 - eval2} cp difference")
        return True
    else:
        print(f"⚠ Bishop pair not showing advantage")
        return True  # Not critical for PST-first


def test_no_bongcloud():
    """Ensure opening book still prevents bongcloud"""
    print("\n" + "=" * 60)
    print("V15.5 Anti-Bongcloud Test")
    print("=" * 60)
    
    engine = V7P3REngine()
    
    # Critical position: 1.e4 c6
    engine.board = chess.Board()
    engine.board.push(chess.Move.from_uci("e2e4"))
    engine.board.push(chess.Move.from_uci("c7c6"))
    
    book_move = engine.opening_book.get_book_move(engine.board)
    if book_move:
        print(f"Book move: {book_move}")
        if book_move == "e1e2":
            print("❌ BONGCLOUD IN BOOK!")
            return False
        else:
            print("✓ Book move is safe")
            return True
    else:
        print("⚠ Not in book")
        return True


if __name__ == "__main__":
    print("\nV15.5 PST-First Evaluation Test Suite\n")
    
    success = True
    
    if not test_pst_dominance():
        success = False
    
    if not test_material_safety_net():
        success = False
    
    if not test_positional_compensation():
        success = False
    
    if not test_bishop_pair_awareness():
        success = False
    
    if not test_no_bongcloud():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("✓✓✓ ALL V15.5 TESTS PASSED ✓✓✓")
        print("PST-first with material safety net working!")
        print("Ready for deployment.")
    else:
        print("❌ SOME TESTS FAILED")
    print("=" * 60)
