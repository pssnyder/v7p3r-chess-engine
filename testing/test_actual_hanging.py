#!/usr/bin/env python3
"""
Test capture-aware evaluation with ACTUALLY hanging pieces
"""

import chess
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine


def test_actually_hanging_knight():
    """Test knight that's REALLY hanging"""
    print("\nTest: Knight ACTUALLY hanging (attacked by pawn)")
    print("="*60)
    
    engine = V7P3REngine()
    
    # Position: White knight on e5, black pawn on d6 can capture
    # (pawn attacks diagonally up-right from d6 to e5)
    board = chess.Board("rnbqkb1r/pppp1ppp/3p1n2/4N3/8/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1")
    engine.board = board
    
    print("Position: White knight on e5, black pawn on d6 attacks it")
    print(board)
    print()
    
    knight_square = chess.E5
    attacked_by_black = board.is_attacked_by(chess.BLACK, knight_square)
    defended_by_white = board.is_attacked_by(chess.WHITE, knight_square)
    
    print(f"Knight on e5:")
    print(f"  Attacked by BLACK: {attacked_by_black}")
    print(f"  Defended by WHITE: {defended_by_white}")
    print(f"  Is hanging: {attacked_by_black and not defended_by_white}")
    print()
    
    eval_hanging = engine._evaluate_position(board)
    print(f"Eval with hanging knight on e5: {eval_hanging:+.2f} cp")
    
    # Compare: knight moves away to safe square
    board2 = chess.Board("rnbqkb1r/pppp1ppp/3p1n2/8/8/4N3/PPPPPPPP/RNBQKB1R w KQkq - 0 1")
    engine.board = board2
    
    knight_square2 = chess.E3
    attacked2 = board2.is_attacked_by(chess.BLACK, knight_square2)
    defended2 = board2.is_attacked_by(chess.WHITE, knight_square2)
    
    print(f"Knight on e3:")
    print(f"  Attacked by BLACK: {attacked2}")
    print(f"  Defended by WHITE: {defended2}")
    print()
    
    eval_safe = engine._evaluate_position(board2)
    print(f"Eval with safe knight on e3: {eval_safe:+.2f} cp")
    print()
    
    if eval_safe > eval_hanging + 200:
        print("✓✓✓ Hanging knight has much worse eval")
    else:
        print(f"✗ Hanging knight eval only {eval_safe - eval_hanging:+.2f} cp worse")
    
    print()


def test_qxh7_actually_hanging():
    """Test that Qxh7 with queen hanging is correctly evaluated"""
    print("\nTest: Qxh7 with queen hanging")
    print("="*60)
    
    engine = V7P3REngine()
    
    # After Qxh7 (before Rxh7)
    board = chess.Board("r1bqkbnr/ppppp1pQ/2n2p2/8/1nP5/5N2/PP1P1PPP/RNB1KB1R b KQkq - 0 1")
    engine.board = board
    
    print("Position: After Qxh7 (queen on h7, rook on h8 can take)")
    print(board)
    print()
    
    queen_square = chess.H7
    attacked = board.is_attacked_by(chess.BLACK, queen_square)
    defended = board.is_attacked_by(chess.WHITE, queen_square)
    
    print(f"Queen on h7:")
    print(f"  Attacked by BLACK: {attacked}")
    print(f"  Defended by WHITE: {defended}")
    print(f"  Is hanging: {attacked and not defended}")
    print()
    
    # Eval from White's perspective (it's Black's turn)
    eval_qxh7 = engine._evaluate_position(board)
    print(f"Eval after Qxh7 (White perspective): {eval_qxh7:+.2f} cp")
    
    # After Rxh7
    board.push(chess.Move.from_uci('h8h7'))
    eval_rxh7 = engine._evaluate_position(board)
    board.pop()
    
    print(f"Eval after Rxh7 (White perspective): {eval_rxh7:+.2f} cp")
    print()
    
    if eval_qxh7 < -700:
        print("✓✓✓ Queen hanging recognized (very negative)")
    else:
        print(f"✗ Queen hanging not recognized well enough")
    
    print()


if __name__ == "__main__":
    print("="*60)
    print("V15.6 CAPTURE-AWARE: ACTUAL HANGING PIECE TESTS")
    print("="*60)
    
    test_actually_hanging_knight()
    test_qxh7_actually_hanging()
