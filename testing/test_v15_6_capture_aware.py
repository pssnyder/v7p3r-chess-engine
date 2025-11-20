#!/usr/bin/env python3
"""
Test V15.6 CAPTURE-AWARE evaluation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine


def test_hanging_piece_pst():
    """Test that hanging pieces don't contribute PST value"""
    print("\nV15.6 CAPTURE-AWARE: Hanging Piece PST Test")
    print("="*60)
    
    engine = V7P3REngine()
    
    # Position: White knight on great square (d5) but undefended and attacked
    board = chess.Board("rnbqkbnr/pppppppp/8/3N4/8/8/PPPPPPPP/R1BQKBNR w KQkq - 0 1")
    engine.board = board
    
    print("Position: White knight on d5 (great square) but HANGING")
    print(board)
    print()
    
    eval_score = engine._evaluate_position(board)
    print(f"Eval with hanging knight: {eval_score:+.2f} cp")
    
    # Compare to position where knight is defended
    board_defended = chess.Board("rnbqkbnr/pppppppp/8/3N4/8/2N5/PPPPPPPP/R1BQKB1R w KQkq - 0 1")
    engine.board = board_defended
    eval_defended = engine._evaluate_position(board_defended)
    print(f"Eval with defended knight: {eval_defended:+.2f} cp")
    print()
    
    if eval_defended > eval_score + 200:
        print("✓ Hanging knight PST not counted (defended is much better)")
    else:
        print(f"✗ Hanging knight may still have PST value")
    
    print()


def test_qxh7_with_capture_aware():
    """Test Qxh7 position with capture-aware eval"""
    print("\nV15.6 CAPTURE-AWARE: Qxh7 Test")
    print("="*60)
    
    engine = V7P3REngine()
    
    # Position after 3...Nb4 where v15.6 played Qxh7??
    board = chess.Board("r1bqkbnr/ppppp1p1/2n2p2/8/1nP5/5N2/PPQP1PPP/RNB1KB1R w KQkq - 0 1")
    engine.board = board
    
    print("Position: After 3...Nb4")
    print(board)
    print()
    
    # Check eval before Qxh7
    eval_before = engine._evaluate_position(board)
    print(f"Current position: {eval_before:+.2f} cp")
    
    # After Qxh7, queen is hanging on h7
    board.push(chess.Move.from_uci('c2h7'))
    eval_qxh7 = engine._evaluate_position(board)
    print(f"After Qxh7 (queen hanging): {eval_qxh7:+.2f} cp")
    board.pop()
    
    # After Qxh7 Rxh7, queen is captured
    board.push(chess.Move.from_uci('c2h7'))
    board.push(chess.Move.from_uci('h8h7'))
    eval_lost_queen = engine._evaluate_position(board)
    board.pop()
    board.pop()
    print(f"After Qxh7 Rxh7 (queen lost): {eval_lost_queen:+.2f} cp")
    print()
    
    if eval_qxh7 < eval_before - 700:
        print("✓ Queen on h7 recognized as hanging (very negative)")
    else:
        print(f"⚠ Queen on h7 may not be recognized as hanging")
    
    # Get best move
    print("\nSearching for best move...")
    move = engine.get_best_move(time_left=5, increment=0)
    print(f"Engine chose: {move.uci() if move else 'None'}")
    
    if move and move.uci() != 'c2h7':
        print("✓✓✓ DID NOT PLAY Qxh7 ✓✓✓")
    else:
        print("✗✗✗ STILL PLAYS Qxh7 ✗✗✗")
    
    print()


def test_safe_piece_advancement():
    """Test that safe piece advancement is valued"""
    print("\nV15.6 CAPTURE-AWARE: Safe Piece Advancement Test")
    print("="*60)
    
    engine = V7P3REngine()
    
    # Position: White can advance knight safely
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    engine.board = board
    
    eval_start = engine._evaluate_position(board)
    print(f"Starting position: {eval_start:+.2f} cp")
    
    # After Nf3 (safe development)
    board.push(chess.Move.from_uci('g1f3'))
    eval_nf3 = engine._evaluate_position(board)
    board.pop()
    
    print(f"After Nf3 (safe): {eval_nf3:+.2f} cp")
    
    if eval_nf3 > eval_start + 20:
        print("✓ Safe piece development valued")
    else:
        print(f"⚠ Safe piece development may not be valued enough")
    
    print()


if __name__ == "__main__":
    print("="*60)
    print("V7P3R v15.6 CAPTURE-AWARE EVALUATION TEST")
    print("="*60)
    print("\nTesting: Hanging pieces contribute ZERO PST value")
    print()
    
    try:
        test_hanging_piece_pst()
        test_qxh7_with_capture_aware()
        test_safe_piece_advancement()
        
        print("="*60)
        print("CAPTURE-AWARE EVALUATION:")
        print("- Hanging pieces have PST = 0")
        print("- Safe pieces have full PST value")
        print("- Material balance tracks all pieces")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
