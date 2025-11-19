#!/usr/bin/env python3
"""
Quick test for V15.6 REVISED - Check queen sacrifice prevention
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine


def test_qxh7_blunder():
    """Test that engine doesn't play Qxh7 sacrificing queen"""
    print("\nV15.6 REVISED: Qxh7 Sacrifice Prevention Test")
    print("="*60)
    
    engine = V7P3REngine()
    
    # Position after 1.c4 Nc6 2.Nf3 f6 3.Qc2 Nb4
    # V15.6 initial played 4.Qxh7?? here
    board = chess.Board("r1bqkbnr/ppppp1p1/2n2p2/8/1nP5/5N2/PPQP1PPP/RNB1KB1R w KQkq - 0 1")
    engine.board = board
    
    print(f"Position: After 3...Nb4")
    print(board)
    print()
    
    # Check evaluation BEFORE Qxh7
    eval_before = engine._evaluate_position(board)
    print(f"Current position eval: {eval_before:+.2f} cp")
    
    # Test what happens AFTER Qxh7 Rxh7 (queen is lost)
    board.push(chess.Move.from_uci('c2h7'))
    print(f"After Qxh7 (queen on h7): {engine._evaluate_position(board):+.2f} cp")
    board.push(chess.Move.from_uci('h8h7'))  # Rxh7
    eval_after_rxh7 = engine._evaluate_position(board)
    board.pop()
    board.pop()
    
    print(f"After Qxh7 Rxh7 (queen lost): {eval_after_rxh7:+.2f} cp")
    print()
    
    if eval_after_rxh7 < -500:
        print("✓ Material evaluation working - losing queen scores very badly")
    else:
        print(f"✗ PROBLEM: After losing queen only scores {eval_after_rxh7:+.2f} (should be < -500)")
    
    # Now get best move
    print("\nSearching for best move...")
    move = engine.get_best_move(time_left=5, increment=0)
    print(f"Engine chose: {move.uci() if move else 'None'} ({board.san(move) if move else 'None'})")
    
    if move and move.uci() != 'c2h7':
        print("✓✓✓ ENGINE DID NOT SACRIFICE QUEEN ✓✓✓")
        print(f"Chose {move.uci()} instead")
    else:
        print("✗✗✗ ENGINE STILL SACRIFICES QUEEN ✗✗✗")
    
    print()


def test_material_dominance():
    """Test that material dominates evaluation"""
    print("\nV15.6 REVISED: Material Dominance Test")
    print("="*60)
    
    engine = V7P3REngine()
    
    # Position: White down queen but has slightly better PST
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/3P4/PPP1PPPP/RNB1KBNR w KQkq - 0 1")
    engine.board = board
    
    eval_score = engine._evaluate_position(board)
    print(f"Down queen (with PST) eval: {eval_score:+.2f} cp")
    
    if eval_score < -500:
        print("✓ Material dominates - negative despite PST")
    else:
        print(f"✗ PST overriding material - should be < -500 cp")
    
    # Position: White up queen
    board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    engine.board = board
    
    eval_score = engine._evaluate_position(board)
    print(f"Up queen eval: {eval_score:+.2f} cp")
    
    if eval_score > 500:
        print("✓ Material dominates - positive for queen advantage")
    else:
        print(f"✗ Material not strong enough - should be > 500 cp")
    
    print()


if __name__ == "__main__":
    print("="*60)
    print("V7P3R v15.6 REVISED MATERIAL TEST")
    print("="*60)
    print("Testing 70% material / 30% PST blend")
    print()
    
    try:
        test_qxh7_blunder()
        test_material_dominance()
        
        print("="*60)
        print("If tests pass, V15.6 REVISED should prevent queen sacrifices")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
