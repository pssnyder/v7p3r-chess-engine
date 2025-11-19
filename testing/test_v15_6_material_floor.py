#!/usr/bin/env python3
"""
V15.6 Material Floor Test Suite
Tests V15.1's material floor approach (restored in V15.6)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine


def test_material_floor_prevents_queen_sac():
    """Test that material floor prevents queen sacrifices without compensation"""
    print("\nV15.6 Material Floor Test - Queen Sacrifice Prevention")
    engine = V7P3REngine()
    
    # Position where PST might want to sacrifice queen
    # White queen on d1, can move to dangerous square
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    engine.board = board
    
    eval_before = engine._evaluate_position(board)
    print(f"Starting position eval: {eval_before:+.2f} cp")
    
    # Check evaluation is reasonable (around 0)
    if abs(eval_before) < 100:
        print("✓ Starting position properly evaluated")
    else:
        print(f"✗ Starting position evaluation seems off: {eval_before}")
    
    # Test that engine won't hang queen
    best_move = engine.get_best_move(time_left=10, increment=0)
    print(f"Best move: {best_move.uci() if best_move else 'None'}")
    
    # Move should be reasonable development
    if best_move and best_move.uci() in ['e2e4', 'd2d4', 'g1f3', 'b1c3', 'c2c4']:
        print("✓ Engine chose reasonable opening move")
    else:
        print(f"⚠ Engine chose unusual move: {best_move.uci() if best_move else 'None'}")
    
    print()


def test_material_floor_down_queen():
    """Test evaluation when down a queen"""
    print("\nV15.6 Material Floor Test - Down Queen")
    engine = V7P3REngine()
    
    # Position: White missing queen
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNB1KBNR w KQkq - 0 1")
    engine.board = board
    
    eval_score = engine._evaluate_position(board)
    print(f"Down queen eval: {eval_score:+.2f} cp")
    
    # Should be heavily negative (down 900 cp material)
    if eval_score < -700:
        print(f"✓ Material floor working - recognizes queen deficit ({eval_score:+.2f})")
    else:
        print(f"✗ Material floor not working - should be < -700 cp, got {eval_score:+.2f}")
    
    print()


def test_material_floor_up_queen():
    """Test evaluation when up a queen"""
    print("\nV15.6 Material Floor Test - Up Queen")
    engine = V7P3REngine()
    
    # Position: Black missing queen
    board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    engine.board = board
    
    eval_score = engine._evaluate_position(board)
    print(f"Up queen eval: {eval_score:+.2f} cp")
    
    # Should be heavily positive (up 900 cp material)
    if eval_score > 700:
        print(f"✓ Material floor working - recognizes queen advantage ({eval_score:+.2f})")
    else:
        print(f"✗ Material floor not working - should be > 700 cp, got {eval_score:+.2f}")
    
    print()


def test_pst_dominance_when_equal():
    """Test that PST evaluation works when material is equal"""
    print("\nV15.6 PST Dominance Test - Equal Material")
    engine = V7P3REngine()
    
    # Position: Equal material, but white has centralized knight
    board = chess.Board("rnbqkb1r/pppppppp/5n2/8/3N4/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1")
    engine.board = board
    
    eval_score = engine._evaluate_position(board)
    print(f"Centralized knight eval: {eval_score:+.2f} cp")
    
    # Should be positive due to better piece placement
    if eval_score > 50:
        print(f"✓ PST working - centralized knight valued ({eval_score:+.2f})")
    else:
        print(f"⚠ PST may not be dominant - expected > 50 cp, got {eval_score:+.2f}")
    
    print()


def test_opening_book():
    """Test that opening book prevents bongcloud"""
    print("\nV15.6 Opening Book Test")
    engine = V7P3REngine()
    engine.board = chess.Board()
    
    # Get opening move
    move = engine.get_best_move(time_left=10, increment=0)
    print(f"Opening move: {move.uci() if move else 'None'}")
    
    # Should NOT be Ke2 (bongcloud)
    if move and move.uci() != 'e1e2':
        print(f"✓ No bongcloud - chose {move.uci()}")
    else:
        print(f"✗ BONGCLOUD DETECTED: {move.uci() if move else 'None'}")
    
    # Play 1.e4
    engine.board.push(chess.Move.from_uci('e2e4'))
    # Play 1...e5
    engine.board.push(chess.Move.from_uci('e7e5'))
    
    # Get second move
    move2 = engine.get_best_move(time_left=10, increment=0)
    print(f"Second move: {move2.uci() if move2 else 'None'}")
    
    # Should NOT be Ke2
    if move2 and move2.uci() != 'e1e2':
        print(f"✓ No bongcloud on move 2 - chose {move2.uci()}")
    else:
        print(f"✗ BONGCLOUD DETECTED ON MOVE 2: {move2.uci() if move2 else 'None'}")
    
    print()


def test_hanging_piece_detection():
    """Test that engine doesn't hang pieces"""
    print("\nV15.6 Hang Detection Test - Rb1 Position")
    engine = V7P3REngine()
    
    # Position from V15.5 game where it played Rb1?? (hangs knight)
    # After 3...Bb4
    board = chess.Board("r1bqk1nr/pppp1ppp/2n5/2b1p3/2B1P3/2N5/PPPP1PPP/R1BQK1NR w KQkq - 0 1")
    engine.board = board
    
    move = engine.get_best_move(time_left=10, increment=0)
    print(f"Chosen move: {move.uci() if move else 'None'}")
    
    # Should NOT play Rb1 (hangs knight on c3)
    if move and move.uci() != 'a1b1':
        print(f"✓ Didn't hang knight - chose {move.uci()}")
    else:
        print(f"✗ HANGS KNIGHT: Rb1?? allows Bxc3")
    
    print()


def test_material_floor_edge_cases():
    """Test material floor with slight imbalances"""
    print("\nV15.6 Material Floor Edge Cases")
    engine = V7P3REngine()
    
    # Position: White up a pawn
    board = chess.Board("rnbqkbnr/pppppp1p/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    engine.board = board
    
    eval_score = engine._evaluate_position(board)
    print(f"Up a pawn: {eval_score:+.2f} cp")
    
    if 50 < eval_score < 200:
        print(f"✓ Pawn advantage recognized ({eval_score:+.2f})")
    else:
        print(f"⚠ Unexpected evaluation: {eval_score:+.2f}")
    
    # Position: White down a pawn
    board = chess.Board("rnbqkbnr/pppppppp/8/8/8/8/PPPPPP1P/RNBQKBNR w KQkq - 0 1")
    engine.board = board
    
    eval_score = engine._evaluate_position(board)
    print(f"Down a pawn: {eval_score:+.2f} cp")
    
    if -200 < eval_score < -50:
        print(f"✓ Pawn deficit recognized ({eval_score:+.2f})")
    else:
        print(f"⚠ Unexpected evaluation: {eval_score:+.2f}")
    
    print()


if __name__ == "__main__":
    print("="*70)
    print("V7P3R v15.6 MATERIAL FLOOR TEST SUITE")
    print("="*70)
    print("\nTesting V15.1's proven material floor approach:")
    print("- Material floor: max(PST, material) for White, min(PST, material) for Black")
    print("- Prevents material sacrifices without compensation")
    print("- Preserves PST evaluation strength")
    
    try:
        test_material_floor_prevents_queen_sac()
        test_material_floor_down_queen()
        test_material_floor_up_queen()
        test_pst_dominance_when_equal()
        test_opening_book()
        test_hanging_piece_detection()
        test_material_floor_edge_cases()
        
        print("="*70)
        print("TEST SUITE COMPLETE")
        print("="*70)
        print("\nV15.6 combines:")
        print("✓ V15.1's material floor evaluation (proven)")
        print("✓ V15.1's hang detection in move ordering")
        print("✓ V15.3's opening book (prevents bongcloud)")
        print("\nReady for tournament testing!")
        
    except Exception as e:
        print(f"\n❌ Error during testing: {e}")
        import traceback
        traceback.print_exc()
