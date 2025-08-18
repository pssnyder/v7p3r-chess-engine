#!/usr/bin/env python3
"""
Test the check capture enhancement: 
When in check, prioritize safe captures of checking piece over king moves
"""

import chess
import sys
import os

# Add the parent directory to path to import our modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v7p3r_move_ordering import MoveOrdering
from v7p3r_mvv_lva import MVVLVA
from v7p3r_tempo import TempoCalculation

def test_check_capture_priority():
    """Test that captures of checking pieces get higher priority than king moves"""
    print("Testing check capture priority enhancement...")
    
    # Create components
    move_ordering = MoveOrdering()
    tempo = TempoCalculation()
    
    # Test case: Knight fork scenario - knight on f3 giving check to king on g1
    # White queen on d2 can capture the knight, or king can move
    board = chess.Board()
    board.clear()
    
    # Set up the critical position
    board.set_piece_at(chess.G1, chess.Piece(chess.KING, chess.WHITE))   # White king 
    board.set_piece_at(chess.F3, chess.Piece(chess.KNIGHT, chess.BLACK)) # Black knight giving check
    board.set_piece_at(chess.D2, chess.Piece(chess.QUEEN, chess.WHITE))  # White queen can capture
    board.set_piece_at(chess.E2, chess.Piece(chess.BISHOP, chess.WHITE)) # White bishop (potential fork target)
    # Add black king and pieces so it's not stalemate after capture
    board.set_piece_at(chess.G8, chess.Piece(chess.KING, chess.BLACK))   # Black king
    board.set_piece_at(chess.A7, chess.Piece(chess.PAWN, chess.BLACK))   # Black pawn
    
    board.turn = chess.WHITE
    
    # Verify position is in check
    print(f"Board is in check: {board.is_check()}")
    
    if board.is_check():
        legal_moves = list(board.legal_moves)
        print(f"Legal moves: {[str(move) for move in legal_moves]}")
        
        # Order the moves
        ordered_moves = move_ordering.order_moves(board, legal_moves)
        print(f"Ordered moves: {[str(move) for move in ordered_moves]}")
        
        # Check if capture of queen (by rook) is prioritized over king moves
        capture_moves = [move for move in ordered_moves if board.is_capture(move)]
        king_moves = [move for move in ordered_moves if board.piece_at(move.from_square).piece_type == chess.KING]
        
        print(f"Capture moves: {[str(move) for move in capture_moves]}")
        print(f"King moves: {[str(move) for move in king_moves]}")
        
        if capture_moves and king_moves:
            # Find first capture and first king move in ordered list
            capture_index = next((i for i, move in enumerate(ordered_moves) if move in capture_moves), len(ordered_moves))
            king_index = next((i for i, move in enumerate(ordered_moves) if move in king_moves), len(ordered_moves))
            
            print(f"First capture at index: {capture_index}")
            print(f"First king move at index: {king_index}")
            
            if capture_index < king_index:
                print("✓ SUCCESS: Capture of checking piece prioritized over king moves!")
                return True
            else:
                print("✗ FAIL: King moves still prioritized over capture")
                return False
        else:
            print("No captures or king moves found")
            return False
    else:
        print("Position is not in check - test setup failed")
        return False

def test_tempo_bonus():
    """Test that tempo calculation gives bonus for capturing checking piece"""
    print("\nTesting tempo bonus for check capture...")
    
    tempo = TempoCalculation()
    
    # Same knight fork test position - knight on f3 checking king on g1
    board = chess.Board()
    board.clear()
    
    board.set_piece_at(chess.G1, chess.Piece(chess.KING, chess.WHITE))   # White king 
    board.set_piece_at(chess.F3, chess.Piece(chess.KNIGHT, chess.BLACK)) # Black knight giving check
    board.set_piece_at(chess.E4, chess.Piece(chess.QUEEN, chess.WHITE))  # White queen can capture knight
    board.set_piece_at(chess.E2, chess.Piece(chess.BISHOP, chess.WHITE)) # White bishop
    # Add black king so it's not stalemate
    board.set_piece_at(chess.G8, chess.Piece(chess.KING, chess.BLACK))   # Black king
    board.set_piece_at(chess.A7, chess.Piece(chess.PAWN, chess.BLACK))   # Black pawn (can move)
    
    board.turn = chess.WHITE
    
    if board.is_check():
        # Test queen captures knight vs king moves
        capture_move = chess.Move(chess.E4, chess.F3)  # Queen captures knight
        king_move = chess.Move(chess.G1, chess.H1)     # King moves to h1
        
        if capture_move in board.legal_moves and king_move in board.legal_moves:
            capture_tempo, capture_critical = tempo.evaluate_tempo(board, capture_move, 4)
            king_tempo, king_critical = tempo.evaluate_tempo(board, king_move, 4)
            
            print(f"Capture move tempo: {capture_tempo}, critical: {capture_critical}")
            print(f"King move tempo: {king_tempo}, critical: {king_critical}")
            
            if capture_tempo > king_tempo:
                print("✓ SUCCESS: Capture gets higher tempo score!")
                return True
            else:
                print("✗ FAIL: King move still gets higher tempo")
                return False
        else:
            print("Moves not legal in position")
            return False
    else:
        print("Position not in check")
        return False

if __name__ == "__main__":
    print("V7P3R Chess Engine - Check Capture Enhancement Test")
    print("=" * 60)
    
    test1_pass = test_check_capture_priority()
    test2_pass = test_tempo_bonus()
    
    print("\n" + "=" * 60)
    if test1_pass and test2_pass:
        print("✓ ALL TESTS PASSED - Enhancement working correctly!")
    else:
        print("✗ SOME TESTS FAILED - Enhancement needs adjustment")
