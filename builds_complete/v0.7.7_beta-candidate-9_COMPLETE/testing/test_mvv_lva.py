#!/usr/bin/env python3
"""
Test the updated _piece_captures function with MVV-LVA
"""
import chess
import sys
import os

# Add the v7p3r_engine path to import modules
sys.path.append(os.path.join(os.path.dirname(__file__), 'v7p3r_engine'))

def test_mvv_lva_captures():
    print("Testing Updated MVV-LVA Piece Captures Function")
    print("=" * 50)
    
    from v7p3r import v7p3rEngine
    
    # Create engine
    engine = v7p3rEngine()
    scoring_calc = engine.scoring_calculator
    
    # Test position with various captures available
    test_positions = [
        # Standard opening position
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        
        # Position with pawn captures
        "rnbqkbnr/ppp1pppp/8/3p4/4P3/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 2",
        
        # Position with piece trades available
        "rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3",
        
        # Complex tactical position
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 4"
    ]
    
    for i, fen in enumerate(test_positions):
        print(f"\nTest Position {i+1}: {fen}")
        board = chess.Board(fen)
        
        # Test for white
        white_score = scoring_calc._piece_captures(board, chess.WHITE)
        print(f"White capture score: {white_score:.3f}")
        
        # Test for black  
        black_score = scoring_calc._piece_captures(board, chess.BLACK)
        print(f"Black capture score: {black_score:.3f}")
        
        # Show available captures
        captures = [move for move in board.legal_moves if board.is_capture(move)]
        if captures:
            print(f"Available captures: {[move.uci() for move in captures[:5]]}")  # Show first 5
        else:
            print("No captures available")

if __name__ == "__main__":
    test_mvv_lva_captures()
