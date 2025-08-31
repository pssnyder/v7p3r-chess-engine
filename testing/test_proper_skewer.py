#!/usr/bin/env python3
"""
Test Proper Skewer Positions
"""

import chess
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from v7p3r_scoring_calculation import V7P3RScoringCalculationClean

def test_proper_rank_skewer():
    """Test a proper rank skewer where all pieces are on the same rank"""
    
    print("🎯 PROPER RANK SKEWER TEST")
    print("=" * 50)
    
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    scorer = V7P3RScoringCalculationClean(piece_values)
    
    # Proper rank skewer: all on same rank (5th rank)
    print("📍 Rank Skewer Position:")
    board = chess.Board("8/8/8/R1q1r3/8/8/8/K7 w - - 0 1")
    print("FEN: 8/8/8/R1q1r3/8/8/8/K7 w - - 0 1")
    print("White Rook a5, Black Queen c5, Black Rook e5 - ALL on rank 5")
    
    # Verify alignment
    rook_white = chess.A5
    queen_black = chess.C5
    rook_black = chess.E5
    
    print(f"White Rook: {chess.square_name(rook_white)} (rank {chess.square_rank(rook_white)})")
    print(f"Black Queen: {chess.square_name(queen_black)} (rank {chess.square_rank(queen_black)})")
    print(f"Black Rook: {chess.square_name(rook_black)} (rank {chess.square_rank(rook_black)})")
    
    # Test the skewer detection
    result = scorer._tactical_skewer_detection(board, chess.WHITE)
    print(f"Skewer detection result: {result:.2f}")
    
    # Manual trace through our method
    print(f"\n🔍 Manual Trace:")
    print(f"White rook attacks: {[chess.square_name(sq) for sq in board.attacks(rook_white)]}")
    print(f"Attacks queen? {queen_black in board.attacks(rook_white)}")
    
    # Test the behind-check method
    behind_result = scorer._check_for_skewer_behind(board, rook_white, queen_black, chess.WHITE)
    print(f"Behind-check result: {behind_result:.2f}")
    
    return result

def test_proper_file_skewer():
    """Test a proper file skewer where all pieces are on the same file"""
    
    print(f"\n🎯 PROPER FILE SKEWER TEST")
    print("=" * 50)
    
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    scorer = V7P3RScoringCalculationClean(piece_values)
    
    # File skewer: all on c-file
    print("📍 File Skewer Position:")
    board = chess.Board("8/8/2r5/2q5/2R5/8/8/K7 w - - 0 1")
    print("FEN: 8/8/2r5/2q5/2R5/8/8/K7 w - - 0 1")
    print("White Rook c4, Black Queen c5, Black Rook c6 - ALL on c-file")
    
    # Verify alignment
    rook_white = chess.C4
    queen_black = chess.C5
    rook_black = chess.C6
    
    print(f"White Rook: {chess.square_name(rook_white)} (file {chess.square_file(rook_white)})")
    print(f"Black Queen: {chess.square_name(queen_black)} (file {chess.square_file(queen_black)})")
    print(f"Black Rook: {chess.square_name(rook_black)} (file {chess.square_file(rook_black)})")
    
    # Test the skewer detection
    result = scorer._tactical_skewer_detection(board, chess.WHITE)
    print(f"Skewer detection result: {result:.2f}")
    
    # Test the behind-check method
    behind_result = scorer._check_for_skewer_behind(board, rook_white, queen_black, chess.WHITE)
    print(f"Behind-check result: {behind_result:.2f}")
    
    return result

def test_proper_diagonal_skewer():
    """Test a proper diagonal skewer"""
    
    print(f"\n🎯 PROPER DIAGONAL SKEWER TEST")
    print("=" * 50)
    
    piece_values = {
        chess.PAWN: 100,
        chess.KNIGHT: 320,
        chess.BISHOP: 330,
        chess.ROOK: 500,
        chess.QUEEN: 900,
        chess.KING: 20000
    }
    
    scorer = V7P3RScoringCalculationClean(piece_values)
    
    # Diagonal skewer: bishop skewering queen and rook
    print("📍 Diagonal Skewer Position:")
    board = chess.Board("8/8/8/4r3/3q4/2B5/8/K7 w - - 0 1")
    print("FEN: 8/8/8/4r3/3q4/2B5/8/K7 w - - 0 1")
    print("White Bishop c3, Black Queen d4, Black Rook e5 - diagonal")
    
    # Verify alignment
    bishop_white = chess.C3
    queen_black = chess.D4
    rook_black = chess.E5
    
    print(f"White Bishop: {chess.square_name(bishop_white)}")
    print(f"Black Queen: {chess.square_name(queen_black)}")
    print(f"Black Rook: {chess.square_name(rook_black)}")
    
    # Test the skewer detection
    result = scorer._tactical_skewer_detection(board, chess.WHITE)
    print(f"Skewer detection result: {result:.2f}")
    
    # Test the behind-check method
    behind_result = scorer._check_for_skewer_behind(board, bishop_white, queen_black, chess.WHITE)
    print(f"Behind-check result: {behind_result:.2f}")
    
    return result

if __name__ == "__main__":
    rank_result = test_proper_rank_skewer()
    file_result = test_proper_file_skewer()
    diagonal_result = test_proper_diagonal_skewer()
    
    print(f"\n📊 SUMMARY:")
    print(f"Rank skewer: {rank_result:.2f}")
    print(f"File skewer: {file_result:.2f}")
    print(f"Diagonal skewer: {diagonal_result:.2f}")
    
    if any([rank_result > 0, file_result > 0, diagonal_result > 0]):
        print("✅ At least one skewer working!")
    else:
        print("❌ All skewers returning 0 - bug in detection logic!")
