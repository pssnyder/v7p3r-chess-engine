#!/usr/bin/env python3
"""
Corrected Tactical Pattern Test
Tests pin, fork, and skewer detection with CORRECT positions
"""

import chess
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from v7p3r_scoring_calculation import V7P3RScoringCalculationClean

def test_correct_tactical_patterns():
    """Test tactical patterns with verified correct positions"""
    
    print("🔍 CORRECTED TACTICAL PATTERN TESTS")
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
    
    # Test 1: Knight Fork (verified working)
    print("\n🏇 TEST 1: Knight Fork")
    board = chess.Board("8/8/3k1r2/8/4N3/8/8/4K3 w - - 0 1")
    print("Position: Knight on e4, Black King on d6, Black Rook on f6")
    print("Knight attacks both king and rook - verified fork!")
    
    fork_score = scorer._tactical_fork_detection(board, chess.WHITE)
    print(f"Fork detection score: {fork_score:.2f}")
    
    # Test 2: Pin - Rook pins knight to king on same file
    print("\n📌 TEST 2: File Pin")
    board = chess.Board("8/8/8/2k5/2n5/8/2R5/K7 w - - 0 1")
    print("Position: White Rook on c2, Black Knight on c4, Black King on c5")
    print("Rook pins knight to king on c-file!")
    
    pin_score = scorer._tactical_pin_detection(board, chess.WHITE)
    print(f"Pin detection score: {pin_score:.2f}")
    
    # Test 3: Diagonal Pin - Bishop pins knight to king
    print("\n⚡ TEST 3: Diagonal Pin")
    board = chess.Board("8/8/2k5/3n4/4B3/8/8/K7 w - - 0 1")
    print("Position: Bishop on e4, Knight on d5, King on c6")
    print("Bishop pins knight to king on long diagonal!")
    
    diag_pin_score = scorer._tactical_pin_detection(board, chess.WHITE)
    print(f"Diagonal pin score: {diag_pin_score:.2f}")
    
    # Test 4: Skewer - Queen in front, rook behind on same rank
    print("\n🎯 TEST 4: Rank Skewer")
    board = chess.Board("8/8/8/R1q1r3/8/8/8/K7 w - - 0 1")
    print("Position: White Rook on a5, Black Queen on c5, Black Rook on e5")
    print("All pieces on rank 5 - Rook attacks queen which must move, exposing rook behind!")
    
    skewer_score = scorer._tactical_skewer_detection(board, chess.WHITE)
    print(f"Skewer detection score: {skewer_score:.2f}")
    
    # Test 5: Knight Fork - Two high value pieces
    print("\n🐎 TEST 5: Knight Fork (Queen + Rook)")
    board = chess.Board("8/8/2q1r3/8/3N4/8/8/K7 w - - 0 1")
    print("Position: Knight on d4, Queen on c6, Rook on e6")
    print("Knight on d4 attacks both c6 and e6!")
    
    multi_fork_score = scorer._tactical_fork_detection(board, chess.WHITE)
    print(f"Queen+Rook fork score: {multi_fork_score:.2f}")
    
    # Test 6: Royal Fork - Knight forks king and queen
    print("\n👑 TEST 6: Royal Fork")
    board = chess.Board("8/8/3k4/2q5/4N3/8/8/K7 w - - 0 1")
    print("Position: Knight on e4, Black King on d6, Black Queen on c5")
    print("Knight forks king and queen - most valuable fork!")
    
    royal_fork_score = scorer._tactical_fork_detection(board, chess.WHITE)
    print(f"Royal fork score: {royal_fork_score:.2f}")
    
    print(f"\n📊 CORRECTED TACTICAL PATTERN SUMMARY:")
    print(f"✅ Knight Fork: {fork_score:.2f}")
    print(f"✅ File Pin: {pin_score:.2f}")
    print(f"✅ Diagonal Pin: {diag_pin_score:.2f}")
    print(f"✅ Rank Skewer: {skewer_score:.2f}")
    print(f"✅ Q+R Fork: {multi_fork_score:.2f}")
    print(f"✅ Royal Fork: {royal_fork_score:.2f}")
    
    total_tactical = fork_score + pin_score + diag_pin_score + skewer_score + multi_fork_score + royal_fork_score
    print(f"\n🎯 Total Tactical Score: {total_tactical:.2f}")
    
    if total_tactical > 100:
        print("🏆 EXCELLENT: All tactical patterns working great!")
    elif total_tactical > 50:
        print("✅ GOOD: Most tactical patterns working!")
    else:
        print("⚠️  Some tactical patterns need debugging")
    
    return total_tactical

def debug_individual_methods():
    """Debug individual detection methods"""
    print(f"\n🔧 DEBUGGING INDIVIDUAL METHODS")
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
    
    # Debug pin detection with simple position
    print("🔍 Debugging Pin Detection:")
    board = chess.Board("8/8/8/2k5/2n5/8/2R5/K7 w - - 0 1")
    print("Position: Rook c2, Knight c4, King c5")
    
    # Check rook attacks
    rook_square = chess.C2
    rook_attacks = board.attacks(rook_square)
    print(f"Rook on c2 attacks: {[chess.square_name(sq) for sq in rook_attacks]}")
    
    # Check if knight is attacked
    knight_square = chess.C4
    print(f"Knight on c4 is attacked by rook: {knight_square in rook_attacks}")
    
    # Check behind knight
    king_square = chess.C5
    print(f"King on c5")
    
    # Test our pin detection method
    pin_score = scorer._detect_pin_opportunities(board, rook_square, chess.WHITE)
    print(f"Pin detection result: {pin_score:.2f}")

if __name__ == "__main__":
    total = test_correct_tactical_patterns()
    if total < 50:
        debug_individual_methods()
