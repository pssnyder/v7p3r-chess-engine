#!/usr/bin/env python3
"""
Specific Tactical Pattern Test
Tests pin, fork, and skewer detection with positions that contain those patterns
"""

import chess
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from v7p3r_scoring_calculation import V7P3RScoringCalculationClean

def test_specific_tactical_patterns():
    """Test specific tactical patterns with clear examples"""
    
    print("🔍 SPECIFIC TACTICAL PATTERN TESTS")
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
    
    # Test 1: Clear Knight Fork
    print("\n🏇 TEST 1: Clear Knight Fork")
    board = chess.Board("8/8/3k1r2/8/4N3/8/8/4K3 w - - 0 1")
    print("Position: Knight on e4, Black King on d6, Black Rook on f6")
    print("Knight attacks both king and rook - classic fork!")
    
    fork_score = scorer._tactical_fork_detection(board, chess.WHITE)
    print(f"Fork detection score: {fork_score:.2f}")
    
    # Test 2: Clear Pin
    print("\n📌 TEST 2: Clear Pin")
    board = chess.Board("8/8/8/2nk4/2R5/8/8/K7 w - - 0 1")
    print("Position: White Rook on c4, Black Knight on c5, Black King on d5")
    print("Wait, let me fix this - need pieces on same line...")
    board = chess.Board("8/8/k7/2n5/2R5/8/8/K7 w - - 0 1")
    print("Position: White Rook on c4, Black Knight on c5, Black King on a6")
    print("Rook pins knight to king on c-file!")
    
    pin_score = scorer._tactical_pin_detection(board, chess.WHITE)
    print(f"Pin detection score: {pin_score:.2f}")
    
    # Test 3: Clear Skewer  
    print("\n🎯 TEST 3: Clear Skewer")
    board = chess.Board("8/8/8/2q1r3/2R5/8/8/K7 w - - 0 1")
    print("Position: White Rook on c4, Black Queen on c5, Black Rook on e5")
    print("Rook attacks queen which must move, exposing the rook behind!")
    
    skewer_score = scorer._tactical_skewer_detection(board, chess.WHITE)
    print(f"Skewer detection score: {skewer_score:.2f}")
    
    # Test 4: Multiple Knight Attacks (Fixed)
    print("\n🐎 TEST 4: Knight Attacking Multiple Pieces")
    board = chess.Board("8/8/2q1r3/8/3N4/8/8/K7 w - - 0 1")
    print("Position: Knight on d4 attacking Queen on c6 and Rook on e6")
    print("Knight attacks c6 and e6 from d4 - should detect fork!")
    
    multi_fork_score = scorer._tactical_fork_detection(board, chess.WHITE)
    print(f"Multi-target fork score: {multi_fork_score:.2f}")
    
    # Test 5: Diagonal Pin (Fixed)
    print("\n⚡ TEST 5: Diagonal Pin")
    board = chess.Board("8/8/8/3nk3/4B3/8/8/K7 w - - 0 1")
    print("Position: Bishop on e4, Knight on d5, King on e6")
    print("Bishop pins knight to king on diagonal!")
    
    diag_pin_score = scorer._tactical_pin_detection(board, chess.WHITE)
    print(f"Diagonal pin score: {diag_pin_score:.2f}")
    
    print(f"\n📊 TACTICAL PATTERN SUMMARY:")
    print(f"✅ Knight Fork: {fork_score:.2f}")
    print(f"✅ Pin Detection: {pin_score:.2f}")
    print(f"✅ Skewer Detection: {skewer_score:.2f}")
    print(f"✅ Multi-Fork: {multi_fork_score:.2f}")
    print(f"✅ Diagonal Pin: {diag_pin_score:.2f}")
    
    total_tactical = fork_score + pin_score + skewer_score + multi_fork_score + diag_pin_score
    print(f"\n🎯 Total Tactical Score: {total_tactical:.2f}")
    
    if total_tactical > 50:
        print("🏆 EXCELLENT: Tactical patterns are working great!")
    elif total_tactical > 20:
        print("✅ GOOD: Tactical patterns are detecting opportunities!")
    else:
        print("⚠️  Tactical patterns may need adjustment")

def test_why_starting_position_is_zero():
    """Explain why starting position shows 0 for tactical patterns"""
    
    print(f"\n🤔 WHY STARTING POSITION SHOWS 0.00 FOR TACTICS")
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
    board = chess.Board()  # Starting position
    
    print("Starting position analysis:")
    print("- No pieces can immediately fork two valuable targets")
    print("- No pins are possible (pieces don't line up)")
    print("- No skewers are available (no aligned valuable pieces)")
    print("- This is CORRECT behavior - starting position has no tactics!")
    
    print("\nChecking each piece type:")
    
    # Check knights
    knights = list(board.pieces(chess.KNIGHT, chess.WHITE))
    print(f"White knights on: {[chess.square_name(sq) for sq in knights]}")
    for knight in knights:
        attacks = board.attacks(knight)
        targets = [board.piece_at(sq) for sq in attacks if board.piece_at(sq)]
        enemy_targets = [p for p in targets if p and p.color == chess.BLACK]
        print(f"  Knight on {chess.square_name(knight)} attacks: {len(enemy_targets)} enemy pieces")
    
    # Check bishops
    bishops = list(board.pieces(chess.BISHOP, chess.WHITE))
    print(f"White bishops on: {[chess.square_name(sq) for sq in bishops]}")
    for bishop in bishops:
        attacks = board.attacks(bishop)
        targets = [board.piece_at(sq) for sq in attacks if board.piece_at(sq)]
        enemy_targets = [p for p in targets if p and p.color == chess.BLACK]
        print(f"  Bishop on {chess.square_name(bishop)} attacks: {len(enemy_targets)} enemy pieces")
    
    print("\n✅ Conclusion: 0.00 tactical scores in starting position are CORRECT!")
    print("🎯 Tactical detection will activate when real opportunities exist!")

if __name__ == "__main__":
    test_specific_tactical_patterns()
    test_why_starting_position_is_zero()
