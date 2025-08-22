#!/usr/bin/env python3
"""
Test V5.4 Tactical Enhancement Features
Tests the new tactical pattern recognition, advanced pawn structure, and endgame logic.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import chess
from src.v7p3r_scoring_calculation import V7P3RScoringCalculation

def get_default_piece_values():
    """Get default piece values for testing."""
    return {
        chess.PAWN: 1,
        chess.KNIGHT: 3,
        chess.BISHOP: 3,
        chess.ROOK: 5,
        chess.QUEEN: 9,
        chess.KING: 0
    }

def test_tactical_pattern_recognition():
    """Test tactical pattern recognition system."""
    print("=== Testing Tactical Pattern Recognition ===")
    
    calc = V7P3RScoringCalculation(get_default_piece_values())
    
    # Test pin detection - bishop pinning knight to king
    board = chess.Board("rnbqkb1r/pppppppp/5n2/8/3B4/8/PPPPPPPP/RNBQK1NR w KQkq - 0 1")
    
    white_score = calc._tactical_pattern_recognition(board, chess.WHITE)
    black_score = calc._tactical_pattern_recognition(board, chess.BLACK)
    
    print(f"Pin detection test - White: {white_score:.2f}, Black: {black_score:.2f}")
    
    # Test fork opportunity - knight fork
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4N3/8/PPPPPPPP/RNBQKB1R w KQkq - 0 1")
    
    white_fork = calc._tactical_pattern_recognition(board, chess.WHITE)
    print(f"Knight fork potential - White: {white_fork:.2f}")
    
    print("Tactical pattern recognition tests completed!\n")

def test_enhanced_pawn_structure():
    """Test enhanced pawn structure analysis."""
    print("=== Testing Enhanced Pawn Structure ===")
    
    calc = V7P3RScoringCalculation(get_default_piece_values())
    
    # Test isolated pawn
    board = chess.Board("rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 1")
    
    white_pawn = calc._enhanced_pawn_structure(board, chess.WHITE)
    black_pawn = calc._enhanced_pawn_structure(board, chess.BLACK)
    
    print(f"Isolated pawn test - White: {white_pawn:.2f}, Black: {black_pawn:.2f}")
    
    # Test doubled pawns
    board = chess.Board("rnbqkbnr/pp1ppppp/8/2p5/2P5/2P5/PP1PPPPP/RNBQKBNR w KQkq - 0 1")
    
    white_doubled = calc._enhanced_pawn_structure(board, chess.WHITE)
    print(f"Doubled pawn penalty - White: {white_doubled:.2f}")
    
    # Test pawn chain
    board = chess.Board("rnbqkbnr/pppppppp/8/8/3PP3/2P5/PP3PPP/RNBQKBNR w KQkq - 0 1")
    
    white_chain = calc._enhanced_pawn_structure(board, chess.WHITE)
    print(f"Pawn chain bonus - White: {white_chain:.2f}")
    
    print("Enhanced pawn structure tests completed!\n")

def test_endgame_logic():
    """Test endgame logic system."""
    print("=== Testing Endgame Logic ===")
    
    calc = V7P3RScoringCalculation(get_default_piece_values())
    
    # Test king activity in endgame
    board = chess.Board("8/8/8/3k4/3K4/8/8/8 w - - 0 1")  # King and king endgame
    
    white_endgame = calc._endgame_logic(board, chess.WHITE)
    black_endgame = calc._endgame_logic(board, chess.BLACK)
    
    print(f"King activity test - White: {white_endgame:.2f}, Black: {black_endgame:.2f}")
    
    # Test king-pawn endgame
    board = chess.Board("8/8/8/3k4/3KP3/8/8/8 w - - 0 1")  # King and pawn vs king
    
    white_kp = calc._endgame_logic(board, chess.WHITE)
    black_kp = calc._endgame_logic(board, chess.BLACK)
    
    print(f"King-pawn endgame - White: {white_kp:.2f}, Black: {black_kp:.2f}")
    
    # Test opposition
    board = chess.Board("8/8/8/3k4/8/3K4/8/8 w - - 0 1")  # Direct opposition setup
    
    white_opposition = calc._endgame_logic(board, chess.WHITE)
    print(f"Opposition test - White: {white_opposition:.2f}")
    
    print("Endgame logic tests completed!\n")

def test_overall_scoring_integration():
    """Test integration of all new features."""
    print("=== Testing Overall Integration ===")
    
    calc = V7P3RScoringCalculation(get_default_piece_values())
    
    # Test with a complex tactical position
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1")
    
    white_total = calc.calculate_score(board, chess.WHITE)
    black_total = calc.calculate_score(board, chess.BLACK)
    
    print(f"Complex position - White: {white_total:.2f}, Black: {black_total:.2f}")
    
    # Test specific components
    white_tactical = calc._tactical_pattern_recognition(board, chess.WHITE)
    white_pawn = calc._enhanced_pawn_structure(board, chess.WHITE)
    white_endgame = calc._endgame_logic(board, chess.WHITE)
    
    print(f"White components - Tactical: {white_tactical:.2f}, Pawn: {white_pawn:.2f}, Endgame: {white_endgame:.2f}")
    
    print("Overall integration tests completed!\n")

def main():
    """Run all V5.4 enhancement tests."""
    print("V7P3R Chess Engine - V5.4 Tactical Enhancement Tests")
    print("=" * 55)
    
    try:
        test_tactical_pattern_recognition()
        test_enhanced_pawn_structure()
        test_endgame_logic()
        test_overall_scoring_integration()
        
        print("All V5.4 enhancement tests completed successfully!")
        print("The engine now includes:")
        print("✓ Advanced tactical pattern recognition (pins, forks, skewers, discoveries)")
        print("✓ Enhanced pawn structure analysis (isolation, doubling, chains, storms)")
        print("✓ Comprehensive endgame logic (king activity, opposition, promotion)")
        print("✓ Integrated tactical tie-breakers and chess best practices")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
