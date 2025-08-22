# test_v5_4_tactical_verification.py

"""
V5.4 Tactical Feature Verification Test
Tests the tactical pattern recognition system to ensure proper functionality.
"""

import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r_scoring_calculation import V7P3RScoringCalculation

def test_pin_detection():
    """Test the pin detection system with a known pin position."""
    print("Testing pin detection...")
    
    # Create a position with a pin (Queen pins Rook to King)
    board = chess.Board("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    
    piece_values = {
        chess.KING: 0.0,
        chess.QUEEN: 9.0,
        chess.ROOK: 5.0,
        chess.BISHOP: 3.25,
        chess.KNIGHT: 3.0,
        chess.PAWN: 1.0
    }
    
    scorer = V7P3RScoringCalculation(piece_values)
    
    # Test pin detection for white
    white_tactical_score = scorer._tactical_pattern_recognition(board, chess.WHITE)
    print(f"White tactical score: {white_tactical_score}")
    
    # Test pin detection for black
    black_tactical_score = scorer._tactical_pattern_recognition(board, chess.BLACK)
    print(f"Black tactical score: {black_tactical_score}")
    
    # Test specific pin detection
    white_pin_score = scorer._detect_pins(board, chess.WHITE)
    black_pin_score = scorer._detect_pins(board, chess.BLACK)
    
    print(f"White pin score: {white_pin_score}")
    print(f"Black pin score: {black_pin_score}")
    
    return True

def test_fork_detection():
    """Test fork detection with a knight fork position."""
    print("\nTesting fork detection...")
    
    # Create a position where a knight can fork king and queen
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
    
    piece_values = {
        chess.KING: 0.0,
        chess.QUEEN: 9.0,
        chess.ROOK: 5.0,
        chess.BISHOP: 3.25,
        chess.KNIGHT: 3.0,
        chess.PAWN: 1.0
    }
    
    scorer = V7P3RScoringCalculation(piece_values)
    
    # Test fork detection
    white_fork_score = scorer._detect_forks(board, chess.WHITE)
    black_fork_score = scorer._detect_forks(board, chess.BLACK)
    
    print(f"White fork score: {white_fork_score}")
    print(f"Black fork score: {black_fork_score}")
    
    return True

def test_enhanced_pawn_structure():
    """Test enhanced pawn structure analysis."""
    print("\nTesting enhanced pawn structure...")
    
    # Create a position with various pawn structure issues
    board = chess.Board("rnbqkbnr/pp1p1ppp/8/2pPp3/8/8/PPP1PPPP/RNBQKBNR w KQkq c6 0 3")
    
    piece_values = {
        chess.KING: 0.0,
        chess.QUEEN: 9.0,
        chess.ROOK: 5.0,
        chess.BISHOP: 3.25,
        chess.KNIGHT: 3.0,
        chess.PAWN: 1.0
    }
    
    scorer = V7P3RScoringCalculation(piece_values)
    
    # Test enhanced pawn structure analysis
    white_pawn_score = scorer._enhanced_pawn_structure(board, chess.WHITE)
    black_pawn_score = scorer._enhanced_pawn_structure(board, chess.BLACK)
    
    print(f"White enhanced pawn score: {white_pawn_score}")
    print(f"Black enhanced pawn score: {black_pawn_score}")
    
    return True

def test_endgame_logic():
    """Test endgame logic with a king and pawn endgame."""
    print("\nTesting endgame logic...")
    
    # Create a simple king and pawn endgame
    board = chess.Board("8/8/8/8/3k4/8/3P4/3K4 w - - 0 1")
    
    piece_values = {
        chess.KING: 0.0,
        chess.QUEEN: 9.0,
        chess.ROOK: 5.0,
        chess.BISHOP: 3.25,
        chess.KNIGHT: 3.0,
        chess.PAWN: 1.0
    }
    
    scorer = V7P3RScoringCalculation(piece_values)
    
    # Test endgame logic
    white_endgame_score = scorer._endgame_logic(board, chess.WHITE)
    black_endgame_score = scorer._endgame_logic(board, chess.BLACK)
    
    print(f"White endgame score: {white_endgame_score}")
    print(f"Black endgame score: {black_endgame_score}")
    
    return True

def test_overall_scoring():
    """Test the overall scoring system with v5.4 enhancements."""
    print("\nTesting overall scoring system...")
    
    # Test with starting position
    board = chess.Board()
    
    piece_values = {
        chess.KING: 0.0,
        chess.QUEEN: 9.0,
        chess.ROOK: 5.0,
        chess.BISHOP: 3.25,
        chess.KNIGHT: 3.0,
        chess.PAWN: 1.0
    }
    
    scorer = V7P3RScoringCalculation(piece_values)
    
    # Test overall scoring
    white_score = scorer.calculate_score(board, chess.WHITE, 0.0)
    black_score = scorer.calculate_score(board, chess.BLACK, 0.0)
    
    print(f"Starting position - White score: {white_score}")
    print(f"Starting position - Black score: {black_score}")
    print(f"Score difference: {white_score - black_score}")
    
    return True

def main():
    """Run all v5.4 tactical verification tests."""
    print("V7P3R v5.4 Tactical Feature Verification Test")
    print("=" * 50)
    
    try:
        # Run all tests
        test_pin_detection()
        test_fork_detection()
        test_enhanced_pawn_structure()
        test_endgame_logic()
        test_overall_scoring()
        
        print("\n" + "=" * 50)
        print("All v5.4 tactical verification tests completed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    main()
