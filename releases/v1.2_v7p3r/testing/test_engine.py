# test_engine.py

"""Basic tests for V7P3R Chess Engine v1.2."""

import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r_engine import V7P3REngine
from v7p3r_config import V7P3RConfig
from v7p3r_pst import V7P3RPST
from v7p3r_scoring import V7P3RScoring
from v7p3r_book import V7P3RBook


def test_config():
    """Test configuration loading."""
    print("Testing configuration...")
    config = V7P3RConfig()
    engine_config = config.get_engine_config()
    assert engine_config['name'] == 'V7P3R'
    assert engine_config['version'] == '1.2.0'
    print("✓ Configuration test passed")


def test_pst():
    """Test piece-square tables."""
    print("Testing piece-square tables...")
    pst = V7P3RPST()
    
    # Test basic functionality
    knight_e4 = pst.get_piece_value(chess.KNIGHT, chess.E4, chess.WHITE)
    knight_a1 = pst.get_piece_value(chess.KNIGHT, chess.A1, chess.WHITE)
    
    # Knight should prefer center
    assert knight_e4 > knight_a1
    
    # Test board evaluation
    board = chess.Board()
    score = pst.evaluate_position(board)
    assert abs(score) < 50  # Starting position should be close to neutral
    
    print("✓ Piece-square tables test passed")


def test_scoring():
    """Test scoring system."""
    print("Testing scoring system...")
    config = V7P3RConfig()
    pst = V7P3RPST()
    
    piece_values = {
        chess.KING: 0.0,
        chess.QUEEN: 9.0,
        chess.ROOK: 5.0,
        chess.BISHOP: 3.25,
        chess.KNIGHT: 3.0,
        chess.PAWN: 1.0
    }
    
    scoring = V7P3RScoring(piece_values, pst, config)
    
    # Test starting position
    board = chess.Board()
    score = scoring.evaluate_board(board)
    assert abs(score) < 0.5  # Should be close to neutral
    
    # Test position with material advantage
    board = chess.Board("rnbqkb1r/pppppppp/5n2/8/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 1 2")
    score_white = scoring.evaluate_board(board)
    
    print("✓ Scoring system test passed")


def test_book():
    """Test opening book."""
    print("Testing opening book...")
    book = V7P3RBook()
    
    # Test starting position
    board = chess.Board()
    book_move = book.get_book_move(board)
    
    assert book_move is not None
    assert book_move in board.legal_moves
    
    # Test that book has some positions
    assert book.get_book_size() > 0
    
    print("✓ Opening book test passed")


def test_basic_engine():
    """Test basic engine functionality."""
    print("Testing basic engine...")
    
    # Test engine creation
    engine = V7P3REngine()
    
    # Test engine info
    info = engine.get_engine_info()
    assert info['name'] == 'V7P3R'
    assert info['version'] == '1.2.0'
    
    # Test position evaluation
    score = engine.evaluate_position()
    assert isinstance(score, (int, float))
    
    # Test legal moves
    moves = engine.get_legal_moves()
    assert len(moves) == 20  # Starting position has 20 legal moves
    
    # Test game over detection
    assert not engine.is_game_over()
    
    print("✓ Basic engine test passed")


def main():
    """Run all tests."""
    print("Running V7P3R Chess Engine v1.2 tests...\n")
    
    try:
        test_config()
        test_pst()
        test_scoring()
        test_book()
        test_basic_engine()
        
        print("\n✓ All tests passed! V7P3R v1.2 basic functionality is working.")
        
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
