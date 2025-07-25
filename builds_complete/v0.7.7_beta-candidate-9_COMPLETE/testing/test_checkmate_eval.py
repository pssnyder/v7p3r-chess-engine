import chess
import chess.pgn
import os
import sys
import pygame

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required engine modules
from v7p3r_play import v7p3rChess
from v7p3r import v7p3rEngine
from v7p3r_debug import v7p3rLogger

def test_checkmate_evaluation():
    """Test that checkmate positions have the correct evaluation in PGN."""
    
    # Set up logger
    logger = v7p3rLogger.setup_logger("test_checkmate_eval")
    logger.info("Starting checkmate evaluation test")
    
    # Create game instance
    game = v7p3rChess()
    
    # Test 1: White checkmated (black wins)
    # Scholar's mate position where white is checkmated
    white_checkmated_fen = "rnb1kbnr/pppp1ppp/8/4p3/6Pq/5P2/PPPPP2P/RNBQKBNR w KQkq - 1 3"
    
    # Load the FEN position
    board = chess.Board(white_checkmated_fen)
    
    # This position should be a checkmate
    assert board.is_checkmate(), "Test board should be in checkmate"
    print(f"Is checkmate: {board.is_checkmate()}")
    print(f"Board FEN: {board.fen()}")
    
    # Set the board in our game instance
    game.board = board
    game.game = chess.pgn.Game.from_board(board)
    game.game_node = game.game
    board.turn = chess.WHITE  # White to move (and is in checkmate)
    
    # Verify it's actually checkmate
    assert board.is_checkmate(), "Test board should be in checkmate"
    print(f"Is checkmate: {board.is_checkmate()}")
    print(f"Board FEN: {board.fen()}")
    
    # Create a temporary game with this position
    game.board = board
    game.game = chess.pgn.Game.from_board(board)
    game.game_node = game.game
    
    
    # Record evaluation
    game.monitoring_enabled = True
    game.verbose_output_enabled = True
    
    print("Testing White checkmated position")
    game.record_evaluation()
    print(f"Recorded evaluation: {game.current_eval}")
    
    # Ensure the evaluation is a large negative number (white is checkmated)
    assert game.current_eval < -900000000, f"White checkmate eval should be negative, got {game.current_eval}"
    
    # Create a board with black checkmated
    board = chess.Board(None)  # Create empty board
    board.set_piece_at(chess.A1, chess.Piece(chess.KING, chess.BLACK))
    board.set_piece_at(chess.H1, chess.Piece(chess.QUEEN, chess.WHITE))
    board.set_piece_at(chess.A8, chess.Piece(chess.KING, chess.WHITE))
    board.turn = chess.BLACK  # Black to move (and is in checkmate)
    
    # Verify it's actually checkmate
    assert board.is_checkmate(), "Test board should be in checkmate"
    print(f"Is checkmate: {board.is_checkmate()}")
    print(f"Board FEN: {board.fen()}")
    
    # Create a temporary game with this position
    game.board = board
    game.game = chess.pgn.Game.from_board(board)
    game.game_node = game.game
    
    # Record evaluation
    print("Testing Black checkmated position")
    game.record_evaluation()
    print(f"Recorded evaluation: {game.current_eval}")
    
    # Ensure the evaluation is a large positive number (black is checkmated)
    assert game.current_eval > 900000000, f"Black checkmate eval should be positive, got {game.current_eval}"
    
    logger.info("Checkmate evaluation test passed!")
    print("Checkmate evaluation test passed!")
    return True

if __name__ == "__main__":
    if test_checkmate_evaluation():
        print("Γ£à Checkmate evaluation test passed!")
    else:
        print("Γ¥î Checkmate evaluation test failed!")
