import os
import sys

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import chess
from v7p3r_config import v7p3rConfig
from v7p3r import v7p3rEngine
from v7p3r_debug import v7p3rLogger

# Set up a basic logger for this test
logger = v7p3rLogger.setup_logger("test_minimax_nobook")

def test_minimax_search_without_opening_book():
    # Load the configuration with "minimax" as the search algorithm and opening book disabled
    config_manager = v7p3rConfig(config_path=os.path.join('configs', 'test_minimax_config.json'))
    engine_config = config_manager.get_engine_config()
    
    # Verify configuration
    logger.info(f"Configuration loaded:")
    logger.info(f"  search_algorithm = {engine_config.get('search_algorithm', 'NOT_FOUND')}")
    logger.info(f"  use_opening_book = {engine_config.get('use_opening_book', 'NOT_FOUND')}")
    
    # Initialize the engine with this configuration
    engine = v7p3rEngine(engine_config=engine_config)
    
    # Create a board
    board = chess.Board()
    
    # We'll patch the opening book method to always return None
    # This ensures we bypass the opening book
    original_get_book_move = engine.search_engine.opening_book.get_book_move
    engine.search_engine.opening_book.get_book_move = lambda board: None
    
    # Search for the best move
    logger.info(f"Starting search with algorithm: {engine.search_engine.search_algorithm}")
    best_move = engine.search_engine.search(board, chess.WHITE)
    
    # Restore the original method
    engine.search_engine.opening_book.get_book_move = original_get_book_move
    
    logger.info(f"Best move found: {best_move}")
    print(f"Engine using search algorithm: {engine.search_engine.search_algorithm}")
    print(f"Best move found: {best_move}")
    
    return True

if __name__ == "__main__":
    test_minimax_search_without_opening_book()
