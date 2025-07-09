import os
import sys

# Add parent directory to path so we can import modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import chess
from v7p3r_config import v7p3rConfig
from v7p3r import v7p3rEngine
from v7p3r_debug import v7p3rLogger

# Set up a basic logger for this test
logger = v7p3rLogger.setup_logger("test_minimax_search")

def test_minimax_search():
    # Load the configuration with "minimax" as the search algorithm
    config_manager = v7p3rConfig()
    engine_config = config_manager.get_engine_config()
    engine_config["search_algorithm"] = "minimax"  # Override to use minimax search
    
    # Initialize the engine with this configuration
    engine = v7p3rEngine(engine_config=engine_config)
    
    # Verify the engine's search algorithm
    logger.info(f"Engine initialized, search_engine.search_algorithm = {engine.search_engine.search_algorithm}")
    
    # Create a board and make a test move
    board = chess.Board()
    
    # Search for the best move
    best_move = engine.search_engine.search(board, chess.WHITE)
    
    logger.info(f"Best move found: {best_move}")
    print(f"Engine using search algorithm: {engine.search_engine.search_algorithm}")
    print(f"Best move found: {best_move}")
    
    return True

if __name__ == "__main__":
    test_minimax_search()
