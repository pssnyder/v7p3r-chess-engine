import chess
import os
import sys
from v7p3r_config import v7p3rConfig
from v7p3r import v7p3rEngine
from v7p3r_debug import v7p3rLogger

# Set up a basic logger for this test
logger = v7p3rLogger.setup_logger("test_search_algorithm")

def test_search_algorithm():
    # Load the configuration that specifies "minimax" as the search algorithm
    config_manager = v7p3rConfig(config_path=os.path.join('configs', 'centipawn_test_config.json'))
    engine_config = config_manager.get_engine_config()
    
    # Verify the configuration has the correct search algorithm
    logger.info(f"Configuration loaded, search_algorithm = {engine_config.get('search_algorithm', 'NOT_FOUND')}")
    
    # Initialize the engine with this configuration
    engine = v7p3rEngine(engine_config=engine_config)
    
    # Verify the engine passed the correct configuration to the search engine
    logger.info(f"Engine initialized, search_engine.search_algorithm = {engine.search_engine.search_algorithm}")
    
    # Create a board and make a test move
    board = chess.Board()
    
    # Log which search algorithm is being used
    logger.info(f"Using search algorithm: {engine.search_engine.search_algorithm}")
    
    # Search for the best move
    best_move = engine.search_engine.search(board, chess.WHITE)
    
    logger.info(f"Best move found: {best_move}")
    
    # Return success
    return engine.search_engine.search_algorithm == "minimax"

if __name__ == "__main__":
    success = test_search_algorithm()
    if success:
        print("SUCCESS: The search algorithm is set to 'minimax' as expected.")
    else:
        print("FAILURE: The search algorithm is not set to 'minimax'.")
