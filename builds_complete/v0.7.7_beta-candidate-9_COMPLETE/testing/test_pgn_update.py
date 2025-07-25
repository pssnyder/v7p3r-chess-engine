# test_pgn_update.py
"""
Test script to verify that active_game.pgn is updated after every move 
and that pgn_watcher.py correctly follows these updates using logging.
"""
import os
import sys
import time
import subprocess
import threading
from pathlib import Path

# Add parent directory to sys.path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from v7p3r_play import v7p3rChess
from v7p3r_debug import v7p3rLogger

# Setup logger for this test
logger = v7p3rLogger.setup_logger("test_pgn_update")

def run_pgn_watcher():
    """Run the PGN watcher in a separate process"""
    logger.info("Starting PGN watcher process")
    process = subprocess.Popen(
        ["python", "pgn_watcher.py"],
        stderr=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True
    )
    return process

def main():
    """Run a test game and verify PGN updates"""
    logger.info("Starting PGN update test")
    
    # First, make sure there's no active_game.pgn or clear it
    pgn_path = Path("active_game.pgn")
    if pgn_path.exists():
        pgn_path.unlink()
        logger.info("Removed existing active_game.pgn")
    
    # Start PGN watcher in a separate process
    pgn_watcher = run_pgn_watcher()
    logger.info("PGN watcher started")
    
    # Give PGN watcher time to initialize
    time.sleep(2)
    
    # Start a new game with a specific config for testing
    logger.info("Creating new game with test configuration")
    game = v7p3rChess("speed_config.json")
    
    # Check if active_game.pgn was created
    if not pgn_path.exists():
        logger.error("active_game.pgn was not created!")
        pgn_watcher.terminate()
        return
    
    logger.info("active_game.pgn was successfully created on game initialization")
    
    # Play 5 moves to test PGN updates
    for i in range(5):
        logger.info(f"Making move {i+1}")
        
        # Get current turn
        current_player = "White" if game.board.turn else "Black"
        
        # Get and make move
        player = game.white_player if game.board.turn else game.black_player
        logger.info(f"Getting move for {player} ({current_player})")
        
        move = game.get_engine_move()
        if move:
            logger.info(f"Move found: {move.uci()}")
            
            # Check PGN file size before move
            pre_size = pgn_path.stat().st_size if pgn_path.exists() else 0
            
            # Make the move
            game.push_move(move)
            
            # Check PGN file size after move
            time.sleep(0.5)  # Give time for file to be written
            post_size = pgn_path.stat().st_size if pgn_path.exists() else 0
            
            if post_size > pre_size:
                logger.info(f"Γ£ô PGN file was updated after move {i+1}")
            else:
                logger.error(f"Γ£ù PGN file was NOT updated after move {i+1}")
            
            # Wait a bit to let the PGN watcher catch up
            time.sleep(1)
        else:
            logger.error(f"No move found for {player}")
            break
        
        # Check if game is over
        if game.board.is_game_over():
            logger.info("Game is over")
            break
    
    # Save final game
    game.save_game_data()
    logger.info("Game completed and saved")
    
    # Give pgn_watcher time to process final state
    time.sleep(2)
    
    # Terminate pgn_watcher
    pgn_watcher.terminate()
    stdout, stderr = pgn_watcher.communicate()
    logger.info("PGN watcher terminated")
    
    if stdout:
        logger.info(f"PGN watcher stdout: {stdout}")
    if stderr:
        logger.error(f"PGN watcher stderr: {stderr}")
    
    logger.info("Test completed successfully")

if __name__ == "__main__":
    main()
