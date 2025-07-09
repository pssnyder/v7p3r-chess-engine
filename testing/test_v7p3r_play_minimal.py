# test_v7p3r_play_minimal.py
"""Minimal test to verify v7p3r_play functionality"""

import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from v7p3r_debug import v7p3rLogger

# First disable all existing logging to start fresh
v7p3rLogger.disable_logging_for_tests()

# Now set up a simple console-only logger
logger = v7p3rLogger.setup_logger("test_minimal", log_level=20)  # INFO level

def main():
    """Main test function"""
    try:
        logger.info("Starting minimal test...")
        
        # Import the main engine modules
        logger.info("Importing core modules...")
        from v7p3r_config import v7p3rConfig
        from chess_core import ChessCore
        
        logger.info("Setting up configuration...")
        config = v7p3rConfig()
        
        logger.info("Creating chess core instance...")
        core = ChessCore(logger_name="test_minimal")
        
        logger.info("Test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
