# test_logging_setup.py
"""Test the setup_logger functionality"""
import sys
import os
import logging

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def main():
    """Run logging setup test"""
    print("Starting logging setup test...")
    
    try:
        # Import v7p3r_debug
        print("Importing v7p3r_debug...")
        from v7p3r_debug import v7p3rLogger
        print("Successfully imported v7p3r_debug")
        
        # First disable all logging to start fresh
        print("Disabling existing logging...")
        logging.getLogger().handlers = []
        
        # Set up a console-only logger
        print("Setting up test logger...")
        logger = v7p3rLogger.setup_logger("test_setup", log_level=logging.INFO)
        
        # Test the logger
        print("Testing logger...")
        logger.info("Test info message via logger")
        logger.debug("This debug message should not appear")
        print("Logger test complete")
        
        print("Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
