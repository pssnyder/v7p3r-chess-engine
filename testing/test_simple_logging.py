# test_simple_logging.py
"""Test the simplified logging system"""

import os
import sys

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def main():
    """Test main function"""
    print("Starting test...")
    
    try:
        # Import the simplified debug module
        from v7p3r_debug_simple import v7p3rLogger
        
        # Create a test logger
        logger = v7p3rLogger.setup_logger("test_simple")
        logger.info("This is a test message")
        
        # Try importing core modules
        from v7p3r_config import v7p3rConfig
        logger.info("Successfully imported config")
        
        config = v7p3rConfig()
        logger.info("Created config instance")
        
        print("Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
