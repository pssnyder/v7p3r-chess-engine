# test_logging_only.py
"""Test only the basic logging functionality"""
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

def main():
    """Run basic logging test"""
    print("Starting logging test...")
    
    try:
        # First try importing v7p3r_debug
        print("Importing v7p3r_debug...")
        from v7p3r_debug import v7p3rLogger
        print("Successfully imported v7p3r_debug")
        
        # Try using basic print-style logging first
        print("Testing basic logging...")
        v7p3rLogger.log_info("Test info message")
        v7p3rLogger.log_debug("Test debug message")
        print("Basic logging works")
        
        print("Test completed successfully!")
        return True
        
    except Exception as e:
        print(f"Test failed with error: {str(e)}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
