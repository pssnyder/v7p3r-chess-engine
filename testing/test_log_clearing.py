# test_log_clearing.py
"""
Test script to demonstrate the log clearing functionality
"""
import os
import sys
from v7p3r_debug import v7p3rLogger

def main():
    # First, setup a few loggers to generate log files
    logger1 = v7p3rLogger.setup_logger("test_module1")
    logger2 = v7p3rLogger.setup_logger("test_module2")
    
    # Write some log messages
    logger1.info("This is a test message from module 1")
    logger2.info("This is a test message from module 2")
    
    # Get the log directory path
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
    log_dir = os.path.join(project_root, 'logging')
    
    # Show the log files that exist
    print("\nBefore clearing logs:")
    print("-" * 30)
    if os.path.exists(log_dir):
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log') or any(f.endswith(f'.log.{i}') for i in range(1, 10))]
        for log_file in log_files:
            file_path = os.path.join(log_dir, log_file)
            file_size = os.path.getsize(file_path)
            print(f"{log_file}: {file_size} bytes")
    
    # Demonstrate clearing only one module's logs
    print("\nClearing only test_module1 logs:")
    print("-" * 30)
    cleared_count = v7p3rLogger.clear_logs("test_module1")
    print(f"Cleared {cleared_count} log files for test_module1")
    
    # Show the log files after clearing one module
    print("\nAfter clearing test_module1 logs:")
    print("-" * 30)
    if os.path.exists(log_dir):
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log') or any(f.endswith(f'.log.{i}') for i in range(1, 10))]
        for log_file in log_files:
            file_path = os.path.join(log_dir, log_file)
            file_size = os.path.getsize(file_path)
            print(f"{log_file}: {file_size} bytes")
    
    # Write a new message to the cleared log
    logger1.info("This is a new message after clearing the log")
    
    # Now clear all logs
    print("\nClearing all logs:")
    print("-" * 30)
    cleared_count = v7p3rLogger.clear_logs()
    print(f"Cleared {cleared_count} log files")
    
    # Show the log files after clearing all
    print("\nAfter clearing all logs:")
    print("-" * 30)
    if os.path.exists(log_dir):
        log_files = [f for f in os.listdir(log_dir) if f.endswith('.log') or any(f.endswith(f'.log.{i}') for i in range(1, 10))]
        for log_file in log_files:
            file_path = os.path.join(log_dir, log_file)
            file_size = os.path.getsize(file_path)
            print(f"{log_file}: {file_size} bytes")

if __name__ == "__main__":
    main()
