# v7p3r_debug.py
"""Debugging and Critical Utility Classes for v7p3rChess"""
import sys
import os
import time
import datetime
import logging
import gc
import psutil
from logging.handlers import RotatingFileHandler
from typing import Optional

class v7p3rLogger:
    @staticmethod
    def log(message: str):
        """Logs a message."""
        print(f"{message}")

    @staticmethod
    def log_error(message: str):
        """Logs an error message."""
        print(f"ERROR: {message}")

    @staticmethod
    def log_warning(message: str):
        """Logs a warning message."""
        print(f"WARNING: {message}")

    @staticmethod
    def log_info(message: str):
        """Logs an informational message."""
        print(f"INFO: {message}")

    @staticmethod
    def log_debug(message: str):
        """Logs a debug message."""
        print(f"DEBUG: {message}")
        
    @staticmethod
    def clear_logs(specific_module: Optional[str] = None):
        """
        Clears log files before starting a new game to avoid confusion with old logs.
        
        Args:
            specific_module: If provided, only clears logs for this module. 
                           If None, clears all logs in the logging directory.
        
        Returns:
            int: Number of log files cleared
        """
        # Find the logging directory
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        log_dir = os.path.join(project_root, 'logging')
        
        if not os.path.exists(log_dir):
            v7p3rLogger.log_info("No logging directory found. Nothing to clear.")
            return 0
        
        count = 0
        try:
            # Get all log files in the directory
            for filename in os.listdir(log_dir):
                # Check if this is a log file (either .log or rotated .log.N)
                if filename.endswith('.log') or any(filename.endswith(f'.log.{i}') for i in range(1, 10)):
                    # If specific module is provided, only clear logs for that module
                    if specific_module is not None:
                        if not filename.startswith(f"{specific_module}."):
                            continue
                    
                    # Clear the file content
                    filepath = os.path.join(log_dir, filename)
                    with open(filepath, 'w') as f:
                        # Just open in write mode to truncate the file
                        pass
                    count += 1
                    
            v7p3rLogger.log_info(f"Cleared {count} log files in {log_dir}")
            return count
            
        except Exception as e:
            v7p3rLogger.log_error(f"Error clearing log files: {str(e)}")
            return 0

    @staticmethod
    def setup_logger(module_name: str, log_level: int = logging.DEBUG) -> logging.Logger:
        """
        Sets up and returns a logger for the specified module.
        
        Args:
            module_name: Name of the module (e.g., 'v7p3r_score', 'v7p3r_search')
            log_level: Logging level (default: DEBUG)
            
        Returns:
            Configured logger instance
        """
        # Create logging directory relative to project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
        log_dir = os.path.join(project_root, 'logging')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # Setup individual logger for the module
        log_filename = f"{module_name}.log"
        log_file_path = os.path.join(log_dir, log_filename)

        logger = logging.getLogger(module_name)
        logger.setLevel(log_level)

        # Avoid adding handlers multiple times
        if not logger.handlers:
            file_handler = RotatingFileHandler(
                log_file_path,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=3,
                delay=True
            )
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(funcName)-15s | %(message)s',
                datefmt='%H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            logger.propagate = False
            
        v7p3rLogger.log_info(f"Logger setup complete for {module_name}")
        return logger

    @staticmethod
    def logging_setup():
        """Sets up the logging configuration."""
        # TODO: Deprecated - use setup_logger() instead
        v7p3rLogger.log_info("Logging setup complete.")
        # Create logging directory relative to project root
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        log_dir = os.path.join(project_root, 'logging')
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)

        # Setup individual logger for this file
        timestamp = v7p3rUtilities.get_timestamp()
        #log_filename = f"v7p3r_score_{timestamp}.log"
        log_filename = f"v7p3r_score.log"  # Use a fixed log filename for simplicity
        log_file_path = os.path.join(log_dir, log_filename)

        #v7p3r_score_logger = logging.getLogger(f"v7p3r_score_{timestamp}")
        v7p3r_score_logger = logging.getLogger("v7p3r_score")
        v7p3r_score_logger.setLevel(logging.DEBUG)

        if not v7p3r_score_logger.handlers:
            file_handler = RotatingFileHandler(
                log_file_path,
                maxBytes=10*1024*1024,
                backupCount=3,
                delay=True
            )
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(funcName)-15s | %(message)s',
                datefmt='%H:%M:%S'
            )
            file_handler.setFormatter(formatter)
            v7p3r_score_logger.addHandler(file_handler)
            v7p3r_score_logger.propagate = False
class v7p3rDebugger:
    @staticmethod
    def debug_condition(condition: bool, message: str):
        """Asserts a condition and logs an error message if it fails."""
        if not condition:
            print(f"ERROR: {message}")
    
    @staticmethod
    def debug_config():
        """Validates the current configuration."""
        # TODO Placeholder for actual settings validation logic that should output a config validation check report to the debug log and terminal if a game is active
        v7p3rLogger.log_debug("Current configuration validated.")

class v7p3rUtilities:
    @staticmethod
    def format_time(seconds: float) -> str:
        """Formats time in seconds to a human-readable string."""
        if seconds < 0.001:
            return f"{seconds * 1000:.2f} ms"
        else:
            return f"{seconds:.2f} s"

    @staticmethod
    def is_valid_move(move: str) -> bool:
        """Checks if a move string is valid."""
        return len(move) == 4 and move.isalnum()
    
    @staticmethod
    def get_timestamp():
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @staticmethod
    def get_time() -> str:
        """Returns the current time as a formatted string."""
        return datetime.datetime.now().strftime("%H:%M:%S")
    
    @staticmethod
    def resource_path(relative_path: str):
        """Get absolute path to resource, works for dev and for PyInstaller"""
        base = getattr(sys, '_MEIPASS', None)
        if base:
            return os.path.join(base, relative_path)
        return os.path.join(os.path.abspath("."), relative_path)
    
    @staticmethod
    def free_memory():
        """Free up memory by clearing unused variables."""
        gc.collect()
        v7p3rLogger.log_info("Memory freed.")

    @staticmethod
    def get_memory_usage() -> str:
        """Returns current memory usage as a string."""
        process = psutil.Process(os.getpid())
        mem = process.memory_info().rss / (1024 * 1024)  # Convert to MB
        return f"Memory usage: {mem:.2f} MB"

class v7p3rFreeze:
    @staticmethod
    def snapshot():
        """Take a snapshot of the engine state."""
        timestamp = v7p3rUtilities.get_timestamp()
        v7p3rLogger.log(f"Engine state snapshot taken at {timestamp}.")
        # TODO: Implement actual snapshot logic for engine state
        return timestamp

    @staticmethod
    def freeze():
        """Freeze the engine state by saving all critical files and configurations."""
        timestamp = v7p3rUtilities.get_timestamp()
        freeze_dir = os.path.join(os.path.dirname(__file__), 'engine_freezes', f"freeze_{timestamp}")
        
        try:
            # Create freeze directory
            os.makedirs(freeze_dir, exist_ok=True)
            
            # TODO: Copy critical engine files, configs, active games, etc.
            # This would include:
            # - All v7p3r_*.py files
            # - configs/ directory
            # - active_game.pgn
            # - Current metrics
            # - Training data states
            
            v7p3rLogger.log(f"Engine state frozen to: {freeze_dir}")
            return freeze_dir
            
        except Exception as e:
            v7p3rLogger.log_error(f"Failed to freeze engine state: {str(e)}")
            return None
    
    @staticmethod
    def unfreeze():
        """Unfreeze the engine state."""
        v7p3rLogger.log("Engine state unfrozen.")
        # TODO: Implement restore from freeze functionality
    
    @staticmethod
    def freeze_code():
        """Freeze the current code state by creating a backup of all source files."""
        timestamp = v7p3rUtilities.get_timestamp()
        code_freeze_dir = os.path.join(os.path.dirname(__file__), 'code_freezes', f"code_freeze_{timestamp}")
        
        try:
            os.makedirs(code_freeze_dir, exist_ok=True)
            # TODO: Copy all .py files to the freeze directory
            v7p3rLogger.log(f"Code state frozen to: {code_freeze_dir}")
            return code_freeze_dir
            
        except Exception as e:
            v7p3rLogger.log_error(f"Failed to freeze code state: {str(e)}")
            return None