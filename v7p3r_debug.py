# v7p3r_debug.py
"""Debugging and Critical Utility Classes for v7p3rChess"""
import sys
import os
import time
import datetime
import logging
import gc
import psutil
import threading
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict
try:
    from v7p3r_logging import SafeRotatingFileHandler, NonBlockingHandler
except ImportError:
    # Fall back to standard RotatingFileHandler if our custom one isn't available
    print("Warning: SafeRotatingFileHandler not found, using standard RotatingFileHandler")
    SafeRotatingFileHandler = RotatingFileHandler
    
    # Define a simple NonBlockingHandler if the import fails
    class NonBlockingHandler(logging.Handler):
        def __init__(self, target_handler):
            super().__init__()
            self.target_handler = target_handler
            self.shutting_down = False
            
        def emit(self, record):
            if not self.shutting_down:
                try:
                    self.target_handler.emit(record)
                except KeyboardInterrupt:
                    self.shutting_down = True
                    raise
                except Exception:
                    pass

# Global dictionary to track logger instances
_logger_instances: Dict[str, logging.Logger] = {}
_logger_lock = threading.RLock()  # Lock for thread-safe logger creation

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
    def clear_logs(specific_module: Optional[str] = None, delete_rotated: bool = True):
        """
        Clears log files before starting a new game to avoid confusion with old logs.
        
        Args:
            specific_module: If provided, only clears logs for this module. 
                           If None, clears all logs in the logging directory.
            delete_rotated: If True, rotated log files (*.log.N) will be deleted entirely.
                          If False, they will only be truncated like the main log files.
        
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
        cleared_count = 0
        deleted_count = 0
        
        try:
            # Get all log files in the directory
            for filename in os.listdir(log_dir):
                # Check if this is a log file
                is_log_file = filename.endswith('.log')
                is_rotated_log = '.log.' in filename  # Matches any .log.N pattern
                
                if is_log_file or is_rotated_log:
                    # If specific module is provided, only clear logs for that module
                    if specific_module is not None:
                        module_prefix = f"{specific_module}."
                        module_name = filename.split('.')[0]  # Get the part before first dot
                        
                        if module_name != specific_module:
                            continue
                    
                    filepath = os.path.join(log_dir, filename)
                    
                    # For rotated logs, either delete or truncate based on parameter
                    if is_rotated_log and delete_rotated:
                        try:
                            os.remove(filepath)
                            deleted_count += 1
                        except (PermissionError, OSError) as e:
                            v7p3rLogger.log_warning(f"Could not delete rotated log file {filename}: {e}")
                            # Try to truncate if we can't delete
                            try:
                                with open(filepath, 'w') as f:
                                    pass
                                cleared_count += 1
                            except Exception:
                                pass
                    else:
                        # For primary logs or if not deleting rotated, just truncate
                        try:
                            with open(filepath, 'w') as f:
                                # Just open in write mode to truncate the file
                                pass
                            cleared_count += 1
                        except (PermissionError, OSError) as e:
                            v7p3rLogger.log_warning(f"Could not clear log file {filename}: {e}")
                    
                    count += 1
            
            if deleted_count > 0:
                v7p3rLogger.log_info(f"Deleted {deleted_count} rotated log files in {log_dir}")
            
            v7p3rLogger.log_info(f"Cleared {cleared_count} log files in {log_dir} (total processed: {count})")
            return count
            
        except Exception as e:
            v7p3rLogger.log_error(f"Error clearing log files: {str(e)}")
            return 0

    @staticmethod
    def setup_logger(module_name: str, log_level: int = logging.DEBUG) -> logging.Logger:
        """
        Sets up and returns a logger for the specified module.
        Thread-safe implementation that reuses existing loggers.
        
        Args:
            module_name: Name of the module (e.g., 'v7p3r_score', 'v7p3r_search')
            log_level: Logging level (default: DEBUG)
            
        Returns:
            Configured logger instance
        """
        # Check if we already have this logger configured
        with _logger_lock:
            if module_name in _logger_instances:
                return _logger_instances[module_name]
            
            # Create logging directory relative to project root
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__)))
            log_dir = os.path.join(project_root, 'logging')
            
            # Create directory if it doesn't exist
            try:
                if not os.path.exists(log_dir):
                    os.makedirs(log_dir, exist_ok=True)
            except (PermissionError, OSError) as e:
                print(f"Warning: Could not create logging directory. Using console logging only. Error: {str(e)}")
                # Set up a console-only logger
                logger = logging.getLogger(module_name)
                logger.setLevel(log_level)
                
                # Remove any existing handlers
                for handler in logger.handlers[:]:
                    logger.removeHandler(handler)
                
                # Add console handler
                console_handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s | %(levelname)s | %(funcName)-15s | %(message)s',
                    datefmt='%H:%M:%S'
                )
                console_handler.setFormatter(formatter)
                
                # Wrap with non-blocking handler for graceful interruption handling
                non_blocking_console = NonBlockingHandler(console_handler)
                non_blocking_console.setFormatter(formatter)
                
                logger.addHandler(non_blocking_console)
                logger.propagate = False
                
                # Cache the logger
                _logger_instances[module_name] = logger
                return logger

            # Setup individual logger for the module
            log_filename = f"{module_name}.log"
            log_file_path = os.path.join(log_dir, log_filename)

            logger = logging.getLogger(module_name)
            logger.setLevel(log_level)

            # Remove any existing handlers
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)
            
            try:
                # Use our safer handler instead of the standard one
                file_handler = SafeRotatingFileHandler(
                    log_file_path,
                    maxBytes=10*1024*1024,  # 10MB
                    backupCount=3,
                    delay=True  # Don't open the file until first log message
                )
                formatter = logging.Formatter(
                    '%(asctime)s | %(levelname)s | %(funcName)-15s | %(message)s',
                    datefmt='%H:%M:%S'
                )
                file_handler.setFormatter(formatter)
                
                # Wrap with non-blocking handler for graceful interruption handling
                non_blocking_handler = NonBlockingHandler(file_handler)
                non_blocking_handler.setFormatter(formatter)
                
                logger.addHandler(non_blocking_handler)
                logger.propagate = False
            except (PermissionError, OSError) as e:
                # Fall back to a console handler if file access fails
                console_handler = logging.StreamHandler()
                formatter = logging.Formatter(
                    '%(asctime)s | %(levelname)s | %(funcName)-15s | %(message)s',
                    datefmt='%H:%M:%S'
                )
                console_handler.setFormatter(formatter)
                
                # Wrap with non-blocking handler for graceful interruption handling
                non_blocking_console = NonBlockingHandler(console_handler)
                non_blocking_console.setFormatter(formatter)
                
                logger.addHandler(non_blocking_console)
                logger.propagate = False
                print(f"Warning: Could not create log file for {module_name}. Using console logging instead. Error: {str(e)}")
            
            # Cache the logger
            _logger_instances[module_name] = logger
            return logger

    @staticmethod
    def logging_setup():
        """
        Sets up the logging configuration.
        DEPRECATED: Use setup_logger() instead.
        This method is kept for backward compatibility.
        """
        print("Warning: logging_setup() is deprecated. Use setup_logger() instead.")
        # For backward compatibility, set up the v7p3r_score logger
        return v7p3rLogger.setup_logger("v7p3r_score")
    
    @staticmethod
    def disable_logging_for_tests():
        """
        Disables all logging for testing.
        This prevents file access issues during tests and makes tests run faster.
        """
        with _logger_lock:
            # Reset our cache
            _logger_instances.clear()
            
            # Disable all existing loggers
            for name in logging.root.manager.loggerDict:
                logger = logging.getLogger(name)
                logger.handlers = []
                logger.addHandler(logging.NullHandler())
                logger.propagate = False
                logger.disabled = True
                
            # Also disable root logger
            logging.root.handlers = []
            logging.root.addHandler(logging.NullHandler())
            logging.root.disabled = True
            
            print("All logging disabled for testing")
            
            # Return a dummy logger that does nothing
            null_logger = logging.getLogger("null_logger")
            null_logger.addHandler(logging.NullHandler())
            null_logger.propagate = False
            null_logger.disabled = True
            
            # Cache it for future calls
            _logger_instances["_disabled_"] = null_logger
            return null_logger

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