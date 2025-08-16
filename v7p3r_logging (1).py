# v7p3r_logging.py
"""Centralized logging handlers for v7p3r chess engine that can handle concurrent file access"""

import os
import sys
import time
import logging
import threading
from logging.handlers import RotatingFileHandler
import time
import datetime
import logging
import gc
import psutil
import threading
from logging.handlers import RotatingFileHandler
from typing import Optional, Dict

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

class SafeRotatingFileHandler(RotatingFileHandler):
    """
    A RotatingFileHandler that is more robust against file access issues.
    This handler catches exceptions that might occur during file operations
    and falls back to console logging if needed.
    
    This handler is designed to be resilient to:
    - File permission errors
    - Concurrent access issues
    - Keyboard interrupts during file operations
    - Network or disk latency
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.console_fallback = logging.StreamHandler()
        self.using_fallback = False
        self.lock = threading.RLock()  # type: ignore
        self.last_size_check = 0  # Time of last file size check
        self.size_check_interval = 1.0  # Seconds between file size checks
        
        # Format the console fallback handler the same as the file handler
        if hasattr(self, 'formatter'):
            self.console_fallback.setFormatter(self.formatter)
    
    def _get_file_size(self):
        """
        Safely get the current file size.
        
        Returns:
            int: Size of the file in bytes, or 0 if the file doesn't exist or can't be accessed
        """
        try:
            if os.path.exists(self.baseFilename) and os.path.isfile(self.baseFilename):
                return os.path.getsize(self.baseFilename)
            return 0
        except (OSError, PermissionError, FileNotFoundError):
            # If we can't access the file, assume it's empty
            return 0
        except Exception:
            # For any other error, assume it's empty
            return 0
    
    def shouldRollover(self, record):
        """
        Determine if rollover should occur, safely handling file access issues.
        Implements a throttling mechanism to avoid excessive file system checks.
        
        Args:
            record: The record to be written to the log file
            
        Returns:
            bool: True if rollover should occur, False otherwise
        """
        # If already using fallback, no need to roll over
        if self.using_fallback:
            return False
            
        # Use a lock to prevent concurrent access
        if self.lock is not None:
            with self.lock:
                try:
                    # Check if we need to roll over based on file size, but limit how often we check
                    current_time = time.time()
                    if current_time - self.last_size_check >= self.size_check_interval:
                        self.last_size_check = current_time
                        
                        # If maxBytes is zero, rollover never occurs
                        if self.maxBytes <= 0:
                            return False
                            
                        # Get current file size (safely)
                        current_size = self._get_file_size()
                        
                        # Estimate size of this record
                        msg = "%s\n" % self.format(record)
                        msg_size = len(msg.encode('utf-8'))  # Get actual bytes
                        
                        # Check if adding this record would exceed maxBytes
                        if current_size + msg_size >= self.maxBytes:
                            return True
                    
                    return False
                    
                except Exception:
                    # If any error occurs during the check, don't try to roll over
                    return False
        
    def doRollover(self):
        """
        Perform a rollover, catching any exceptions that might occur.
        """
        if self.using_fallback:
            return
            
        if self.lock is not None:
            with self.lock:
                try:
                    # Try to roll over the file
                    super().doRollover()
                except Exception as e:
                    # If rollover fails, switch to console logging
                    self.using_fallback = True
                    print(f"WARNING: Log rollover failed, switching to console logging: {e}")
        
    def emit(self, record):
        """
        Emit a record, safely handling any exceptions including KeyboardInterrupt.
        
        Args:
            record: The record to be emitted
        """
        if self.using_fallback:
            try:
                self.console_fallback.emit(record)
            except Exception:
                # Last resort: if even console logging fails, just ignore it
                pass
            return
            
        try:
            # Use a lock to prevent concurrent access
            if self.lock is not None:
                with self.lock:
                    super().emit(record)
        except KeyboardInterrupt:
            # Don't let KeyboardInterrupt crash the application
            self.using_fallback = True
            try:
                self.console_fallback.emit(record)
                # Also log that we've switched to console
                fallback_record = logging.LogRecord(
                    name=record.name,
                    level=logging.WARNING,
                    pathname=record.pathname,
                    lineno=record.lineno,
                    msg="Logging interrupted by KeyboardInterrupt, switched to console",
                    args=(),
                    exc_info=None
                )
                self.console_fallback.emit(fallback_record)
            except Exception:
                # If even this fails, just continue
                pass
            # Re-raise the KeyboardInterrupt for proper application shutdown
            raise
        except Exception as e:
            # If file emission fails, switch to console logging
            self.using_fallback = True
            try:
                self.console_fallback.emit(record)
                # Also log the original error
                fallback_record = logging.LogRecord(
                    name=record.name,
                    level=logging.WARNING,
                    pathname=record.pathname,
                    lineno=record.lineno,
                    msg=f"Logging to file failed, switched to console: {e}",
                    args=(),
                    exc_info=None
                )
                self.console_fallback.emit(fallback_record)
            except Exception:
                # If even this fails, just continue
                pass

# Non-blocking handler for logging during shutdown
class NonBlockingHandler(logging.Handler):
    """
    A logging handler that doesn't block during shutdown.
    This is useful for handling KeyboardInterrupt and other shutdown scenarios.
    """
    
    def __init__(self, target_handler):
        """
        Initialize with a target handler that will receive the log records.
        
        Args:
            target_handler: The handler that will actually process the log records
        """
        super().__init__()
        self.target_handler = target_handler
        
        # Copy the formatter from the target handler
        if hasattr(target_handler, 'formatter'):
            self.setFormatter(target_handler.formatter)
            
        # Flag to indicate shutdown is in progress
        self.shutting_down = False
        
    def emit(self, record):
        """
        Emit a record, ignoring any errors during shutdown.
        
        Args:
            record: The record to be emitted
        """
        if self.shutting_down:
            # During shutdown, just ignore log records
            return
            
        try:
            self.target_handler.emit(record)
        except KeyboardInterrupt:
            # Mark that we're shutting down and don't try to log anymore
            self.shutting_down = True
            # Re-raise to allow proper shutdown
            raise
        except Exception:
            # Ignore any other errors during emission
            pass
    
    def handleError(self, record):
        """
        Handle errors which may occur during emission.
        This method just ignores errors.
        
        Args:
            record: The log record that led to the error
        """
        # Just ignore errors, especially during shutdown
        pass
