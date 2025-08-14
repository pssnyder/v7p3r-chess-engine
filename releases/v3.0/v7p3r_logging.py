# v7p3r_logging.py
"""Enhanced logging handlers for v7p3r chess engine that can handle concurrent file access"""

import os
import time
import logging
import threading
from logging.handlers import RotatingFileHandler

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
        self.lock = threading.RLock()  # Reentrant lock for thread safety
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
