# v7p3r_debug_simple.py
"""Simplified debugging and logging system for v7p3rChess"""
import sys
import os
import datetime
import logging
from typing import Dict, Optional

# Global dictionary to track logger instances
_logger_instances: Dict[str, logging.Logger] = {}

class v7p3rLogger:
    @staticmethod
    def setup_logger(module_name: str, log_level: int = logging.DEBUG) -> logging.Logger:
        """
        Sets up and returns a logger for the specified module.
        Uses a simple file handler without threading or complex rotation.
        
        Args:
            module_name: Name of the module
            log_level: Logging level (default: DEBUG)
            
        Returns:
            Configured logger instance
        """
        # Return existing logger if already configured
        if module_name in _logger_instances:
            return _logger_instances[module_name]
            
        # Create logging directory
        project_root = os.path.abspath(os.path.dirname(__file__))
        log_dir = os.path.join(project_root, 'logging')
        try:
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        except Exception as e:
            print(f"Warning: Could not create logging directory: {e}")
            
        # Setup individual logger for the module
        logger = logging.getLogger(module_name)
        logger.setLevel(log_level)
        
        # Remove any existing handlers
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)
        
        # Add console handler
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s | %(levelname)s | %(name)s | %(message)s',
            datefmt='%H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # Try to set up file handler
        try:
            log_file_path = os.path.join(log_dir, f"{module_name}.log")
            file_handler = logging.FileHandler(log_file_path, mode='w')
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            print(f"Warning: Could not set up file logging for {module_name}: {e}")
            
        logger.propagate = False
        _logger_instances[module_name] = logger
        return logger

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
        """Clear log files."""
        try:
            project_root = os.path.abspath(os.path.dirname(__file__))
            log_dir = os.path.join(project_root, 'logging')
            
            if not os.path.exists(log_dir):
                return
                
            for filename in os.listdir(log_dir):
                if specific_module and not filename.startswith(specific_module):
                    continue
                    
                if filename.endswith('.log'):
                    filepath = os.path.join(log_dir, filename)
                    try:
                        with open(filepath, 'w') as f:
                            pass
                    except Exception as e:
                        print(f"Could not clear log file {filename}: {e}")
                        
        except Exception as e:
            print(f"Error clearing log files: {e}")

class v7p3rUtilities:
    @staticmethod
    def get_timestamp():
        """Get current timestamp in standard format."""
        return datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
