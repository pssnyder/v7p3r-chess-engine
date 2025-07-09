"""
Utilities module for v7p3r Chess Engine providing common functionality.
"""
import os
import sys
import datetime
import datetime

def resource_path(relative_path: str) -> str:
    """
    Get the absolute path to a resource file, works both in development and when packaged.
    
    Args:
        relative_path: The relative path to the resource file
        
    Returns:
        str: The absolute path to the resource
    """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = getattr(sys, '_MEIPASS', os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(base_path, relative_path)
    except Exception as e:
        return os.path.join(os.path.dirname(os.path.abspath(__file__)), relative_path)

def get_project_root() -> str:
    """
    Get the absolute path to the project root directory.
    
    Returns:
        str: The absolute path to the project root
    """
    return os.path.dirname(os.path.abspath(__file__))

def ensure_directory_exists(path: str) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path: The path to the directory to check/create
    """
    try:
        if not os.path.exists(path):
            os.makedirs(path)
    except Exception as e:
        print(f"Warning: Could not create directory {path}: {e}")

def get_timestamp() -> str:
    """
    Get a formatted timestamp string.
    
    Returns:
        str: Current timestamp in YYYY-MM-DD_HH-MM-SS format
    """
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
