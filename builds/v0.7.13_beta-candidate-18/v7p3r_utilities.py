def ensure_parent_in_syspath(filepath: str):
    """Ensure the parent directory of the given file is in sys.path."""
    import os, sys
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(filepath), '..'))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
"""V7P3R Chess Engine Utilities Module.

This module provides common utility functions used across the engine,
including file/path operations and timestamp generation.
"""
import datetime
from pathlib import Path
from typing import Union
from v7p3r_paths import paths

def ensure_directory_exists(path: Union[str, Path]) -> None:
    """Ensure that a directory exists, creating it if necessary.
    
    Args:
        path: The path to the directory to check/create
    """
    try:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create directory {path}: {e}")

def get_timestamp() -> str:
    """Get a formatted timestamp string.
    
    Returns:
        A timestamp in YYYY-MM-DD_HH-MM-SS format
    """
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%SS")

def get_resource_path(relative_path: Union[str, Path]) -> Path:
    """Get the absolute path to a resource file.
    
    Args:
        relative_path: The relative path to the resource
        
    Returns:
        The absolute path to the resource
    """
    return paths.get_resource_path(relative_path)

def get_project_root() -> Path:
    """Get the absolute path to the project root directory.
    
    Returns:
        Path: The absolute path to the project root directory
    """
    return paths.root_dir
