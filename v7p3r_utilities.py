"""
V7P3R Chess Engine Utilities Module

This module provides common utility functions used across the engine.
All path-related functionality has been moved to v7p3r_paths.py.
"""
import datetime
from pathlib import Path
from typing import Any, Union, Optional

from v7p3r_paths import paths  # Central path management

def ensure_directory_exists(path: Union[str, Path]) -> None:
    """
    Ensure that a directory exists, creating it if necessary.
    
    Args:
        path: The path to the directory to check/create, can be string or Path
    """
    try:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        print(f"Warning: Could not create directory {path}: {e}")

def get_timestamp() -> str:
    """
    Get a formatted timestamp string.
    
    Returns:
        str: Current timestamp in YYYY-MM-DD_HH-MM-SS format
    """
    return datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%SS")

def get_resource_path(relative_path: Union[str, Path]) -> Path:
    """
    Get the absolute path to a resource file. Delegates to paths module.
    
    Args:
        relative_path: The relative path to the resource file
        
    Returns:
        Path: The absolute path to the resource
    """
    return paths.get_resource_path(relative_path)

def get_project_root() -> Path:
    """
    Get the absolute path to the project root directory. Delegates to paths module.
    
    Returns:
        Path: The absolute path to the project root
    """
    return paths.root_dir
