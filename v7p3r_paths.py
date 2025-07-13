# v7p3r_paths.py

"""Path management module for the V7P3R chess engine.
This module provides a singleton path manager that handles all path-related operations.
"""

import os
import sys
import platform
from pathlib import Path
from typing import Dict, Union, Optional

class V7P3RPaths:
    """Central path management singleton for V7P3R chess engine."""
    
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(V7P3RPaths, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._initialize()
            V7P3RPaths._initialized = True
    
    def _initialize(self):
        """Initialize path configuration."""
        # Ensure we find the actual project root
        self.root_dir = Path(__file__).resolve().parent
        
        # Define standard directories
        self.config_dir = self.root_dir / 'configs'
        self.stockfish_dir = self.root_dir / 'stockfish'
        self.training_dir = self.root_dir / 'training_data'
        self.docs_dir = self.root_dir / 'docs'
        self.test_dir = self.root_dir / 'testing'
        self.metrics_dir = self.root_dir / 'metrics'
        
        # Define all paths that need to be created or accessed
        self.paths = {
            'config': self.config_dir,
            'stockfish': self.stockfish_dir / platform.system().lower(),
            'training': self.training_dir,
            'docs': self.docs_dir,
            'test': self.test_dir,
            'metrics': self.metrics_dir,
            'book': self.root_dir / 'move_library.db',
            'puzzle': self.root_dir / 'puzzle_data.db',
            'active_game': self.root_dir / 'active_game.pgn'
        }
        
        # Add root directory to Python path if not already there
        if str(self.root_dir) not in sys.path:
            sys.path.insert(0, str(self.root_dir))
        
        # Create directories
        self._create_directories()
    
    def _create_directories(self):
        """Create all required directories."""
        # Create directories that should exist (skip files with extensions)
        for key, path in self.paths.items():
            if isinstance(path, Path) and not path.suffix:  # Only create directories, not files
                path.mkdir(parents=True, exist_ok=True)
    
    def get_config_file(self, name: str = 'default_config') -> Path:
        """Get the path to a configuration file.
        
        Args:
            name: Name of the configuration file without extension
            
        Returns:
            Path to the configuration file
        """
        return self.config_dir / f"{name}.json"
    
    def resolve_config_path(self, name: str = 'default_config') -> Path:
        """Get the path to a configuration file.
        
        Args:
            name: Name of the configuration file without extension
            
        Returns:
            Path to the configuration file
        """
        return self.get_config_file(name)
    
    def get_book_path(self) -> Path:
        """Get the path to the opening book database."""
        return self.paths['book']
    
    def get_active_game_pgn_path(self) -> Path:
        """Get the path to the active game PGN file."""
        return self.paths['active_game']
    
    def get_config_path(self) -> Path:
        """Get the path to the config directory."""
        return self.paths['config']
        
    def get_doc_path(self) -> Path:
        """Get the path to the documentation directory."""
        return self.paths['docs']
    
    def get_stockfish_path(self) -> Path:
        """Get the path to the Stockfish binary."""
        return self.paths['stockfish']
        
    def get_training_path(self) -> Path:
        """Get the path to the training data directory."""
        return self.paths['training']
    
    def get_test_path(self) -> Path:
        """Get the path to the test directory."""
        return self.paths['test']
    
    def get_metrics_path(self) -> Path:
        """Get the path to the metrics directory."""
        return self.paths['metrics']
    
    def get_resource_path(self, relative_path: Union[str, Path]) -> Path:
        """Get the absolute path to a resource file.
        
        Args:
            relative_path: The relative path to the resource
            
        Returns:
            The absolute path to the resource
        """
        return self.root_dir / relative_path

# Global singleton instance
paths = V7P3RPaths()
