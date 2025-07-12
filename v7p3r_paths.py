# v7p3r_paths.py

"""Path management module for the V7P3R chess engine.
Centralizes all path-related functionality and ensures consistent path handling across modules.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Union

class V7P3RPaths:
    """Central path management for V7P3R chess engine."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(V7P3RPaths, cls).__new__(cls)
            cls._instance._initialize()
        return cls._instance
    
    def _initialize(self):
        """Initialize path configuration."""
        self.root_dir = Path(__file__).parent.absolute()
        self.config_dir = self.root_dir / 'configs'
        self.stockfish_dir = self.root_dir / 'stockfish'
        self.training_dir = self.root_dir / 'training_data'
        self.docs_dir = self.root_dir / 'docs'
        self.test_dir = self.root_dir / 'testing'
        self.metrics_dir = self.root_dir / 'metrics'
        
        # Ensure required directories exist
        self._ensure_directories()
        
        # Add root directory to Python path if not already there
        if str(self.root_dir) not in sys.path:
            sys.path.insert(0, str(self.root_dir))
    
    def _ensure_directories(self):
        """Ensure all required directories exist."""
        for directory in [
            self.config_dir,
            self.stockfish_dir,
            self.training_dir,
            self.docs_dir,
            self.test_dir,
            self.metrics_dir
        ]:
            directory.mkdir(parents=True, exist_ok=True)
    
    def get_config_path(self, config_name: str) -> Path:
        """Get the path to a configuration file."""
        return self.config_dir / f"{config_name}.json"
    
    def get_stockfish_path(self) -> Path:
        """Get the path to the Stockfish executable."""
        if sys.platform == "win32":
            return self.stockfish_dir / "stockfish-windows-x86-64-avx2.exe"
        else:
            return self.stockfish_dir / "stockfish"
    
    def get_resource_path(self, relative_path: Union[str, Path]) -> Path:
        """
        Get absolute path to a resource file, works both for dev and for PyInstaller.
        
        Args:
            relative_path: Relative path to the resource file
            
        Returns:
            Absolute path to the resource file
        """
        try:
            # PyInstaller creates a temp folder and stores path in _MEIPASS
            base_path = sys._MEIPASS  # type: ignore # PyInstaller-specific attribute
        except Exception:
            base_path = os.path.abspath(".")
            
        # Convert to Path object for consistent handling
        rel_path = Path(relative_path)
        abs_path = Path(base_path) / rel_path
        
        # Check if path exists
        if not abs_path.exists():
            # Try looking relative to engine root
            engine_root = Path(__file__).parent
            abs_path = engine_root / rel_path
            
        return abs_path
    
    def get_metrics_db_path(self) -> Path:
        """Get the path to the metrics database."""
        return self.metrics_dir / "chess_metrics.db"
    
    def get_book_path(self) -> Path:
        """Get the path to the opening book database."""
        return self.root_dir / "move_library.db"
    
    def get_active_game_pgn_path(self) -> Path:
        """Get the path to the active game PGN file."""
        return self.root_dir / "active_game.pgn"
    
    def get_puzzle_db_path(self) -> Path:
        """Get the path to the puzzle database."""
        return self.root_dir / "puzzle_data.db"
    
    @property
    def paths(self) -> dict[str, Path]:
        """Get all configured paths as a dictionary."""
        return {
            'root': self.root_dir,
            'config': self.config_dir,
            'stockfish': self.stockfish_dir,
            'training': self.training_dir,
            'docs': self.docs_dir,
            'test': self.test_dir,
            'metrics': self.metrics_dir
        }

# Global instance
paths = V7P3RPaths()
