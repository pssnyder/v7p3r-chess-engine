# test_paths.py

"""Test suite for v7p3r_paths module."""

import os
import sys
import unittest
from pathlib import Path
from v7p3r_paths import V7P3RPaths, paths

class TestV7P3RPaths(unittest.TestCase):
    """Test cases for V7P3RPaths class."""
    
    def setUp(self):
        self.paths = V7P3RPaths()
    
    def test_singleton(self):
        """Test that V7P3RPaths is a singleton."""
        paths2 = V7P3RPaths()
        self.assertIs(self.paths, paths2)
    
    def test_root_dir(self):
        """Test root directory is set correctly."""
        expected_root = Path(__file__).parent.parent.absolute()
        self.assertEqual(self.paths.root_dir, expected_root)
    
    def test_directories_exist(self):
        """Test that all required directories exist."""
        for dir_path in self.paths.paths.values():
            self.assertTrue(dir_path.exists())
            self.assertTrue(dir_path.is_dir())
    
    def test_config_path(self):
        """Test config path generation."""
        config_path = self.paths.get_config_path("test_config")
        self.assertEqual(config_path.suffix, ".json")
        self.assertTrue(str(config_path).endswith("configs/test_config.json"))
    
    def test_stockfish_path(self):
        """Test Stockfish path is platform appropriate."""
        stockfish_path = self.paths.get_stockfish_path()
        if sys.platform == "win32":
            self.assertTrue(str(stockfish_path).endswith(".exe"))
        else:
            self.assertFalse(str(stockfish_path).endswith(".exe"))
    
    def test_resource_path(self):
        """Test resource path resolution."""
        test_resource = "test.txt"
        resource_path = self.paths.get_resource_path(test_resource)
        self.assertIsInstance(resource_path, Path)
        self.assertTrue(str(resource_path).endswith(test_resource))
    
    def test_metrics_db_path(self):
        """Test metrics database path."""
        db_path = self.paths.get_metrics_db_path()
        self.assertTrue(str(db_path).endswith("chess_metrics.db"))
        self.assertEqual(db_path.parent, self.paths.metrics_dir)
    
    def test_book_path(self):
        """Test opening book database path."""
        book_path = self.paths.get_book_path()
        self.assertTrue(str(book_path).endswith("move_library.db"))
    
    def test_active_game_pgn_path(self):
        """Test active game PGN path."""
        pgn_path = self.paths.get_active_game_pgn_path()
        self.assertTrue(str(pgn_path).endswith("active_game.pgn"))
    
    def test_puzzle_db_path(self):
        """Test puzzle database path."""
        puzzle_path = self.paths.get_puzzle_db_path()
        self.assertTrue(str(puzzle_path).endswith("puzzle_data.db"))
    
    def test_global_instance(self):
        """Test global paths instance."""
        self.assertIsInstance(paths, V7P3RPaths)
        self.assertIs(paths, self.paths)

if __name__ == '__main__':
    unittest.main()
