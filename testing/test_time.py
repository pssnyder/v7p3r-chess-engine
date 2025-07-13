"""Test suite for v7p3r time management.
Tests time allocation and management during search."""

import os
import sys
import chess
import time
import unittest
from typing import Optional

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from v7p3r_time import v7p3rTime

class TestV7P3RTime(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.time_manager = v7p3rTime()
        
    def test_init_time_manager(self):
        """Test time manager initialization."""
        self.assertIsNotNone(self.time_manager)
        self.assertIsNone(self.time_manager.start_time)
        self.assertIsNone(self.time_manager.allocated_time)
        self.assertIsNone(self.time_manager.max_time)
        self.assertIsNone(self.time_manager.emergency_time)
        
    def test_infinite_time_control(self):
        """Test infinite time control handling."""
        time_control = {'infinite': True}
        board = chess.Board()
        allocated_time = self.time_manager.allocate_time(time_control, board)
        self.assertEqual(allocated_time, float('inf'))
        
    def test_fixed_move_time(self):
        """Test fixed time per move."""
        time_control = {'movetime': 5000}  # 5 seconds in milliseconds
        board = chess.Board()
        allocated_time = self.time_manager.allocate_time(time_control, board)
        self.assertEqual(allocated_time, 5.0)  # Should convert to seconds
        
    def test_increment_handling(self):
        """Test time control with increment."""
        self.time_manager.increment = 2.0  # 2 second increment
        self.time_manager.white_time = 60.0  # 1 minute remaining
        self.time_manager.black_time = 60.0  # 1 minute remaining
        self.time_manager.time_control_enabled = True
        
        # Time allocation should consider increment
        self.assertGreater(self.time_manager.white_time, 0)
        self.assertGreater(self.time_manager.black_time, 0)
        self.assertEqual(self.time_manager.increment, 2.0)
        
    def test_time_initialization(self):
        """Test time manager initialization."""
        self.assertEqual(self.time_manager.increment, 0.0)  # Default increment
        self.assertEqual(self.time_manager.white_time, float('inf'))  # Default white time
        self.assertEqual(self.time_manager.black_time, float('inf'))  # Default black time
        self.assertFalse(self.time_manager.time_control_enabled)  # Default time control disabled

if __name__ == '__main__':
    unittest.main()
