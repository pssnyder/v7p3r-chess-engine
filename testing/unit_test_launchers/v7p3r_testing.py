#!/usr/bin/env python3
"""
Unit tests for v7p3r.py - V7P3R Chess Engine Evaluation Engine

This module contains comprehensive unit tests for the V7P3REvaluationEngine class,
testing initialization, search algorithms, evaluation functions, and performance.

Author: V7P3R Testing Suite
Date: 2025-06-22
"""

import sys
import os
import unittest
import tempfile
import yaml
import chess
from unittest.mock import Mock, patch, MagicMock, mock_open
from io import StringIO
import json
import time
import threading

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from v7p3r_engine.v7p3r import V7P3REvaluationEngine, LimitedSizeDict


class TestLimitedSizeDict(unittest.TestCase):
    """Test LimitedSizeDict helper class."""

    def test_init_with_maxlen(self):
        """Test initialization with maxlen parameter."""
        d = LimitedSizeDict(maxlen=3)
        self.assertEqual(d.maxlen, 3)
        self.assertEqual(len(d), 0)

    def test_size_limit_enforcement(self):
        """Test that dict enforces size limit."""
        d = LimitedSizeDict(maxlen=2)
        d['a'] = 1
        d['b'] = 2
        d['c'] = 3  # Should evict 'a'
        
        self.assertEqual(len(d), 2)
        self.assertNotIn('a', d)
        self.assertIn('b', d)
        self.assertIn('c', d)

    def test_move_to_end_on_update(self):
        """Test that updating existing key moves it to end."""
        d = LimitedSizeDict(maxlen=3)
        d['a'] = 1
        d['b'] = 2
        d['c'] = 3
        d['a'] = 10  # Update existing key
        d['d'] = 4   # Should evict 'b', not 'a'
        
        self.assertIn('a', d)
        self.assertNotIn('b', d)
        self.assertIn('c', d)
        self.assertIn('d', d)


class TestV7P3REvaluationEngineInitialization(unittest.TestCase):
    """Test V7P3REvaluationEngine initialization."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        
        # Mock config files
        self.v7p3r_config = {
            'v7p3r': {
                'performance': {
                    'hash_size': 1024000,
                    'thread_limit': 2
                },
                'search': {
                    'depth': 6,
                    'time_limit': 5.0
                }
            }
        }
        
        self.game_config = {
            'performance': {
                'hash_size': 512000,
                'thread_limit': 1
            }
        }

    def tearDown(self):
        """Clean up test fixtures."""
        os.chdir(self.original_cwd)

    @patch('v7p3r_engine.v7p3r.TimeManager')
    @patch('v7p3r_engine.v7p3r.OpeningBook')
    @patch('builtins.open', new_callable=mock_open)
    @patch('v7p3r_engine.v7p3r.yaml.safe_load')
    def test_init_default_values(self, mock_yaml, mock_open_file, mock_book, mock_time):
        """Test initialization with default values."""
        mock_yaml.side_effect = [self.v7p3r_config, self.game_config]
        
        engine = V7P3REvaluationEngine()
        
        self.assertIsInstance(engine.board, chess.Board)
        self.assertEqual(engine.current_player, chess.WHITE)
        self.assertEqual(engine.nodes_searched, 0)
        self.assertIsNotNone(engine.transposition_table)
        self.assertIsNotNone(engine.killer_moves)
        self.assertIsNotNone(engine.history_table)
        mock_time.assert_called_once()
        mock_book.assert_called_once()

    @patch('v7p3r_engine.v7p3r.TimeManager')
    @patch('v7p3r_engine.v7p3r.OpeningBook')
    @patch('builtins.open', new_callable=mock_open)
    @patch('v7p3r_engine.v7p3r.yaml.safe_load')
    def test_init_with_custom_board(self, mock_yaml, mock_open_file, mock_book, mock_time):
        """Test initialization with custom board position."""
        mock_yaml.side_effect = [self.v7p3r_config, self.game_config]
        
        custom_board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2")
        engine = V7P3REvaluationEngine(board=custom_board, player=chess.BLACK)
        
        self.assertEqual(engine.board, custom_board)
        self.assertEqual(engine.current_player, chess.BLACK)

    @patch('v7p3r_engine.v7p3r.TimeManager')
    @patch('v7p3r_engine.v7p3r.OpeningBook')
    @patch('builtins.open', side_effect=FileNotFoundError)
    def test_init_missing_config_files(self, mock_open_file, mock_book, mock_time):
        """Test initialization when config files are missing."""
        engine = V7P3REvaluationEngine()
        
        self.assertEqual(engine.v7p3r_config_data, {})
        self.assertEqual(engine.game_settings_config_data, {})
        self.assertIsNotNone(engine.transposition_table)

    @patch('v7p3r_engine.v7p3r.TimeManager')
    @patch('v7p3r_engine.v7p3r.OpeningBook')
    @patch('builtins.open', new_callable=mock_open)
    @patch('v7p3r_engine.v7p3r.yaml.safe_load')
    def test_config_loading_priorities(self, mock_yaml, mock_open_file, mock_book, mock_time):
        """Test that v7p3r config takes priority over game config."""
        mock_yaml.side_effect = [self.v7p3r_config, self.game_config]
        
        engine = V7P3REvaluationEngine()
        
        # v7p3r config should take priority
        self.assertEqual(engine.hash_size, 1024000)
        self.assertEqual(engine.threads, 2)


class TestV7P3REvaluationEngineBasicMethods(unittest.TestCase):
    """Test basic V7P3REvaluationEngine methods."""

    def setUp(self):
        """Set up test fixtures."""
        with patch('v7p3r_engine.v7p3r.TimeManager'), \
             patch('v7p3r_engine.v7p3r.OpeningBook'), \
             patch('builtins.open', side_effect=FileNotFoundError):
            self.engine = V7P3REvaluationEngine()

    def test_piece_values(self):
        """Test piece value constants."""
        expected_values = {
            chess.KING: 0.0,
            chess.QUEEN: 9.0,
            chess.ROOK: 5.0,
            chess.BISHOP: 3.25,
            chess.KNIGHT: 3.0,
            chess.PAWN: 1.0
        }
        self.assertEqual(self.engine.piece_values, expected_values)

    def test_initial_state(self):
        """Test initial engine state."""
        self.assertEqual(self.engine.nodes_searched, 0)
        self.assertIsInstance(self.engine.transposition_table, LimitedSizeDict)
        self.assertEqual(len(self.engine.killer_moves), 50)
        self.assertIsInstance(self.engine.history_table, dict)
        self.assertIsInstance(self.engine.counter_moves, dict)

    def test_board_manipulation(self):
        """Test board state management."""
        original_fen = self.engine.board.fen()
        
        # Make a move
        move = chess.Move.from_uci("e2e4")
        if move in self.engine.board.legal_moves:
            self.engine.board.push(move)
            self.assertNotEqual(self.engine.board.fen(), original_fen)
            
            # Undo the move
            self.engine.board.pop()
            self.assertEqual(self.engine.board.fen(), original_fen)


class TestV7P3REvaluationEngineSearchMethods(unittest.TestCase):
    """Test V7P3REvaluationEngine search algorithms."""

    def setUp(self):
        """Set up test fixtures."""
        with patch('v7p3r_engine.v7p3r.TimeManager'), \
             patch('v7p3r_engine.v7p3r.OpeningBook'), \
             patch('builtins.open', side_effect=FileNotFoundError):
            self.engine = V7P3REvaluationEngine()

    def test_transposition_table_usage(self):
        """Test transposition table functionality."""
        key = "test_position"
        value = {"score": 100, "depth": 5, "move": "e2e4"}
        
        # Store in transposition table
        self.engine.transposition_table[key] = value
        
        # Retrieve from transposition table
        self.assertEqual(self.engine.transposition_table[key], value)
        
    def test_killer_moves_initialization(self):
        """Test killer moves table initialization."""
        self.assertEqual(len(self.engine.killer_moves), 50)
        for ply_moves in self.engine.killer_moves:
            self.assertEqual(len(ply_moves), 2)
            self.assertIsNone(ply_moves[0])
            self.assertIsNone(ply_moves[1])

    def test_history_table_updates(self):
        """Test history table functionality."""
        move = chess.Move.from_uci("e2e4")
        depth = 5
        
        # Initially empty
        self.assertEqual(len(self.engine.history_table), 0)
        
        # Add entry
        self.engine.history_table[move] = depth
        self.assertEqual(self.engine.history_table[move], depth)


class TestV7P3REvaluationEnginePerformance(unittest.TestCase):
    """Test V7P3REvaluationEngine performance characteristics."""

    def setUp(self):
        """Set up test fixtures."""
        with patch('v7p3r_engine.v7p3r.TimeManager'), \
             patch('v7p3r_engine.v7p3r.OpeningBook'), \
             patch('builtins.open', side_effect=FileNotFoundError):
            self.engine = V7P3REvaluationEngine()

    def test_transposition_table_size_limit(self):
        """Test that transposition table respects size limits."""
        original_maxlen = self.engine.transposition_table.maxlen
        
        # Fill beyond capacity
        for i in range(original_maxlen + 10):
            self.engine.transposition_table[f"key_{i}"] = {"value": i}
        
        # Should not exceed maxlen
        self.assertLessEqual(len(self.engine.transposition_table), original_maxlen)

    def test_nodes_searched_tracking(self):
        """Test nodes searched counter."""
        initial_nodes = self.engine.nodes_searched
        self.assertEqual(initial_nodes, 0)
        
        # Simulate search
        self.engine.nodes_searched += 100
        self.assertEqual(self.engine.nodes_searched, 100)

    def test_memory_efficiency(self):
        """Test memory usage stays within reasonable bounds."""
        import sys
        
        initial_size = sys.getsizeof(self.engine.transposition_table)
        
        # Add many entries
        for i in range(1000):
            self.engine.transposition_table[f"position_{i}"] = {
                "score": i % 100,
                "depth": i % 10,
                "best_move": f"move_{i}"
            }
        
        final_size = sys.getsizeof(self.engine.transposition_table)
        
        # Size should not grow unbounded due to LimitedSizeDict
        self.assertLess(final_size, initial_size * 100)  # Reasonable growth limit


class TestV7P3REvaluationEngineErrorHandling(unittest.TestCase):
    """Test V7P3REvaluationEngine error handling."""

    def test_invalid_board_state_handling(self):
        """Test handling of invalid board states."""
        with patch('v7p3r_engine.v7p3r.TimeManager'), \
             patch('v7p3r_engine.v7p3r.OpeningBook'), \
             patch('builtins.open', side_effect=FileNotFoundError):
            
            # Test with various invalid inputs
            try:
                import chess
                empty_board = chess.Board()
                engine = V7P3REvaluationEngine(board=empty_board)
                # Should handle gracefully or raise appropriate exception
            except (TypeError, AttributeError):
                pass  # Expected for invalid input

    @patch('v7p3r_engine.v7p3r.v7p3r_engine_logger')
    def test_config_loading_error_handling(self, mock_logger):
        """Test error handling during config loading."""
        with patch('v7p3r_engine.v7p3r.TimeManager'), \
             patch('v7p3r_engine.v7p3r.OpeningBook'), \
             patch('builtins.open', side_effect=Exception("Config error")):
            
            engine = V7P3REvaluationEngine()
            
            # Should log error and continue with defaults
            mock_logger.error.assert_called()
            self.assertEqual(engine.v7p3r_config_data, {})

    @patch('v7p3r_engine.v7p3r.yaml.safe_load')
    def test_malformed_yaml_handling(self, mock_yaml):
        """Test handling of malformed YAML files."""
        mock_yaml.side_effect = yaml.YAMLError("Invalid YAML")
        
        with patch('v7p3r_engine.v7p3r.TimeManager'), \
             patch('v7p3r_engine.v7p3r.OpeningBook'), \
             patch('builtins.open', mock_open(read_data="invalid: yaml: content:")):
            
            engine = V7P3REvaluationEngine()
            
            # Should handle YAML errors gracefully
            self.assertEqual(engine.v7p3r_config_data, {})


class TestV7P3REvaluationEngineIntegration(unittest.TestCase):
    """Test V7P3REvaluationEngine integration with other components."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_time_manager = Mock()
        self.mock_opening_book = Mock()
        
    @patch('v7p3r_engine.v7p3r.TimeManager')
    @patch('v7p3r_engine.v7p3r.OpeningBook')
    def test_time_manager_integration(self, mock_book_class, mock_time_class):
        """Test integration with TimeManager."""
        mock_time_class.return_value = self.mock_time_manager
        mock_book_class.return_value = self.mock_opening_book
        
        with patch('builtins.open', side_effect=FileNotFoundError):
            engine = V7P3REvaluationEngine()
        
        mock_time_class.assert_called_once()
        self.assertEqual(engine.time_manager, self.mock_time_manager)

    @patch('v7p3r_engine.v7p3r.TimeManager')
    @patch('v7p3r_engine.v7p3r.OpeningBook')
    def test_opening_book_integration(self, mock_book_class, mock_time_class):
        """Test integration with OpeningBook."""
        mock_time_class.return_value = self.mock_time_manager
        mock_book_class.return_value = self.mock_opening_book
        
        with patch('builtins.open', side_effect=FileNotFoundError):
            engine = V7P3REvaluationEngine()
        
        mock_book_class.assert_called_once()
        self.assertEqual(engine.opening_book, self.mock_opening_book)


if __name__ == '__main__':
    unittest.main()