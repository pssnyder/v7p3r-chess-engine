#!/usr/bin/env python3
"""
Unit tests for chess_game.py - V7P3R Chess Engine

This module contains comprehensive unit tests for the ChessGame class,
testing initialization, game logic, move validation, and integration points.

Author: V7P3R Testing Suite
Date: 2025-06-22
"""

import sys
import os
import unittest
import tempfile
import yaml
import chess
from unittest.mock import Mock, patch, MagicMock
from io import StringIO
import json
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from chess_game import ChessGame, get_timestamp, get_log_file_path


class TestChessGameInitialization(unittest.TestCase):
    """Test ChessGame class initialization and configuration."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.original_cwd = os.getcwd()
        
        # Create mock config files
        self.chess_game_config = {
            'monitoring': {
                'enable_logging': True,
                'show_thinking': True
            },
            'game_config': {
                'human_color': 'white'
            }
        }
        
        self.v7p3r_config = {
            'engine': {
                'depth': 6,
                'time_limit': 5.0
            }
        }
        
        self.stockfish_config = {
            'path': '/usr/bin/stockfish',
            'depth': 15,
            'time': 1.0
        }

    def tearDown(self):
        """Clean up test fixtures."""
        os.chdir(self.original_cwd)

    def test_init_with_valid_fen(self):
        """Test initialization with a valid FEN position."""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        with patch('chess_game.yaml.safe_load'), \
             patch('builtins.open'), \
             patch('chess_game.pygame.init'), \
             patch('chess_game.pygame.time.Clock'):
            
            game = ChessGame(
                fen_position=fen,
                game_config=self.chess_game_config,
                v7p3r_config=self.v7p3r_config,
                stockfish_config=self.stockfish_config
            )
            
            self.assertEqual(game.starting_position, fen)
            self.assertIsNotNone(game.game_config_data)
            self.assertIsNotNone(game.v7p3r_config_data)
            self.assertIsNotNone(game.stockfish_config_data)

    def test_init_with_invalid_fen(self):
        """Test initialization with an invalid FEN position."""
        invalid_fen = "invalid_fen_string"
        
        with self.assertRaises(ValueError) as context:
            ChessGame(
                fen_position=invalid_fen,
                game_config=self.chess_game_config,
                v7p3r_config=self.v7p3r_config,
                stockfish_config=self.stockfish_config
            )
        
        self.assertIn("Invalid FEN position", str(context.exception))

    def test_init_with_non_string_fen(self):
        """Test initialization with non-string FEN position."""
        with self.assertRaises(ValueError) as context:
            ChessGame(
                fen_position=12345,
                game_config=self.chess_game_config,
                v7p3r_config=self.v7p3r_config,
                stockfish_config=self.stockfish_config
            )
        
        self.assertIn("FEN position must be a string", str(context.exception))

    def test_init_without_fen(self):
        """Test initialization without FEN position."""
        with patch('chess_game.yaml.safe_load'), \
             patch('builtins.open'), \
             patch('chess_game.pygame.init'), \
             patch('chess_game.pygame.time.Clock'):
            
            game = ChessGame(
                game_config=self.chess_game_config,
                v7p3r_config=self.v7p3r_config,
                stockfish_config=self.stockfish_config
            )
            
            self.assertIsNone(game.starting_position)

    @patch('chess_game.yaml.safe_load')
    @patch('builtins.open')
    @patch('chess_game.pygame.init')
    @patch('chess_game.pygame.time.Clock')
    def test_config_loading_from_files(self, mock_clock, mock_pygame_init, mock_open, mock_yaml):
        """Test configuration loading from files when configs not provided."""
        mock_yaml.side_effect = [
            self.chess_game_config,
            self.v7p3r_config,
            self.stockfish_config
        ]
        
        game = ChessGame()
        
        # Verify files were opened
        expected_calls = [
            unittest.mock.call("chess_game.yaml"),
            unittest.mock.call("v7p3r.yaml"),
            unittest.mock.call("engine_utilities/stockfish_handler.yaml")
        ]
        mock_open.assert_has_calls(expected_calls, any_order=True)

    def test_data_collector_assignment(self):
        """Test data collector function assignment."""
        mock_collector = Mock()
        
        with patch('chess_game.yaml.safe_load'), \
             patch('builtins.open'), \
             patch('chess_game.pygame.init'), \
             patch('chess_game.pygame.time.Clock'):
            
            game = ChessGame(
                game_config=self.chess_game_config,
                v7p3r_config=self.v7p3r_config,
                stockfish_config=self.stockfish_config,
                data_collector=mock_collector
            )
            
            self.assertEqual(game.data_collector, mock_collector)


class TestChessGameUtilityFunctions(unittest.TestCase):
    """Test utility functions in chess_game module."""

    def test_get_timestamp_format(self):
        """Test timestamp format is correct."""
        timestamp = get_timestamp()
        
        # Should be in format YYYYMMDD_HHMMSS
        self.assertEqual(len(timestamp), 15)
        self.assertIn('_', timestamp)
        
        # Should be parseable as datetime
        import datetime
        try:
            datetime.datetime.strptime(timestamp, "%Y%m%d_%H%M%S")
        except ValueError:
            self.fail("Timestamp format is incorrect")

    def test_get_log_file_path(self):
        """Test log file path generation."""
        log_path = get_log_file_path()
        
        self.assertTrue(log_path.startswith("logging"))
        self.assertTrue(log_path.endswith("chess_game.log"))
        self.assertIn(os.sep, log_path)

    @patch('chess_game.os.path.exists')
    @patch('chess_game.os.makedirs')
    def test_log_directory_creation(self, mock_makedirs, mock_exists):
        """Test that log directory is created if it doesn't exist."""
        mock_exists.return_value = False
        
        get_log_file_path()
        
        mock_makedirs.assert_called_once_with("logging", exist_ok=True)


class TestChessGameBoardOperations(unittest.TestCase):
    """Test chess board operations and game logic."""

    def setUp(self):
        """Set up test fixtures."""
        self.chess_game_config = {
            'monitoring': {'enable_logging': True, 'show_thinking': True},
            'game_config': {'human_color': 'white'}
        }
        
        self.v7p3r_config = {'engine': {'depth': 6, 'time_limit': 5.0}}
        self.stockfish_config = {'path': '/usr/bin/stockfish', 'depth': 15}

    @patch('chess_game.yaml.safe_load')
    @patch('builtins.open')
    @patch('chess_game.pygame.init')
    @patch('chess_game.pygame.time.Clock')
    def test_board_initialization_standard_position(self, mock_clock, mock_pygame_init, mock_open, mock_yaml):
        """Test board initialization with standard starting position."""
        mock_yaml.side_effect = [self.chess_game_config, self.v7p3r_config, self.stockfish_config]
        
        game = ChessGame()
        
        # Test that pygame was initialized
        mock_pygame_init.assert_called_once()
        mock_clock.assert_called_once()

    def test_board_initialization_custom_position(self):
        """Test board initialization with custom FEN position."""
        # King and pawn endgame position
        custom_fen = "8/8/8/8/8/8/4K3/4k3 w - - 0 1"
        
        with patch('chess_game.yaml.safe_load'), \
             patch('builtins.open'), \
             patch('chess_game.pygame.init'), \
             patch('chess_game.pygame.time.Clock'):
            
            game = ChessGame(
                fen_position=custom_fen,
                game_config=self.chess_game_config,
                v7p3r_config=self.v7p3r_config,
                stockfish_config=self.stockfish_config
            )
            
            self.assertEqual(game.starting_position, custom_fen)


class TestChessGamePerformance(unittest.TestCase):
    """Test performance characteristics of ChessGame operations."""

    def setUp(self):
        """Set up test fixtures."""
        self.chess_game_config = {
            'monitoring': {'enable_logging': False, 'show_thinking': False},
            'game_config': {'human_color': 'white'}
        }
        
        self.v7p3r_config = {'engine': {'depth': 3, 'time_limit': 1.0}}
        self.stockfish_config = {'path': '/usr/bin/stockfish', 'depth': 5}

    def test_initialization_performance(self):
        """Test that game initialization completes within reasonable time."""
        start_time = time.time()
        
        with patch('chess_game.yaml.safe_load'), \
             patch('builtins.open'), \
             patch('chess_game.pygame.init'), \
             patch('chess_game.pygame.time.Clock'):
            
            game = ChessGame(
                game_config=self.chess_game_config,
                v7p3r_config=self.v7p3r_config,
                stockfish_config=self.stockfish_config
            )
        
        initialization_time = time.time() - start_time
        
        # Should initialize in under 1 second
        self.assertLess(initialization_time, 1.0, 
                       f"Initialization took {initialization_time:.3f}s, expected < 1.0s")


class TestChessGameIntegration(unittest.TestCase):
    """Test integration points with other components."""

    def setUp(self):
        """Set up test fixtures."""
        self.chess_game_config = {
            'monitoring': {'enable_logging': True, 'show_thinking': True},
            'game_config': {'human_color': 'white'}
        }
        
        self.v7p3r_config = {'engine': {'depth': 6, 'time_limit': 5.0}}
        self.stockfish_config = {'path': '/usr/bin/stockfish', 'depth': 15}

    @patch('chess_game.V7P3REvaluationEngine')
    @patch('chess_game.StockfishHandler')
    @patch('chess_game.MetricsStore')
    @patch('chess_game.CloudStore')
    def test_engine_integrations(self, mock_cloud_store, mock_metrics_store, 
                                mock_stockfish, mock_v7p3r):
        """Test integration with chess engines and storage systems."""
        with patch('chess_game.yaml.safe_load'), \
             patch('builtins.open'), \
             patch('chess_game.pygame.init'), \
             patch('chess_game.pygame.time.Clock'):
            
            game = ChessGame(
                game_config=self.chess_game_config,
                v7p3r_config=self.v7p3r_config,
                stockfish_config=self.stockfish_config
            )
            
            # Verify imports are accessible (engines should be importable)
            self.assertTrue(hasattr(game, 'logger'))
            self.assertIsNotNone(game.game_config_data)


def run_chess_game_tests():
    """
    Main function to run all chess_game tests.
    
    Returns:
        dict: Test results with success/failure counts and details
    """
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestChessGameInitialization,
        TestChessGameUtilityFunctions,
        TestChessGameBoardOperations,
        TestChessGamePerformance,
        TestChessGameIntegration
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests with custom result collector
    stream = StringIO()
    runner = unittest.TextTestRunner(stream=stream, verbosity=2)
    result = runner.run(suite)
    
    # Collect results
    test_output = stream.getvalue()
    
    results = {
        'module': 'chess_game',
        'timestamp': get_timestamp(),
        'tests_run': result.testsRun,
        'failures': len(result.failures),
        'errors': len(result.errors),
        'skipped': len(result.skipped) if hasattr(result, 'skipped') else 0,
        'success_count': result.testsRun - len(result.failures) - len(result.errors),
        'success_rate': ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100) if result.testsRun > 0 else 0,
        'details': {
            'failures': [{'test': str(test), 'error': traceback} for test, traceback in result.failures],
            'errors': [{'test': str(test), 'error': traceback} for test, traceback in result.errors],
            'output': test_output
        }
    }
    
    return results


if __name__ == '__main__':
    # Run tests when executed directly
    print("=" * 60)
    print("V7P3R Chess Engine - Chess Game Unit Tests")
    print("=" * 60)
    
    results = run_chess_game_tests()
    
    # Print summary
    print(f"\nTest Results Summary:")
    print(f"Module: {results['module']}")
    print(f"Tests Run: {results['tests_run']}")
    print(f"Successes: {results['success_count']}")
    print(f"Failures: {results['failures']}")
    print(f"Errors: {results['errors']}")
    print(f"Success Rate: {results['success_rate']:.1f}%")
    
    if results['failures'] > 0 or results['errors'] > 0:
        print(f"\nFailure/Error Details:")
        for failure in results['details']['failures']:
            print(f"FAIL: {failure['test']}")
        for error in results['details']['errors']:
            print(f"ERROR: {error['test']}")
    
    print("=" * 60)
    
    # Exit with appropriate code
    sys.exit(0 if results['failures'] == 0 and results['errors'] == 0 else 1)
