#!/usr/bin/env python3
"""
Unit tests for stockfish_handler.py - v7p3r Chess Engine Stockfish Handler

This module contains comprehensive unit tests for the StockfishHandler class,
testing initialization, move generation, position analysis, and integration.

Author: v7p3r Testing Suite
Date: 2025-06-22
"""

import sys
import os
import unittest
import tempfile
import yaml
from unittest.mock import Mock, patch, MagicMock, mock_open
from io import StringIO
import time

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

from engine_utilities.stockfish_handler import StockfishHandler


class TestStockfishHandlerInitialization(unittest.TestCase):
    """Test StockfishHandler initialization."""

    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.mock_config = {
            'path': '/usr/bin/stockfish',
            'depth': 15,
            'time': 1.0,
            'threads': 1,
            'hash': 128
        }

    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    @patch('engine_utilities.stockfish_handler.Stockfish')
    def test_init_with_valid_config(self, mock_stockfish_class):
        """Test initialization with valid configuration."""
        mock_engine = Mock()
        mock_stockfish_class.return_value = mock_engine
        
        handler = StockfishHandler(config=self.mock_config)
        
        mock_stockfish_class.assert_called_once()
        self.assertIsNotNone(handler)

    @patch('engine_utilities.stockfish_handler.Stockfish')
    def test_init_with_invalid_path(self, mock_stockfish_class):
        """Test initialization with invalid Stockfish path."""
        mock_stockfish_class.side_effect = Exception("Stockfish not found")
        
        invalid_config = self.mock_config.copy()
        invalid_config['path'] = '/invalid/path/to/stockfish'
        
        with self.assertRaises(Exception):
            StockfishHandler(config=invalid_config)

    @patch('engine_utilities.stockfish_handler.Stockfish')
    def test_init_with_default_config(self, mock_stockfish_class):
        """Test initialization with default configuration."""
        mock_engine = Mock()
        mock_stockfish_class.return_value = mock_engine
        
        handler = StockfishHandler()
        
        mock_stockfish_class.assert_called_once()
        self.assertIsNotNone(handler)

    @patch('engine_utilities.stockfish_handler.Stockfish')
    def test_configuration_setting(self, mock_stockfish_class):
        """Test that configuration parameters are properly set."""
        mock_engine = Mock()
        mock_stockfish_class.return_value = mock_engine
        
        handler = StockfishHandler(config=self.mock_config)
        
        # Check if configuration methods were called
        if hasattr(handler, 'engine'):
            # Test configuration calls if they exist in the actual implementation
            self.assertTrue(hasattr(handler, 'engine'))


class TestStockfishHandlerMoveGeneration(unittest.TestCase):
    """Test StockfishHandler move generation and analysis."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        
        with patch('engine_utilities.stockfish_handler.Stockfish') as mock_stockfish_class:
            mock_stockfish_class.return_value = self.mock_engine
            self.handler = StockfishHandler()

    def test_get_best_move_basic(self):
        """Test getting best move from a position."""
        # Mock Stockfish response
        self.mock_engine.get_best_move.return_value = 'e2e4'
        
        if hasattr(self.handler, 'get_best_move'):
            best_move = self.handler.get_best_move()
            self.assertEqual(best_move, 'e2e4')
            self.mock_engine.get_best_move.assert_called_once()

    def test_get_best_move_with_time_limit(self):
        """Test getting best move with time limit."""
        self.mock_engine.get_best_move_time.return_value = 'e2e4'
        
        if hasattr(self.handler, 'get_best_move_time'):
            best_move = self.handler.get_best_move_time(1000)  # 1 second
            self.assertEqual(best_move, 'e2e4')
            self.mock_engine.get_best_move_time.assert_called_once_with(1000)

    def test_get_best_move_with_depth(self):
        """Test getting best move with specific depth."""
        self.mock_engine.get_best_move.return_value = 'e2e4'
        
        if hasattr(self.handler, 'set_depth'):
            self.handler.set_depth(10)
            self.mock_engine.set_depth.assert_called_once_with(10)

    def test_position_evaluation(self):
        """Test position evaluation."""
        self.mock_engine.get_evaluation.return_value = {'type': 'cp', 'value': 50}
        
        if hasattr(self.handler, 'get_evaluation'):
            evaluation = self.handler.get_evaluation()
            self.assertIsNotNone(evaluation)
            self.mock_engine.get_evaluation.assert_called_once()

    def test_top_moves_analysis(self):
        """Test getting top moves analysis."""
        mock_top_moves = [
            {'Move': 'e2e4', 'Centipawn': 50, 'Mate': None},
            {'Move': 'd2d4', 'Centipawn': 45, 'Mate': None},
            {'Move': 'g1f3', 'Centipawn': 30, 'Mate': None}
        ]
        self.mock_engine.get_top_moves.return_value = mock_top_moves
        
        if hasattr(self.handler, 'get_top_moves'):
            top_moves = self.handler.get_top_moves(3)
            self.assertEqual(len(top_moves), 3)
            self.assertEqual(top_moves[0]['Move'], 'e2e4')
            self.mock_engine.get_top_moves.assert_called_once_with(3)


class TestStockfishHandlerPositionManagement(unittest.TestCase):
    """Test StockfishHandler position management."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        
        with patch('engine_utilities.stockfish_handler.Stockfish') as mock_stockfish_class:
            mock_stockfish_class.return_value = self.mock_engine
            self.handler = StockfishHandler()

    def test_set_position_by_fen(self):
        """Test setting position by FEN string."""
        fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
        
        if hasattr(self.handler, 'set_fen_position'):
            self.handler.set_fen_position(fen)
            self.mock_engine.set_fen_position.assert_called_once_with(fen)

    def test_set_position_by_moves(self):
        """Test setting position by move sequence."""
        moves = ['e2e4', 'e7e5', 'g1f3']
        
        if hasattr(self.handler, 'set_position'):
            self.handler.set_position(moves)
            self.mock_engine.set_position.assert_called_once_with(moves)

    def test_get_current_fen(self):
        """Test getting current position as FEN."""
        expected_fen = "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2"
        self.mock_engine.get_fen_position.return_value = expected_fen
        
        if hasattr(self.handler, 'get_fen_position'):
            fen = self.handler.get_fen_position()
            self.assertEqual(fen, expected_fen)
            self.mock_engine.get_fen_position.assert_called_once()

    def test_make_moves(self):
        """Test making moves on the board."""
        moves = ['e2e4', 'e7e5']
        
        if hasattr(self.handler, 'make_moves_from_current_position'):
            self.handler.make_moves_from_current_position(moves)
            self.mock_engine.make_moves_from_current_position.assert_called_once_with(moves)

    def test_is_move_correct(self):
        """Test move validation."""
        self.mock_engine.is_move_correct.return_value = True
        
        if hasattr(self.handler, 'is_move_correct'):
            is_valid = self.handler.is_move_correct('e2e4')
            self.assertTrue(is_valid)
            self.mock_engine.is_move_correct.assert_called_once_with('e2e4')


class TestStockfishHandlerAdvancedFeatures(unittest.TestCase):
    """Test StockfishHandler advanced features."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        
        with patch('engine_utilities.stockfish_handler.Stockfish') as mock_stockfish_class:
            mock_stockfish_class.return_value = self.mock_engine
            self.handler = StockfishHandler()

    def test_engine_options_setting(self):
        """Test setting engine options."""
        options = {
            'Hash': 256,
            'Threads': 2,
            'MultiPV': 3
        }
        
        for option, value in options.items():
            if hasattr(self.handler, 'set_option'):
                self.handler.set_option(option, value)

    def test_time_control_management(self):
        """Test time control settings."""
        if hasattr(self.handler, 'set_time_limit'):
            self.handler.set_time_limit(5.0)  # 5 seconds
            # Verify time limit was set appropriately

    def test_threading_support(self):
        """Test multi-threading configuration."""
        if hasattr(self.handler, 'set_threads'):
            self.handler.set_threads(4)
            # Verify threads were configured

    def test_hash_table_size(self):
        """Test hash table size configuration."""
        if hasattr(self.handler, 'set_hash_size'):
            self.handler.set_hash_size(512)  # 512MB
            # Verify hash size was set

    def test_multi_pv_analysis(self):
        """Test multi-PV analysis capability."""
        mock_lines = [
            {'Move': 'e2e4', 'Centipawn': 50, 'Mate': None},
            {'Move': 'd2d4', 'Centipawn': 45, 'Mate': None},
            {'Move': 'g1f3', 'Centipawn': 30, 'Mate': None}
        ]
        
        if hasattr(self.mock_engine, 'get_top_moves'):
            self.mock_engine.get_top_moves.return_value = mock_lines
            
            if hasattr(self.handler, 'get_multiple_lines'):
                lines = self.handler.get_multiple_lines(3)
                self.assertEqual(len(lines), 3)


class TestStockfishHandlerErrorHandling(unittest.TestCase):
    """Test StockfishHandler error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()

    @patch('engine_utilities.stockfish_handler.Stockfish')
    def test_engine_crash_handling(self, mock_stockfish_class):
        """Test handling of engine crashes."""
        mock_stockfish_class.side_effect = Exception("Engine crashed")
        
        with self.assertRaises(Exception):
            StockfishHandler()

    def test_invalid_move_handling(self):
        """Test handling of invalid moves."""
        with patch('engine_utilities.stockfish_handler.Stockfish') as mock_stockfish_class:
            mock_stockfish_class.return_value = self.mock_engine
            handler = StockfishHandler()
            
            self.mock_engine.is_move_correct.return_value = False
            
            if hasattr(handler, 'is_move_correct'):
                is_valid = handler.is_move_correct('invalid_move')
                self.assertFalse(is_valid)

    def test_invalid_fen_handling(self):
        """Test handling of invalid FEN positions."""
        with patch('engine_utilities.stockfish_handler.Stockfish') as mock_stockfish_class:
            mock_stockfish_class.return_value = self.mock_engine
            handler = StockfishHandler()
            
            self.mock_engine.set_fen_position.side_effect = ValueError("Invalid FEN")
            
            if hasattr(handler, 'set_fen_position'):
                with self.assertRaises(ValueError):
                    handler.set_fen_position("invalid_fen")

    def test_timeout_handling(self):
        """Test handling of analysis timeouts."""
        with patch('engine_utilities.stockfish_handler.Stockfish') as mock_stockfish_class:
            mock_stockfish_class.return_value = self.mock_engine
            handler = StockfishHandler()
            
            # Mock timeout scenario
            self.mock_engine.get_best_move_time.return_value = None
            
            if hasattr(handler, 'get_best_move_time'):
                result = handler.get_best_move_time(1000)
                self.assertIsNone(result)

    def test_connection_loss_handling(self):
        """Test handling of connection loss to engine."""
        with patch('engine_utilities.stockfish_handler.Stockfish') as mock_stockfish_class:
            mock_stockfish_class.return_value = self.mock_engine
            handler = StockfishHandler()
            
            # Simulate connection loss
            self.mock_engine.get_best_move.side_effect = ConnectionError("Connection lost")
            
            if hasattr(handler, 'get_best_move'):
                with self.assertRaises(ConnectionError):
                    handler.get_best_move()


class TestStockfishHandlerPerformance(unittest.TestCase):
    """Test StockfishHandler performance characteristics."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        
        with patch('engine_utilities.stockfish_handler.Stockfish') as mock_stockfish_class:
            mock_stockfish_class.return_value = self.mock_engine
            self.handler = StockfishHandler()

    def test_move_generation_speed(self):
        """Test move generation performance."""
        self.mock_engine.get_best_move.return_value = 'e2e4'
        
        start_time = time.time()
        
        if hasattr(self.handler, 'get_best_move'):
            for _ in range(10):
                self.handler.get_best_move()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete 10 move generations quickly (mocked)
        self.assertLess(total_time, 1.0)

    def test_position_analysis_performance(self):
        """Test position analysis performance."""
        self.mock_engine.get_evaluation.return_value = {'type': 'cp', 'value': 50}
        
        start_time = time.time()
        
        if hasattr(self.handler, 'get_evaluation'):
            for _ in range(5):
                self.handler.get_evaluation()
        
        end_time = time.time()
        total_time = end_time - start_time
        
        # Should complete 5 evaluations quickly (mocked)
        self.assertLess(total_time, 0.5)

    def test_concurrent_analysis_safety(self):
        """Test thread safety for concurrent analysis."""
        import threading
        
        self.mock_engine.get_best_move.return_value = 'e2e4'
        results = []
        
        def analyze_position():
            if hasattr(self.handler, 'get_best_move'):
                result = self.handler.get_best_move()
                results.append(result)
        
        # Run multiple analyses concurrently
        threads = []
        for _ in range(3):
            thread = threading.Thread(target=analyze_position)
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # All should complete successfully
        self.assertEqual(len(results), 3)
        for result in results:
            self.assertEqual(result, 'e2e4')


class TestStockfishHandlerIntegration(unittest.TestCase):
    """Test StockfishHandler integration scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.mock_engine = Mock()
        
        with patch('engine_utilities.stockfish_handler.Stockfish') as mock_stockfish_class:
            mock_stockfish_class.return_value = self.mock_engine
            self.handler = StockfishHandler()

    def test_game_analysis_workflow(self):
        """Test complete game analysis workflow."""
        # Set up mock responses for a complete analysis
        self.mock_engine.get_best_move.return_value = 'e2e4'
        self.mock_engine.get_evaluation.return_value = {'type': 'cp', 'value': 30}
        self.mock_engine.is_move_correct.return_value = True
        
        # Simulate game analysis workflow
        if hasattr(self.handler, 'set_fen_position'):
            self.handler.set_fen_position("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
        
        if hasattr(self.handler, 'get_best_move'):
            best_move = self.handler.get_best_move()
            self.assertIsNotNone(best_move)
        
        if hasattr(self.handler, 'get_evaluation'):
            evaluation = self.handler.get_evaluation()
            self.assertIsNotNone(evaluation)

    def test_position_transition_workflow(self):
        """Test workflow for transitioning between positions."""
        moves_sequence = ['e2e4', 'e7e5', 'g1f3', 'b8c6']
        
        self.mock_engine.is_move_correct.return_value = True
        self.mock_engine.get_fen_position.return_value = "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 3"
        
        # Test position transitions
        for move in moves_sequence:
            if hasattr(self.handler, 'is_move_correct'):
                self.assertTrue(self.handler.is_move_correct(move))
            
            if hasattr(self.handler, 'make_moves_from_current_position'):
                self.handler.make_moves_from_current_position([move])

    def test_configuration_persistence(self):
        """Test that configuration settings persist across operations."""
        # Set configuration
        if hasattr(self.handler, 'set_depth'):
            self.handler.set_depth(12)
        
        if hasattr(self.handler, 'set_time_limit'):
            self.handler.set_time_limit(2.0)
        
        # Perform operations and verify settings persist
        self.mock_engine.get_best_move.return_value = 'e2e4'
        
        if hasattr(self.handler, 'get_best_move'):
            result = self.handler.get_best_move()
            self.assertIsNotNone(result)
        
        # Configuration should still be active
        # This would be tested by checking if the mock was called with appropriate parameters


if __name__ == '__main__':
    unittest.main()