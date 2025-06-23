#!/usr/bin/env python3
"""
Unit tests for opening_book.py - v7p3r Chess Engine Opening Book

This module contains comprehensive unit tests for the OpeningBook class,
testing book population, move selection, and opening theory integration.

Author: v7p3r Testing Suite
Date: 2025-06-22
"""

import sys
import os
import unittest
import tempfile
from unittest.mock import Mock, patch, MagicMock
import random

# Add parent directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

try:
    import chess
    from engine_utilities.opening_book import OpeningBook
    CHESS_AVAILABLE = True
except ImportError:
    CHESS_AVAILABLE = False


@unittest.skipUnless(CHESS_AVAILABLE, "python-chess library not available")
class TestOpeningBookInitialization(unittest.TestCase):
    """Test OpeningBook initialization."""

    def test_init_creates_book(self):
        """Test that initialization creates opening book."""
        book = OpeningBook()
        self.assertIsNotNone(book.book)
        self.assertIsInstance(book.book, dict)

    def test_book_population(self):
        """Test that book is populated with openings."""
        book = OpeningBook()
        
        # Should have entries for starting position
        start_board = chess.Board()
        start_fen = start_board.fen()
        
        self.assertIn(start_fen, book.book)
        self.assertGreater(len(book.book[start_fen]), 0)

    def test_starting_position_moves(self):
        """Test moves available from starting position."""
        book = OpeningBook()
        start_board = chess.Board()
        start_fen = start_board.fen()
        
        moves = book.book[start_fen]
        self.assertIsInstance(moves, list)
        
        # Each move should be a tuple of (move, weight)
        for move, weight in moves:
            self.assertIsInstance(move, chess.Move)
            self.assertIsInstance(weight, (int, float))
            self.assertGreater(weight, 0)

    def test_common_openings_included(self):
        """Test that common openings are included."""
        book = OpeningBook()
        start_board = chess.Board()
        start_fen = start_board.fen()
        
        moves = [move for move, _ in book.book[start_fen]]
        move_ucis = [move.uci() for move in moves]
        
        # Check for common first moves
        expected_moves = ['e2e4', 'd2d4', 'c2c4', 'g1f3']
        for expected in expected_moves:
            self.assertIn(expected, move_ucis)


@unittest.skipUnless(CHESS_AVAILABLE, "python-chess library not available")
class TestOpeningBookMoveSelection(unittest.TestCase):
    """Test OpeningBook move selection functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.book = OpeningBook()

    def test_get_book_move_from_starting_position(self):
        """Test getting book move from starting position."""
        board = chess.Board()
        
        if hasattr(self.book, 'get_book_move'):
            move = self.book.get_book_move(board)
            
            if move is not None:
                self.assertIsInstance(move, chess.Move)
                self.assertIn(move, board.legal_moves)

    def test_get_book_move_weighted_selection(self):
        """Test that move selection respects weights."""
        board = chess.Board()
        
        if hasattr(self.book, 'get_book_move'):
            # Get multiple moves to test weight distribution
            moves = []
            for _ in range(100):  # Large sample for statistical significance
                move = self.book.get_book_move(board)
                if move:
                    moves.append(move.uci())
            
            if moves:
                # e2e4 and d2d4 should be most common (equal weights)
                e4_count = moves.count('e2e4')
                d4_count = moves.count('d2d4')
                
                # Should have reasonable distribution
                self.assertGreater(e4_count + d4_count, len(moves) * 0.5)

    def test_get_book_move_after_e4(self):
        """Test getting book move after 1.e4."""
        board = chess.Board()
        board.push(chess.Move.from_uci("e2e4"))
        
        if hasattr(self.book, 'get_book_move'):
            move = self.book.get_book_move(board)
            
            if move is not None:
                self.assertIsInstance(move, chess.Move)
                self.assertIn(move, board.legal_moves)
                
                # Should be a reasonable response to e4
                reasonable_responses = ['e7e5', 'c7c5', 'e7e6', 'c7c6']
                self.assertIn(move.uci(), reasonable_responses)

    def test_get_book_move_after_d4(self):
        """Test getting book move after 1.d4."""
        board = chess.Board()
        board.push(chess.Move.from_uci("d2d4"))
        
        if hasattr(self.book, 'get_book_move'):
            move = self.book.get_book_move(board)
            
            if move is not None:
                self.assertIsInstance(move, chess.Move)
                self.assertIn(move, board.legal_moves)
                
                # Should be a reasonable response to d4
                reasonable_responses = ['d7d5', 'g8f6', 'f7f5', 'e7e6']
                self.assertIn(move.uci(), reasonable_responses)

    def test_get_book_move_unknown_position(self):
        """Test getting book move from unknown position."""
        board = chess.Board()
        # Make several moves to get out of book
        moves = ['e2e4', 'e7e5', 'g1f3', 'g8f6', 'f1c4', 'f8c5', 'b2b3']
        
        for move_uci in moves:
            board.push(chess.Move.from_uci(move_uci))
        
        if hasattr(self.book, 'get_book_move'):
            move = self.book.get_book_move(board)
            # Should return None for positions not in book
            self.assertIsNone(move)

    def test_random_selection_consistency(self):
        """Test that random selection is consistent with weights."""
        board = chess.Board()
        
        if hasattr(self.book, 'get_book_move'):
            # Set random seed for reproducibility
            random.seed(42)
            
            moves = []
            for _ in range(50):
                move = self.book.get_book_move(board)
                if move:
                    moves.append(move.uci())
            
            # Should have variety but follow weight distribution
            unique_moves = set(moves)
            self.assertGreater(len(unique_moves), 1)  # Should have variety


@unittest.skipUnless(CHESS_AVAILABLE, "python-chess library not available")
class TestOpeningBookAdvancedFeatures(unittest.TestCase):
    """Test OpeningBook advanced features."""

    def setUp(self):
        """Set up test fixtures."""
        self.book = OpeningBook()

    def test_position_lookup_by_fen(self):
        """Test position lookup using FEN strings."""
        start_fen = chess.Board().fen()
        self.assertIn(start_fen, self.book.book)

    def test_move_weight_system(self):
        """Test move weight system."""
        start_board = chess.Board()
        start_fen = start_board.fen()
        
        moves = self.book.book[start_fen]
        
        # Check that weights are reasonable
        for move, weight in moves:
            self.assertGreater(weight, 0)
            self.assertLessEqual(weight, 100)  # Reasonable upper bound

    def test_opening_transpositions(self):
        """Test handling of opening transpositions."""
        # Test if different move orders lead to same position
        board1 = chess.Board()
        board1.push(chess.Move.from_uci("g1f3"))
        board1.push(chess.Move.from_uci("d7d5"))
        board1.push(chess.Move.from_uci("d2d4"))
        
        board2 = chess.Board()
        board2.push(chess.Move.from_uci("d2d4"))
        board2.push(chess.Move.from_uci("d7d5"))
        board2.push(chess.Move.from_uci("g1f3"))
        
        # Same position should have same book treatment
        self.assertEqual(board1.fen(), board2.fen())

    def test_book_coverage_depth(self):
        """Test opening book coverage depth."""
        # Test that book has reasonable coverage depth
        positions_in_book = len(self.book.book)
        self.assertGreater(positions_in_book, 5)  # Should have multiple positions

    def test_symmetrical_openings(self):
        """Test handling of symmetrical openings."""
        # Test positions after common symmetrical sequences
        board = chess.Board()
        moves = ['e2e4', 'e7e5', 'd2d4', 'd7d5']  # Center game-like
        
        for move_uci in moves[:2]:  # Just first two moves
            board.push(chess.Move.from_uci(move_uci))
        
        # Should have book moves for common symmetrical positions
        if board.fen() in self.book.book:
            moves_available = self.book.book[board.fen()]
            self.assertGreater(len(moves_available), 0)


@unittest.skipUnless(CHESS_AVAILABLE, "python-chess library not available")
class TestOpeningBookErrorHandling(unittest.TestCase):
    """Test OpeningBook error handling."""

    def setUp(self):
        """Set up test fixtures."""
        self.book = OpeningBook()

    def test_invalid_board_state(self):
        """Test handling of invalid board states."""
        if hasattr(self.book, 'get_book_move'):
            # Test with None
            result = self.book.get_book_move(None)
            # Should handle gracefully
            self.assertIsNone(result)

    def test_corrupted_book_data(self):
        """Test handling of corrupted book data."""
        # Temporarily corrupt book data
        original_book = self.book.book.copy()
        self.book.book['invalid_fen'] = [('invalid_move', 10)]
        
        board = chess.Board('invalid_fen')  # This will raise exception
        
        # Restore original book
        self.book.book = original_book

    def test_empty_move_list(self):
        """Test handling of empty move lists."""
        # Create position with empty move list
        test_fen = "8/8/8/8/8/8/8/8 w - - 0 1"  # Empty board
        self.book.book[test_fen] = []
        
        try:
            board = chess.Board(test_fen)
            if hasattr(self.book, 'get_book_move'):
                move = self.book.get_book_move(board)
                self.assertIsNone(move)
        except ValueError:
            pass  # Invalid FEN expected

    def test_invalid_move_objects(self):
        """Test handling of invalid move objects in book."""
        start_fen = chess.Board().fen()
        original_moves = self.book.book[start_fen].copy()
        
        # Add invalid move object
        self.book.book[start_fen].append((None, 10))
        
        if hasattr(self.book, 'get_book_move'):
            try:
                board = chess.Board()
                move = self.book.get_book_move(board)
                # Should handle gracefully
            except Exception:
                pass  # Some error handling expected
        
        # Restore original moves
        self.book.book[start_fen] = original_moves


@unittest.skipUnless(CHESS_AVAILABLE, "python-chess library not available")
class TestOpeningBookPerformance(unittest.TestCase):
    """Test OpeningBook performance characteristics."""

    def setUp(self):
        """Set up test fixtures."""
        self.book = OpeningBook()

    def test_initialization_performance(self):
        """Test opening book initialization performance."""
        import time
        
        start_time = time.time()
        new_book = OpeningBook()
        end_time = time.time()
        
        initialization_time = end_time - start_time
        
        # Should initialize quickly (under 1 second)
        self.assertLess(initialization_time, 1.0)

    def test_move_lookup_performance(self):
        """Test move lookup performance."""
        import time
        
        board = chess.Board()
        
        if hasattr(self.book, 'get_book_move'):
            start_time = time.time()
            
            # Perform many lookups
            for _ in range(1000):
                self.book.get_book_move(board)
            
            end_time = time.time()
            lookup_time = end_time - start_time
            
            # Should be very fast (under 0.1 seconds for 1000 lookups)
            self.assertLess(lookup_time, 0.1)

    def test_memory_usage(self):
        """Test memory usage of opening book."""
        import sys
        
        book_size = sys.getsizeof(self.book.book)
        
        # Should not use excessive memory (under 1MB for basic book)
        self.assertLess(book_size, 1024 * 1024)  # 1MB

    def test_position_coverage_efficiency(self):
        """Test efficiency of position coverage."""
        # Count positions vs moves ratio
        total_positions = len(self.book.book)
        total_moves = sum(len(moves) for moves in self.book.book.values())
        
        if total_positions > 0:
            moves_per_position = total_moves / total_positions
            
            # Should have reasonable move density
            self.assertGreater(moves_per_position, 1.0)  # At least 1 move per position
            self.assertLess(moves_per_position, 10.0)    # Not too many moves per position


@unittest.skipUnless(CHESS_AVAILABLE, "python-chess library not available")
class TestOpeningBookIntegration(unittest.TestCase):
    """Test OpeningBook integration scenarios."""

    def setUp(self):
        """Set up test fixtures."""
        self.book = OpeningBook()

    def test_integration_with_chess_engine(self):
        """Test integration with chess engine workflow."""
        board = chess.Board()
        
        # Simulate engine consultation workflow
        if hasattr(self.book, 'get_book_move'):
            book_move = self.book.get_book_move(board)
            
            if book_move:
                # Verify move is legal
                self.assertIn(book_move, board.legal_moves)
                
                # Make the move
                board.push(book_move)
                
                # Check for follow-up book move
                followup_move = self.book.get_book_move(board)
                if followup_move:
                    self.assertIn(followup_move, board.legal_moves)

    def test_opening_sequence_consistency(self):
        """Test consistency of opening sequences."""
        board = chess.Board()
        opening_sequence = []
        
        if hasattr(self.book, 'get_book_move'):
            # Play out opening sequence
            for _ in range(6):  # Up to 3 moves per side
                book_move = self.book.get_book_move(board)
                if book_move:
                    opening_sequence.append(book_move.uci())
                    board.push(book_move)
                else:
                    break
            
            # Should have played some opening moves
            self.assertGreater(len(opening_sequence), 0)
            
            # All moves should form valid opening
            test_board = chess.Board()
            for move_uci in opening_sequence:
                move = chess.Move.from_uci(move_uci)
                self.assertIn(move, test_board.legal_moves)
                test_board.push(move)

    def test_book_move_vs_random_move(self):
        """Test that book moves are different from random moves."""
        board = chess.Board()
        
        if hasattr(self.book, 'get_book_move'):
            book_move = self.book.get_book_move(board)
            
            if book_move:
                # Get random legal move
                legal_moves = list(board.legal_moves)
                random_move = random.choice(legal_moves)
                
                # Book move should be one of the expected good moves
                expected_good_moves = [
                    chess.Move.from_uci("e2e4"),
                    chess.Move.from_uci("d2d4"),
                    chess.Move.from_uci("c2c4"),
                    chess.Move.from_uci("g1f3")
                ]
                
                self.assertIn(book_move, expected_good_moves)


if __name__ == '__main__':
    unittest.main()