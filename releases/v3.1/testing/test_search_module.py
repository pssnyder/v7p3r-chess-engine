"""
Test suite for v7p3r_search module functionality.
This covers core search algorithms, move ordering, and integration with other components.
"""

import os
import sys
import chess
import unittest
from typing import Optional, List

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from v7p3r_search import v7p3rSearch
from v7p3r_score import v7p3rScore
from v7p3r_time import v7p3rTime
from v7p3r_config import v7p3rConfig
from v7p3r_rules import v7p3rRules
from v7p3r_pst import v7p3rPST

class TestV7P3RSearch(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures for each test method."""
        self.config_manager = v7p3rConfig()
        self.pst = v7p3rPST()
        self.rules_manager = v7p3rRules(pst=self.pst)
        self.scoring_calculator = v7p3rScore(
            rules_manager=self.rules_manager,
            pst=self.pst
        )
        self.time_manager = v7p3rTime()
        self.search_engine = v7p3rSearch(
            scoring_calculator=self.scoring_calculator,
            time_manager=self.time_manager
        )

    def test_init_configuration(self):
        """Test proper initialization of search module configuration."""
        self.assertIsNotNone(self.search_engine.scoring_calculator)
        self.assertIsNotNone(self.search_engine.time_manager)
        self.assertIsNotNone(self.search_engine.ordering)
        self.assertTrue(self.search_engine.move_ordering_enabled)
        self.assertIsNotNone(self.search_engine.tempo)

    def test_search_starting_position(self):
        """Test search functionality from starting position."""
        board = chess.Board()
        move = self.search_engine.search(board, chess.WHITE)
        self.assertIsNotNone(move)
        self.assertTrue(isinstance(move, chess.Move))
        self.assertIn(move, list(board.legal_moves))

    def test_negamax_return_values(self):
        """Test negamax returns valid evaluation scores."""
        board = chess.Board()
        color = chess.WHITE
        depth = 3
        alpha = float('-inf')
        beta = float('inf')
        
        score = self._negamax_wrapper(board, depth, alpha, beta, color)
        self.assertIsInstance(score, float)
        self.assertTrue(-10000 <= score <= 10000)  # Reasonable score range

    def test_iterative_deepening(self):
        """Test iterative deepening search functionality."""
        board = chess.Board()
        move = self.search_engine.iterative_deepening_search(board, chess.WHITE)
        self.assertIsNotNone(move)
        self.assertTrue(isinstance(move, chess.Move))
        self.assertIn(move, list(board.legal_moves))

    def test_checkmate_detection(self):
        """Test checkmate detection in search."""
        # Set up a basic checkmate in one position
        board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 0 1")
        move = self.search_engine.find_checkmate_in_n(board, 1)
        self.assertIsNotNone(move, "Expected to find checkmate move but got None")
        self.assertTrue(isinstance(move, chess.Move))
        # Verify it's actually checkmate
        if move:  # Type guard for mypy
            board_copy = board.copy()
            board_copy.push(move)
            self.assertTrue(board_copy.is_checkmate())

    def test_quiescence_search(self):
        """Test quiescence search handles captures properly."""
        # Position with captures available
        board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 0 1")
        alpha = float('-inf')
        beta = float('inf')
        score = self.search_engine._quiescence_search(board, alpha, beta, chess.WHITE)
        self.assertIsInstance(score, float)

    def test_move_ordering(self):
        """Test move ordering prioritizes captures and good moves."""
        board = chess.Board()
        tempo_bonus = self.search_engine.scoring_calculator.tempo.assess_tempo(board, chess.WHITE)
        ordered_moves = self.search_engine.ordering.order_moves(
            board,
            max_moves=None,
            tempo_bonus=tempo_bonus
        )
        self.assertIsInstance(ordered_moves, list)
        self.assertTrue(len(ordered_moves) > 0)

    def _negamax_wrapper(self, board: chess.Board, depth: int, alpha: float, beta: float, color: chess.Color) -> float:
        """Wrapper to safely test negamax function."""
        try:
            return self.search_engine._negamax(board, depth, alpha, beta, color)
        except Exception as e:
            self.fail(f"Negamax failed with error: {str(e)}")

if __name__ == '__main__':
    unittest.main()
