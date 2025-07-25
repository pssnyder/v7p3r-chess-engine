"""Test suite for v7p3r scoring system.
Tests position evaluation, material counting, and scoring components."""

import os
import sys
import chess
import unittest
from typing import Dict, Any

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from v7p3r_score import v7p3rScore
from v7p3r_pst import v7p3rPST
from v7p3r_rules import v7p3rRules

class TestV7P3RScore(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.pst = v7p3rPST()
        self.rules = v7p3rRules(pst=self.pst)
        self.score_calculator = v7p3rScore(
            rules_manager=self.rules,
            pst=self.pst
        )
        
    def test_init_scoring(self):
        """Test scoring system initialization."""
        self.assertIsNotNone(self.score_calculator)
        self.assertIsNotNone(self.score_calculator.pst)
        self.assertIsNotNone(self.score_calculator.tempo)
        
    def test_material_evaluation(self):
        """Test basic material evaluation."""
        board = chess.Board()
        score = self.score_calculator._evaluate_material(board)
        # Starting position should be equal
        self.assertEqual(score, 0)
        
        # Test after capturing a pawn
        board.set_fen("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
        board.push_san("exd5")  # Capture black's e-pawn
        score = self.score_calculator._evaluate_material(board)
        self.assertGreater(score, 0)  # White should be up material
        
    def test_position_evaluation(self):
        """Test full position evaluation."""
        board = chess.Board()
        white_score = self.score_calculator.evaluate_position(board, chess.WHITE)
        black_score = self.score_calculator.evaluate_position(board, chess.BLACK)
        # Starting position should be roughly equal
        self.assertAlmostEqual(white_score, -black_score, places=2)
        
    def test_piece_placement_scoring(self):
        """Test piece-square table evaluation."""
        board = chess.Board()
        # Move knight to better square
        board.push_san("Nf3")
        score_after = self.score_calculator.evaluate_position(board, chess.WHITE)
        self.assertGreater(score_after, 0)  # Knight on f3 should be positive
        
    def test_pawn_structure_evaluation(self):
        """Test pawn structure evaluation."""
        # Test doubled pawns
        board = chess.Board("rnbqkbnr/ppp1pppp/8/8/3p4/3P4/PPP1PPPP/RNBQKBNR w KQkq - 0 1")
        score = self.score_calculator.evaluate_position(board, chess.WHITE)
        # Doubled pawns should be penalized
        self.assertLess(score, 0)
        
    def test_king_safety(self):
        """Test king safety evaluation."""
        # Test exposed king
        board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
        board.push_san("Ke2")  # Expose king
        score = self.score_calculator.evaluate_position(board, chess.WHITE)
        # Exposed king should be penalized
        self.assertLess(score, 0)

if __name__ == '__main__':
    unittest.main()
