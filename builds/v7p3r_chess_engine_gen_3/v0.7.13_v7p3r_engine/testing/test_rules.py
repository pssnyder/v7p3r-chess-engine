"""Test suite for v7p3r Rules module.
Tests chess move generation rules and game state validation."""

import os
import sys
import chess
import unittest
from typing import List, Dict

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from v7p3r_rules import v7p3rRules
from v7p3r_pst import v7p3rPST

class TestV7P3RRules(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.pst = v7p3rPST()
        self.rules = v7p3rRules(pst=self.pst)
        
    def test_init_rules(self):
        """Test rules initialization."""
        self.assertIsNotNone(self.rules)
        self.assertIsNotNone(self.rules.pst)
        
    def test_legal_moves(self):
        """Test legal move generation."""
        board = chess.Board()
        legal_moves = list(board.legal_moves)
        # Standard initial position should have 20 moves
        self.assertEqual(len(legal_moves), 20)
        
    def test_check_detection(self):
        """Test check detection."""
        # Set up a position with check
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
        board.push_san("Qh5")  # Put black in check
        
        self.assertTrue(board.is_check())
        
    def test_checkmate_detection(self):
        """Test checkmate detection."""
        # Scholar's mate position
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/4P3/5Q2/PPPP1PPP/RNB1KBNR b KQkq - 0 1")
        board.push_san("Ke7")
        board.push_san("Qf7")
        
        self.assertTrue(board.is_checkmate())
        
    def test_stalemate_detection(self):
        """Test stalemate detection."""
        # Common stalemate position
        board = chess.Board("k7/8/1Q6/8/8/8/8/K7 b - - 0 1")
        self.assertTrue(board.is_stalemate())
        
    def test_insufficient_material(self):
        """Test insufficient material detection."""
        # King vs King
        board = chess.Board("k7/8/8/8/8/8/8/K7 w - - 0 1")
        self.assertTrue(board.is_insufficient_material())
        
        # King and Bishop vs King
        board = chess.Board("k7/8/8/8/8/8/B7/K7 w - - 0 1")
        self.assertTrue(board.is_insufficient_material())
        
    def test_castling_rights(self):
        """Test castling rights tracking."""
        board = chess.Board()
        # Initial position should have all castling rights
        self.assertTrue(board.has_kingside_castling_rights(chess.WHITE))
        self.assertTrue(board.has_queenside_castling_rights(chess.WHITE))
        self.assertTrue(board.has_kingside_castling_rights(chess.BLACK))
        self.assertTrue(board.has_queenside_castling_rights(chess.BLACK))
        
        # Move rook, lose castling rights
        board.push_san("Ra4")
        self.assertFalse(board.has_queenside_castling_rights(chess.WHITE))
        
    def test_promotion_moves(self):
        """Test pawn promotion move generation."""
        # Set up a position with possible promotion
        board = chess.Board("8/P7/8/8/8/8/8/k6K w - - 0 1")
        promotion_moves = [move for move in board.legal_moves if move.promotion]
        
        # Should have 4 possible promotions (Q, R, B, N)
        self.assertEqual(len(promotion_moves), 4)
        promotion_types = set(move.promotion for move in promotion_moves)
        self.assertEqual(len(promotion_types), 4)
        
    def test_en_passant(self):
        """Test en passant move generation."""
        # Set up en passant position
        board = chess.Board("rnbqkbnr/ppp2ppp/8/3pP3/8/8/PPPP1PPP/RNBQKBNR w KQkq d6 0 1")
        en_passant_moves = [move for move in board.legal_moves if board.is_en_passant(move)]
        
        # Should have one en passant capture available
        self.assertEqual(len(en_passant_moves), 1)

if __name__ == '__main__':
    unittest.main()
