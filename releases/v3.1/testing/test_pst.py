"""Test suite for v7p3r Piece-Square Tables module.
Tests piece placement evaluation and position-specific scoring."""

import os
import sys
import chess
import unittest
from typing import Dict

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from v7p3r_pst import v7p3rPST

class TestV7P3RPST(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.pst = v7p3rPST()
        
    def test_init_pst(self):
        """Test PST initialization."""
        self.assertIsNotNone(self.pst)
        self.assertIsNotNone(self.pst.piece_values)
        self.assertIsNotNone(self.pst.pst_tables)
        
    def test_piece_values(self):
        """Test basic piece values."""
        # Standard piece values
        self.assertEqual(self.pst.piece_values[chess.PAWN], 100)
        self.assertEqual(self.pst.piece_values[chess.KNIGHT], 320)
        self.assertEqual(self.pst.piece_values[chess.BISHOP], 330)
        self.assertEqual(self.pst.piece_values[chess.ROOK], 500)
        self.assertEqual(self.pst.piece_values[chess.QUEEN], 900)
        self.assertEqual(self.pst.piece_values[chess.KING], 20000)
        
    def test_pawn_pst(self):
        """Test pawn PST values."""
        # Central pawns should be worth more than edge pawns
        central_square = chess.E4
        edge_square = chess.A4
        
        central_value = self.pst.get_piece_square_value(chess.PAWN, central_square, chess.WHITE)
        edge_value = self.pst.get_piece_square_value(chess.PAWN, edge_square, chess.WHITE)
        
        self.assertGreater(central_value, edge_value)
        
    def test_knight_outpost(self):
        """Test knight positioning values."""
        # Knights should be worth more in the center
        central_square = chess.E4
        corner_square = chess.A1
        
        central_value = self.pst.get_piece_square_value(chess.KNIGHT, central_square, chess.WHITE)
        corner_value = self.pst.get_piece_square_value(chess.KNIGHT, corner_square, chess.WHITE)
        
        self.assertGreater(central_value, corner_value)
        
    def test_bishop_diagonal(self):
        """Test bishop positioning values."""
        # Bishops should be worth more on long diagonals
        long_diagonal = chess.C1
        blocked_square = chess.D2
        
        diagonal_value = self.pst.get_piece_square_value(chess.BISHOP, long_diagonal, chess.WHITE)
        blocked_value = self.pst.get_piece_square_value(chess.BISHOP, blocked_square, chess.WHITE)
        
        self.assertGreater(diagonal_value, blocked_value)
        
    def test_rook_seventh(self):
        """Test rook positioning values."""
        # Rooks should be worth more on the 7th rank
        seventh_rank = chess.E7
        first_rank = chess.E1
        
        seventh_value = self.pst.get_piece_square_value(chess.ROOK, seventh_rank, chess.WHITE)
        first_value = self.pst.get_piece_square_value(chess.ROOK, first_rank, chess.WHITE)
        
        self.assertGreater(seventh_value, first_value)
        
    def test_king_safety(self):
        """Test king positioning values."""
        # King should be worth more in protected corners in the opening/middlegame
        corner_square = chess.G1
        center_square = chess.E4
        
        corner_value = self.pst.get_piece_square_value(chess.KING, corner_square, chess.WHITE)
        center_value = self.pst.get_piece_square_value(chess.KING, center_square, chess.WHITE)
        
        self.assertGreater(corner_value, center_value)
        
    def test_color_perspective(self):
        """Test piece values from both colors' perspectives."""
        square = chess.E4
        
        white_value = self.pst.get_piece_square_value(chess.PAWN, square, chess.WHITE)
        black_value = self.pst.get_piece_square_value(chess.PAWN, chess.square_mirror(square), chess.BLACK)
        
        self.assertEqual(white_value, black_value)

if __name__ == '__main__':
    unittest.main()
