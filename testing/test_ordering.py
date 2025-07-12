"""Test suite for v7p3r move ordering system.
Tests move ordering, MVV-LVA, and tempo-aware ordering."""

import os
import sys
import chess
import unittest
from typing import List

# Add parent directory to path for imports
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from v7p3r_ordering import v7p3rOrdering
from v7p3r_score import v7p3rScore
from v7p3r_pst import v7p3rPST
from v7p3r_rules import v7p3rRules
from v7p3r_mvv_lva import v7p3rMVVLVA

class TestV7P3ROrdering(unittest.TestCase):
    def setUp(self):
        """Set up test fixtures."""
        self.pst = v7p3rPST()
        self.rules = v7p3rRules(pst=self.pst)
        self.score_calculator = v7p3rScore(
            rules_manager=self.rules,
            pst=self.pst
        )
        self.ordering = v7p3rOrdering(self.score_calculator)
        
    def test_init_ordering(self):
        """Test move ordering initialization."""
        self.assertIsNotNone(self.ordering)
        self.assertIsNotNone(self.ordering.mvv_lva)
        
    def test_capture_ordering(self):
        """Test ordering of capture moves."""
        # Position with multiple captures
        board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1")
        moves = self.ordering.order_moves(board)
        
        # Check if captures are ordered first
        first_move = moves[0]
        self.assertTrue(board.is_capture(first_move))
        
    def test_mvv_lva_ordering(self):
        """Test MVV-LVA ordering of captures."""
        # Position with queen and pawn captures
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
        moves = self.ordering.order_moves(board)
        
        # Find capture moves
        captures = [m for m in moves if board.is_capture(m)]
        if len(captures) >= 2:
            # Compare piece values of first two captures
            piece1 = board.piece_at(captures[0].to_square)
            piece2 = board.piece_at(captures[1].to_square)
            if piece1 and piece2:
                value1 = self.pst.piece_values[piece1.piece_type]
                value2 = self.pst.piece_values[piece2.piece_type]
                self.assertGreaterEqual(value1, value2)
                
    def test_tempo_aware_ordering(self):
        """Test tempo-aware move ordering."""
        board = chess.Board()
        # Order moves with tempo bonus
        moves_with_tempo = self.ordering.order_moves(board, tempo_bonus=1.0)
        # Order moves without tempo bonus
        moves_without_tempo = self.ordering.order_moves(board, tempo_bonus=0.0)
        
        # Orders should be different with tempo consideration
        self.assertNotEqual(moves_with_tempo, moves_without_tempo)
        
    def test_history_heuristic(self):
        """Test history heuristic in move ordering."""
        board = chess.Board()
        # Make some moves to build history
        moves = list(board.legal_moves)
        if moves:
            move = moves[0]
            self.ordering.update_history_score(move, 2)  # Update history score
            
            # Get ordered moves
            ordered_moves = self.ordering.order_moves(board)
            # First move should be the one with history
            self.assertEqual(ordered_moves[0], move)
            
    def test_killer_move_ordering(self):
        """Test killer move ordering."""
        board = chess.Board()
        # Set up a killer move
        moves = list(board.legal_moves)
        if moves:
            killer = moves[0]
            self.ordering.add_killer_move(killer, depth=2)
            
            # Get ordered moves
            ordered_moves = self.ordering.order_moves(board)
            # Killer move should be among first moves
            self.assertIn(killer, ordered_moves[:3])

if __name__ == '__main__':
    unittest.main()
