import unittest
import chess
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from v7p3r_ordering import v7p3rOrdering
from v7p3r_score import v7p3rScore
from v7p3r_rules import v7p3rRules
from v7p3r_pst import v7p3rPST

class TestModuleOrganization(unittest.TestCase):
    def setUp(self):
        """Set up test environment with engine components."""
        self.rules = v7p3rRules()
        self.pst = v7p3rPST()
        self.score = v7p3rScore(self.rules, self.pst)
        self.ordering = v7p3rOrdering(self.score)

    def test_move_ordering_integration(self):
        """Test that move ordering properly uses MVV-LVA and tactical patterns."""
        # Position with captures and tactical opportunities
        board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 3")
        moves = list(board.legal_moves)
        
        # Get ordered moves
        ordered_moves = self.ordering.sort_moves(moves, board)
        
        # Verify captures are prioritized
        first_move = ordered_moves[0]
        self.assertTrue(board.is_capture(first_move), "First move should be a capture")

    def test_history_move_integration(self):
        """Test that history moves are properly integrated in move ordering."""
        board = chess.Board()
        moves = list(board.legal_moves)
        
        # Simulate some history for a move
        test_move = chess.Move.from_uci("e2e4")
        self.ordering._update_history_score(test_move, depth=3)
        
        # Get ordered moves
        ordered_moves = self.ordering.sort_moves(moves, board)
        
        # Verify the move with history gets priority
        self.assertEqual(ordered_moves[0], test_move, "Move with history should be prioritized")

    def test_killer_move_integration(self):
        """Test that killer moves are properly integrated in move ordering."""
        board = chess.Board()
        moves = list(board.legal_moves)
        
        # Add a killer move
        test_move = chess.Move.from_uci("e2e4")
        self.ordering.killer_moves[0].append(test_move)
        
        # Get ordered moves
        ordered_moves = self.ordering.sort_moves(moves, board)
        
        # Verify the killer move gets priority
        self.assertEqual(ordered_moves[0], test_move, "Killer move should be prioritized")

if __name__ == '__main__':
    unittest.main()
