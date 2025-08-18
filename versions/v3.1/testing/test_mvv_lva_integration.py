import unittest
import chess
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from v7p3r_score import v7p3rScore
from v7p3r_search import v7p3rSearch
from v7p3r_utilities import get_timestamp
from v7p3r_config import v7p3rConfig
from v7p3r_rules import v7p3rRules
from v7p3r_pst import v7p3rPST
from v7p3r_book import v7p3rBook
from v7p3r_time import v7p3rTime

class TestMVVLVAIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test environment with engine components."""
        self.config = v7p3rConfig()
        self.rules = v7p3rRules()
        self.pst = v7p3rPST()
        self.book = v7p3rBook()
        self.time = v7p3rTime()
        self.score = v7p3rScore(self.rules, self.pst)
        
        # Update search class initialization with correct arguments
        self.search = v7p3rSearch(
            scoring_calculator=self.score,
            time_manager=self.time
        )

    def test_move_ordering_integration(self):
        """Test that MVV-LVA move ordering is properly integrated in search."""
        # Position with multiple captures
        board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 3")
        moves = list(board.legal_moves)
        
        # Get ordered moves
        ordered_moves = self.search._order_moves(moves, board)
        
        # Verify captures are prioritized
        first_move = ordered_moves[0]
        self.assertTrue(board.is_capture(first_move), "First move should be a capture")

    def test_evaluation_integration(self):
        """Test that MVV-LVA scoring is properly integrated in position evaluation."""
        # Position with tactical opportunities
        board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/3PP3/5N2/PPP2PPP/RNBQKB1R b KQkq - 0 3")
        
        # Get position score
        score = self.score.evaluate_position(board)
        
        # Verify tactical score is included
        self.assertIn('piece_captures_score', self.score.score_dataset)
        self.assertNotEqual(self.score.score_dataset['piece_captures_score'], 0.0)

    def test_search_with_mvv_lva(self):
        """Test that MVV-LVA affects move ordering in search."""
        # Position with captures
        board = chess.Board("rnbqkbnr/ppp2ppp/3p4/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 0 4")
        moves = list(board.legal_moves)
        
        # Get ordered moves
        ordered_moves = self.search._order_moves(moves, board)
        
        # Find capture moves
        capture_moves = [m for m in moves if board.is_capture(m)]
        
        # Verify captures are ordered first
        self.assertTrue(any(board.is_capture(m) for m in ordered_moves[:len(capture_moves)]))

if __name__ == '__main__':
    unittest.main()
