import unittest
import chess
import os
import sys
from typing import List, Optional

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from v7p3r_score import v7p3rScore
from v7p3r_search import v7p3rSearch
from v7p3r_ordering import v7p3rOrdering
from v7p3r_tempo import v7p3rTempo
from v7p3r_config import v7p3rConfig
from v7p3r_rules import v7p3rRules
from v7p3r_pst import v7p3rPST
from v7p3r_book import v7p3rBook
from v7p3r_time import v7p3rTime

class TestTempoIntegration(unittest.TestCase):
    def setUp(self):
        """Set up test environment with engine components."""
        self.config = v7p3rConfig()
        self.pst = v7p3rPST()
        self.rules = v7p3rRules(pst=self.pst)
        self.book = v7p3rBook()
        self.time = v7p3rTime()
        
        # Initialize in correct dependency order
        self.score = v7p3rScore(self.rules, self.pst)
        self.ordering = v7p3rOrdering(self.score)
        
        # Update search class initialization with correct arguments
        self.search = v7p3rSearch(
            scoring_calculator=self.score,
            time_manager=self.time
        )

    def test_tempo_aware_move_ordering(self):
        """Test that moves are ordered considering tempo."""
        # Position where White has a development and center advantage
        board = chess.Board("rnbqkbnr/pppppppp/8/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2")
        
        # Get ordered moves without tempo consideration
        moves_without_tempo = self.ordering.order_moves(board, tempo_bonus=0.0)
        
        # Get ordered moves with tempo consideration
        tempo_score = self.score.tempo.assess_tempo(board, chess.BLACK)
        moves_with_tempo = self.ordering.order_moves(board, tempo_bonus=tempo_score)
        
        # Verify moves are ordered differently
        self.assertNotEqual(
            [m.uci() for m in moves_without_tempo],
            [m.uci() for m in moves_with_tempo],
            "Move ordering should be affected by tempo consideration"
        )
        
        # Verify equalizing moves are prioritized
        first_move = moves_with_tempo[0]
        self.assertTrue(
            chess.square_file(first_move.from_square) in [2, 3, 4, 5] or
            chess.square_file(first_move.to_square) in [2, 3, 4, 5],
            "First move should target or control central squares"
        )

    def test_search_depth_adaptation(self):
        """Test that search depth adapts based on game phase and tempo."""
        # Opening position
        board = chess.Board()
        move = self.search.search(board, chess.WHITE)
        self.assertEqual(self.search.depth, 3)  # Default depth
        
        # Endgame position with critical tempo
        board = chess.Board("4k3/4P3/8/8/8/8/8/4K3 w - - 0 1")  # White to move, pawn about to queen
        move = self.search.search(board, chess.WHITE)
        self.assertEqual(self.search.depth, 5)  # Should search deeper

    def test_position_history_tracking(self):
        """Test that position history is properly tracked for draw prevention."""
        board = chess.Board()
        
        # Play a series of repetitive moves
        moves = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"]
        for uci in moves:
            move = chess.Move.from_uci(uci)
            board.push(move)
            # Update history after each move
            self.score.tempo.update_position_history(board.fen())
        
        # Check repetition count
        rep_count = self.score.tempo.get_repetition_count(board.fen())
        self.assertEqual(rep_count, 3, "Position should be repeated 3 times")
        
        # Verify draw prevention logic
        move = self.search.search(board, board.turn)
        self.assertNotEqual(
            move.uci(),
            "Ng1",
            "Search should avoid move leading to repetition"
        )

    def test_integration_full_position(self):
        """Test full integration in a complex middlegame position."""
        # Position with multiple tactical and positional possibilities
        fen = "r2qk2r/ppp2ppp/2n1bn2/3p4/3P4/2N2N2/PPP1BPPP/R1BQ1RK1 w kq - 0 9"
        board = chess.Board(fen)
        
        # Get the engine's move
        move = self.search.search(board, chess.WHITE)
        
        # Verify move meets basic requirements
        self.assertIsNotNone(move, "Search should return a valid move")
        if move:  # Add type guard for move
            self.assertNotEqual(move, chess.Move.null())
            self.assertTrue(move in board.legal_moves)
            
            # Make the move and verify it doesn't worsen the position
            eval_before = self.score.evaluate_position(board)
            board.push(move)
            eval_after = -self.score.evaluate_position(board)  # Negate because it's from opponent's perspective
        
        self.assertGreaterEqual(
            eval_after,
            eval_before - 0.5,  # Allow small evaluation drop for positional compensation
            "Move should not significantly worsen the position"
        )

if __name__ == '__main__':
    unittest.main()
