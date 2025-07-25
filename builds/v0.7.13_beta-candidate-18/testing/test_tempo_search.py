import unittest
import chess
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from v7p3r_config import v7p3rConfig
from v7p3r_pst import v7p3rPST
from v7p3r_rules import v7p3rRules
from v7p3r_score import v7p3rScore
from v7p3r_ordering import v7p3rOrdering
from v7p3r_search import v7p3rSearch
from v7p3r_book import v7p3rBook
from v7p3r_time import v7p3rTime

class TestTempoSearch(unittest.TestCase):
    def setUp(self):
        """Set up test environment with all engine components."""
        # Core engine components
        self.config = v7p3rConfig()
        self.pst = v7p3rPST()
        self.rules = v7p3rRules(pst=self.pst)
        self.book = v7p3rBook()
        self.time = v7p3rTime()
        
        # Configure components with tempo awareness
        engine_config = {
            'use_game_phase': True,
            'use_tempo': True,
            'use_mvv_lva': True,
            'use_move_ordering': True,
            'depth': 3,
            'max_depth': 5
        }
        
        # Initialize in dependency order
        self.score = v7p3rScore(self.rules, self.pst)
        self.ordering = v7p3rOrdering(self.score)
        
        # Update search class initialization with correct arguments
        self.search = v7p3rSearch(
            scoring_calculator=self.score,
            time_manager=self.time
        )
        # Update config after initialization
        self.search.engine_config.update(engine_config)

    def test_depth_adaptation(self):
        """Test that search depth increases in critical positions."""
        # Endgame position with clear win
        board = chess.Board("4k3/4P3/8/8/8/8/8/4K3 w - - 0 1")
        move = self.search.search(board, chess.WHITE)
        
        # Should find the winning move
        self.assertIsNotNone(move)
        if move:
            board.push(move)
            self.assertTrue(
                move.uci() in ['e7e8q', 'e7e8r', 'e7e8b', 'e7e8n'],
                "Should promote pawn"
            )

    def test_move_ordering_critical(self):
        """Test move ordering in critical positions."""
        # Position with a clear tactical sequence
        board = chess.Board("r1bqkb1r/pppp1ppp/2n5/4p3/3Pn3/2N2N2/PPP1PPPP/R1BQKB1R w KQkq - 0 1")
        
        print("\nTest position:")
        print(board)
        print("\nLegal moves:", [m.uci() for m in board.legal_moves])
        print("Available captures:", [m.uci() for m in board.legal_moves if board.is_capture(m)])
        
        # Get ordered moves with tempo consideration
        scored_moves = self.ordering.order_moves_with_debug(board, tempo_bonus=0.5)
        first_move = scored_moves[0][0]
        
        # First move should be one of the strong tactical moves
        self.assertTrue(
            first_move.uci() in ['d4e5', 'c3e4'],
            "First move should capture a piece"
        )

    def test_checkmate_detection(self):
        """Test that search finds quick checkmates."""
        # Scholar's mate position
        board = chess.Board("rnb1kbnr/pppp1ppp/8/4p3/2B1P3/5Q2/PPPP1PPP/RNB1K1NR w KQkq - 0 1")
        move = self.search.search(board, chess.WHITE)
        
        # Should find Qf7#
        self.assertEqual(
            move.uci(),
            "f3f7",
            "Should find Scholar's mate"
        )

    def test_draw_prevention(self):
        """Test that search avoids repetition in equal positions."""
        board = chess.Board()
        
        # Make some moves back and forth
        moves = ["g1f3", "g8f6", "f3g1", "f6g8"]
        for uci in moves:
            board.push(chess.Move.from_uci(uci))
            
        # Search in this position
        move = self.search.search(board, board.turn)
        
        # Should not repeat the same move
        self.assertNotEqual(
            move.uci(),
            "g1f3",
            "Should avoid move repetition"
        )

    def test_tempo_advantage(self):
        """Test that search considers tempo in positional play."""
        # Position where White has development advantage
        board = chess.Board("rnbqkbnr/pppppppp/8/8/3PP3/8/PPP2PPP/RNBQKBNR b KQkq - 0 2")
        
        # Black to move - should try to equalize
        move = self.search.search(board, chess.BLACK)
        
        # Should make a central pawn move or develop a piece to a good square
        to_square = chess.square_name(move.to_square)
        self.assertTrue(
            to_square in ['e5', 'd5', 'c5', 'f5', 'e6', 'd6', 'c6', 'f6', 'c6', 'f6'],
            f"Move {move.uci()} should contest the center"
        )

if __name__ == '__main__':
    unittest.main()
