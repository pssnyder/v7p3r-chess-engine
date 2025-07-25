import unittest
import chess
import os
import sys

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from v7p3r_mvv_lva import v7p3rMVVLVA

class TestMVVLVA(unittest.TestCase):
    def setUp(self):
        self.mvv_lva = v7p3rMVVLVA()
        self.board = chess.Board()

    def test_basic_piece_values(self):
        """Test that piece values are correctly initialized and accessed."""
        expected_values = {
            chess.PAWN: 100,
            chess.KNIGHT: 320,
            chess.BISHOP: 330,
            chess.ROOK: 500,
            chess.QUEEN: 900,
            chess.KING: 20000
        }
        for piece, value in expected_values.items():
            self.assertEqual(self.mvv_lva.piece_values[piece], value)

    def test_mvv_lva_scoring_matrix(self):
        """Test that the MVV-LVA scoring matrix produces expected relative scores."""
        # Queen takes pawn should be worth less than pawn takes queen
        queen_takes_pawn = self.mvv_lva.get_mvv_lva_matrix_score(chess.QUEEN, chess.PAWN)
        pawn_takes_queen = self.mvv_lva.get_mvv_lva_matrix_score(chess.PAWN, chess.QUEEN)
        self.assertLess(queen_takes_pawn, pawn_takes_queen)

    def test_tactical_pattern_fork(self):
        """Test detection of fork potential."""
        # Set up a knight fork position
        board = chess.Board("4k3/8/3n4/8/8/8/2P1P3/4K3 b - - 0 1")
        move = chess.Move.from_uci("d6c4")  # Knight move that forks king and pawn
        score = self.mvv_lva.evaluate_tactical_pattern(board, move)
        self.assertGreater(score, 0, "Fork potential should increase tactical score")

    def test_tactical_pattern_pin(self):
        """Test detection of pin creation."""
        # Set up a bishop pin position
        board = chess.Board("4k3/8/8/8/3b4/8/4N3/4K3 b - - 0 1")
        move = chess.Move.from_uci("d4f2")  # Bishop move that pins knight
        score = self.mvv_lva.evaluate_tactical_pattern(board, move)
        self.assertGreater(score, 0, "Pin creation should increase tactical score")

    def test_tactical_pattern_discovery(self):
        """Test detection of discovered attack potential."""
        # Set up a discovered attack position
        board = chess.Board("4k3/8/8/3P4/8/2B5/8/4K3 w - - 0 1")
        move = chess.Move.from_uci("d5d6")  # Pawn move that discovers bishop
        score = self.mvv_lva.evaluate_tactical_pattern(board, move)
        self.assertGreater(score, 0, "Discovery potential should increase tactical score")

    def test_mvv_lva_move_sorting(self):
        """Test that moves are sorted correctly by MVV-LVA score."""
        # Set up a position with multiple captures
        # Place the knight where it can capture the pawn on d5
        board = chess.Board("4k3/8/8/3p4/8/2N5/8/4K3 w - - 0 1")
        
        # Print all legal moves for debugging
        print("Legal moves:", [move.uci() for move in board.legal_moves])
        
        moves = list(board.legal_moves)
        # Create the specific move we want to test
        capture_move = chess.Move(chess.C3, chess.D5)
        
        # Verify the move is legal
        self.assertTrue(capture_move in moves, "Knight capture to d5 should be a legal move")
        
        # Sort moves
        sorted_moves = self.mvv_lva.sort_moves_by_mvv_lva(moves, board)
        
        # Verify the capture is first
        self.assertEqual(sorted_moves[0], capture_move, "Knight capture should be the first move")

    def test_safety_evaluation(self):
        """Test the safety evaluation in MVV-LVA scoring."""
        # Set up an unsafe capture position
        board = chess.Board("4k3/8/8/3p4/2P5/8/8/4K3 w - - 0 1")
        move = chess.Move.from_uci("c4d5")  # Pawn takes defended pawn
        score = self.mvv_lva.calculate_mvv_lva_score(move, board)
        
        # Now test a safe capture
        board = chess.Board("4k3/8/8/3p4/2N5/8/8/4K3 w - - 0 1")
        safe_move = chess.Move.from_uci("c4d5")  # Knight takes undefended pawn
        safe_score = self.mvv_lva.calculate_mvv_lva_score(safe_move, board)
        
        self.assertGreater(safe_score, score, "Safe capture should score higher than unsafe capture")

if __name__ == '__main__':
    unittest.main()
