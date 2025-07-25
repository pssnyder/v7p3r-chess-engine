import unittest
import chess
from v7p3r_config import v7p3rConfig
from v7p3r_pst import v7p3rPST
from v7p3r_rules import v7p3rRules
from v7p3r_tempo import v7p3rTempo

class TestTempo(unittest.TestCase):
    def setUp(self):
        self.config = v7p3rConfig()
        self.pst = v7p3rPST()
        self.rules = v7p3rRules(pst=self.pst)
        self.tempo = v7p3rTempo(config=self.config, pst=self.pst, rules=self.rules)

    def test_game_phase_detection(self):
        """Test accurate game phase detection."""
        # Starting position
        board = chess.Board()
        phase, factor = self.tempo.calculate_game_phase(board)
        self.assertEqual(phase, 'opening')
        self.assertEqual(factor, 0.0)

        # Middle game position (some exchanges, one side castled)
        board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
        phase, factor = self.tempo.calculate_game_phase(board)
        self.assertEqual(phase, 'opening')
        self.assertGreater(factor, 0.0)

        # Endgame position
        board = chess.Board("4k3/8/8/8/8/8/4P3/4K3 w - - 0 1")
        phase, factor = self.tempo.calculate_game_phase(board)
        self.assertEqual(phase, 'endgame')
        self.assertEqual(factor, 1.0)

    def test_tempo_assessment(self):
        """Test tempo evaluation."""
        # Starting position
        board = chess.Board()
        white_tempo = self.tempo.assess_tempo(board, chess.WHITE)
        black_tempo = self.tempo.assess_tempo(board, chess.BLACK)
        self.assertGreater(white_tempo, 0)  # White should have positive tempo
        self.assertLessEqual(black_tempo, 0)  # Black should have non-positive tempo

        # After 1.e4
        board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1")
        # Evaluate from both perspectives
        white_tempo = self.tempo.assess_tempo(board, chess.WHITE)
        black_tempo = self.tempo.assess_tempo(board, chess.BLACK)
        self.assertGreater(white_tempo, black_tempo)  # White has development and center control advantage
        
        # After 1.e4 e5 - equal development
        board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2")
        white_tempo = self.tempo.assess_tempo(board, chess.WHITE)
        black_tempo = self.tempo.assess_tempo(board, chess.BLACK)
        self.assertAlmostEqual(white_tempo, -black_tempo, places=2)  # Should be roughly equal but opposite

    def test_zugzwang_detection(self):
        """Test zugzwang position detection."""
        # Classic zugzwang position
        board = chess.Board("8/8/p7/1p6/1P6/P7/8/k1K5 b - - 0 1")
        
        # Update position history
        self.tempo.update_position_score(board, 0.0)
        
        # Black to move is in zugzwang
        score = self.tempo.assess_zugzwang(board, chess.BLACK)
        self.assertLess(score, 0)  # Should detect disadvantageous position

    def test_checkmate_threat_detection(self):
        """Test checkmate threat evaluation."""
        # Position with mate in 2
        board = chess.Board("r2qkb1r/pp2nppp/3p4/2pNN1B1/2BnP3/3P4/PPP2PPP/R2bK2R w KQkq - 1 1")
        score = self.tempo.assess_checkmate_threats(board, chess.WHITE)
        self.assertGreater(score, 0)  # White has mating attack

        # Position with mate threat against
        board = chess.Board("r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 1")
        score = self.tempo.assess_checkmate_threats(board, chess.BLACK)
        self.assertLess(score, 0)  # Black is in danger of mate

    def test_stalemate_threat_detection(self):
        """Test stalemate threat evaluation."""
        # Position where next move could cause stalemate
        board = chess.Board("7k/8/5K2/8/8/8/8/3Q4 w - - 0 1")
        
        # Test winning side perspective
        score = self.tempo.assess_stalemate_threats(board, chess.WHITE)
        self.assertLess(score, 0)  # Should penalize potential stalemate

        # Test losing side perspective
        score = self.tempo.assess_stalemate_threats(board, chess.BLACK)
        self.assertGreater(score, -1000000.0)  # Should not penalize as heavily

    def test_drawish_position_detection(self):
        """Test detection and evaluation of drawish positions."""
        # Insufficient material
        board = chess.Board("8/8/8/3k4/8/3K4/8/8 w - - 0 1")
        score = self.tempo.assess_drawish_positions(board)
        self.assertLess(score, 0)  # Should detect drawn position

        # Test position repetition tracking
        board = chess.Board()
        self.tempo.assess_drawish_positions(board)  # First occurrence
        self.tempo.assess_drawish_positions(board)  # Second occurrence
        score = self.tempo.assess_drawish_positions(board)  # Third occurrence
        self.assertLess(score, 0)  # Should penalize repetition

    def test_position_score_tracking(self):
        """Test position score history tracking."""
        board = chess.Board()
        test_score = 0.5
        
        # Update and verify position score
        self.tempo.update_position_score(board, test_score)
        fen_key = board.fen().split(' ')[0]
        self.assertEqual(self.tempo.position_scores[fen_key], test_score)

        # Update with new score and verify change
        new_score = 1.0
        self.tempo.update_position_score(board, new_score)
        self.assertEqual(self.tempo.position_scores[fen_key], new_score)

if __name__ == '__main__':
    unittest.main()
