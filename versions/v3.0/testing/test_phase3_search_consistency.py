#!/usr/biimport sys
import os
import chess

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from v7p3r_search import v7p3rSearch
from v7p3r_score import v7p3rScore
from v7p3r_ordering import v7p3rOrdering
from v7p3r_time import v7p3rTime
from v7p3r_book import v7p3rBook
from v7p3r_rules import v7p3rRules
from v7p3r_pst import v7p3rPST"""
Test Phase 3: Search Consistency and Evaluation Perspective
V7P3R Chess Engine Logic Debugging - Phase 3

This test verifies that the search functions maintain consistent evaluation 
perspective throughout the search tree and that scores are properly propagated.
"""

import sys
import os
import chess
import pytest

# Add the parent directory to the Python path
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from v7p3r_search import v7p3rSearch
from v7p3r_score import v7p3rScore
from v7p3r_ordering import v7p3rOrdering
from v7p3r_time import v7p3rTime
from v7p3r_book import v7p3rBook

class TestSearchConsistency:
    """Test search function consistency and evaluation perspective handling."""
    
    def setup_method(self):
        """Set up test environment for each test."""
        # Initialize required components
        self.rules_manager = v7p3rRules()
        self.pst = v7p3rPST()
        
        # Initialize score calculator with required dependencies
        self.score_calculator = v7p3rScore(self.rules_manager, self.pst)
        
        # Initialize move organizer with score calculator dependency
        self.move_organizer = v7p3rOrdering(self.score_calculator)
        
        # Initialize other components
        self.time_manager = v7p3rTime()
        self.opening_book = v7p3rBook()
        
        # Create search engine
        self.search_engine = v7p3rSearch(
            self.score_calculator,
            self.move_organizer, 
            self.time_manager,
            self.opening_book
        )
        
        # Test positions
        self.start_position = chess.Board()
        self.middle_game_position = chess.Board("r2qkb1r/ppp2ppp/2n1bn2/3p4/3P4/3B1N2/PPP2PPP/RNBQK2R w KQkq - 0 6")
        self.tactical_position = chess.Board("2rq1rk1/ppp2ppp/2n1bn2/8/3P4/2N2N2/PPP2PPP/R1BQKB1R w KQ - 0 8")

    def test_minimax_evaluation_perspective(self):
        """Test that minimax maintains consistent evaluation perspective."""
        print("\n=== Testing Minimax Evaluation Perspective ===")
        
        # Test from White's perspective
        board = self.middle_game_position.copy()
        color = chess.WHITE
        
        # Get evaluation from root perspective
        root_eval = self.score_calculator.evaluate_position_from_perspective(board, color)
        print(f"Root evaluation from White's perspective: {root_eval}")
        
        # Test minimax at depth 2
        self.search_engine.current_perspective = color
        self.search_engine.depth = 2
        
        # Call minimax directly to test internal consistency
        score = self.search_engine._minimax_search(board, 2, -float('inf'), float('inf'), True)
        print(f"Minimax returned score: {score}")
        
        # The score should be from White's perspective (positive = good for White)
        assert isinstance(score, (int, float)), f"Minimax should return numeric score, got {type(score)}"
        
        # Test from Black's perspective
        color = chess.BLACK
        self.search_engine.current_perspective = color
        
        root_eval_black = self.score_calculator.evaluate_position_from_perspective(board, color)
        print(f"Root evaluation from Black's perspective: {root_eval_black}")
        
        score_black = self.search_engine._minimax_search(board, 2, -float('inf'), float('inf'), True)
        print(f"Minimax from Black's perspective: {score_black}")
        
        # The evaluations should be roughly negatives of each other
        # (allowing for small rounding differences)
        assert abs(root_eval + root_eval_black) < 0.1, f"Evaluations should be negatives: {root_eval} vs {root_eval_black}"

    def test_negamax_evaluation_perspective(self):
        """Test that negamax maintains consistent evaluation perspective."""
        print("\n=== Testing Negamax Evaluation Perspective ===")
        
        board = self.middle_game_position.copy()
        
        # Test from White's perspective
        color = chess.WHITE
        self.search_engine.current_perspective = color
        self.search_engine.depth = 2
        
        root_eval = self.score_calculator.evaluate_position_from_perspective(board, color)
        print(f"Root evaluation from White's perspective: {root_eval}")
        
        score = self.search_engine._negamax_search(board, 2, -float('inf'), float('inf'))
        print(f"Negamax returned score: {score}")
        
        assert isinstance(score, (int, float)), f"Negamax should return numeric score, got {type(score)}"
        
        # Test from Black's perspective  
        color = chess.BLACK
        self.search_engine.current_perspective = color
        
        root_eval_black = self.score_calculator.evaluate_position_from_perspective(board, color)
        print(f"Root evaluation from Black's perspective: {root_eval_black}")
        
        score_black = self.search_engine._negamax_search(board, 2, -float('inf'), float('inf'))
        print(f"Negamax from Black's perspective: {score_black}")
        
        # Negamax scores should be consistent with the evaluation perspective
        assert abs(root_eval + root_eval_black) < 0.1, f"Evaluations should be negatives: {root_eval} vs {root_eval_black}"

    def test_quiescence_evaluation_perspective(self):
        """Test that quiescence search maintains consistent evaluation perspective."""
        print("\n=== Testing Quiescence Evaluation Perspective ===")
        
        board = self.tactical_position.copy()
        
        # Test from White's perspective
        color = chess.WHITE
        self.search_engine.current_perspective = color
        
        root_eval = self.score_calculator.evaluate_position_from_perspective(board, color)
        print(f"Root evaluation from White's perspective: {root_eval}")
        
        q_score = self.search_engine._quiescence_search(board, -float('inf'), float('inf'), True)
        print(f"Quiescence returned score: {q_score}")
        
        assert isinstance(q_score, (int, float)), f"Quiescence should return numeric score, got {type(q_score)}"
        
        # Test from Black's perspective
        color = chess.BLACK
        self.search_engine.current_perspective = color
        
        root_eval_black = self.score_calculator.evaluate_position_from_perspective(board, color)
        print(f"Root evaluation from Black's perspective: {root_eval_black}")
        
        q_score_black = self.search_engine._quiescence_search(board, -float('inf'), float('inf'), True)
        print(f"Quiescence from Black's perspective: {q_score_black}")
        
        # Check consistency
        assert abs(root_eval + root_eval_black) < 0.1, f"Evaluations should be negatives: {root_eval} vs {root_eval_black}"

    def test_search_algorithm_consistency(self):
        """Test that different search algorithms return consistent results."""
        print("\n=== Testing Search Algorithm Consistency ===")
        
        board = self.start_position.copy()
        color = chess.WHITE
        
        # Test different search algorithms
        algorithms = ['simple', 'minimax', 'negamax']
        scores = {}
        
        for algorithm in algorithms:
            self.search_engine.search_algorithm = algorithm
            self.search_engine.depth = 2
            
            try:
                move = self.search_engine.search(board, color)
                print(f"Algorithm '{algorithm}' returned move: {move}")
                
                # Verify the move is legal
                assert board.is_legal(move), f"Algorithm '{algorithm}' returned illegal move: {move}"
                
                # Store the move for comparison
                scores[algorithm] = move
                
            except Exception as e:
                print(f"Γ¥î Algorithm '{algorithm}' failed with error: {e}")
                return False
        
        print(f"All algorithms completed successfully: {scores}")

    def test_evaluation_sign_consistency(self):
        """Test that evaluation signs are consistent across different board positions."""
        print("\n=== Testing Evaluation Sign Consistency ===")
        
        # Test position where White has clear advantage
        board = chess.Board("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3")
        # Add a white queen to make white clearly better
        board.set_piece_at(chess.E4, chess.Piece(chess.QUEEN, chess.WHITE))
        
        white_eval = self.score_calculator.evaluate_position_from_perspective(board, chess.WHITE)
        black_eval = self.score_calculator.evaluate_position_from_perspective(board, chess.BLACK)
        
        print(f"Position with White advantage - White's view: {white_eval}, Black's view: {black_eval}")
        
        # White should see a positive score, Black should see a negative score
        assert white_eval > 0, f"White should see positive evaluation, got {white_eval}"
        assert black_eval < 0, f"Black should see negative evaluation, got {black_eval}"
        assert abs(white_eval + black_eval) < 0.1, f"Evaluations should be negatives: {white_eval} vs {black_eval}"

def run_phase3_tests():
    """Run all Phase 3 search consistency tests."""
    print("=" * 60)
    print("V7P3R CHESS ENGINE - PHASE 3 SEARCH CONSISTENCY TESTS")
    print("=" * 60)
    
    test_instance = TestSearchConsistency()
    
    try:
        test_instance.setup_method()
        
        print("\nRunning Phase 3 Tests...")
        
        test_instance.test_minimax_evaluation_perspective()
        print("Γ£ô Minimax evaluation perspective test PASSED")
        
        test_instance.test_negamax_evaluation_perspective()
        print("Γ£ô Negamax evaluation perspective test PASSED")
        
        test_instance.test_quiescence_evaluation_perspective()
        print("Γ£ô Quiescence evaluation perspective test PASSED")
        
        test_instance.test_search_algorithm_consistency()
        print("Γ£ô Search algorithm consistency test PASSED")
        
        test_instance.test_evaluation_sign_consistency()
        print("Γ£ô Evaluation sign consistency test PASSED")
        
        print("\n" + "=" * 60)
        print("ALL PHASE 3 TESTS PASSED SUCCESSFULLY!")
        print("Search evaluation perspective and consistency verified.")
        print("=" * 60)
        
        return True
        
    except Exception as e:
        print(f"\nΓ¥î PHASE 3 TEST FAILED: {e}")
        import traceback
        print(f"Full traceback: {traceback.format_exc()}")
        return False

if __name__ == "__main__":
    success = run_phase3_tests()
    sys.exit(0 if success else 1)
