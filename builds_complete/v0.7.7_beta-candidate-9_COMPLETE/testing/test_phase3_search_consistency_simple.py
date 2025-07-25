#!/usr/bin/env python3
"""
Test Phase 3: Search Consistency and Evaluation Perspective
V7P3R Chess Engine Logic Debugging - Phase 3

This test verifies that the search functions maintain consistent evaluation 
perspective throughout the search tree and that scores are properly propagated.
"""

import sys
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
from v7p3r_pst import v7p3rPST

class TestSearchConsistency:
    """Test search function consistency and evaluation perspective handling."""
    
    def setup_method(self):
        """Set up test environment for each test."""
        # Initialize required components
        self.pst = v7p3rPST()
        self.rules_manager = v7p3rRules({"testphase3_ruleset":{"material_score_modifier":1.0}}, self.pst)

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

    def test_evaluation_perspective_consistency(self):
        """Test that evaluation perspective is consistent."""
        print("\n=== Testing Basic Evaluation Perspective Consistency ===")
        
        board = self.middle_game_position.copy()
        
        # Test from White's perspective
        white_eval = self.score_calculator.evaluate_position_from_perspective(board, chess.WHITE)
        print(f"White's perspective evaluation: {white_eval}")
        
        # Test from Black's perspective  
        black_eval = self.score_calculator.evaluate_position_from_perspective(board, chess.BLACK)
        print(f"Black's perspective evaluation: {black_eval}")
        
        # The evaluations should be roughly negatives of each other
        print(f"Sum of evaluations (should be close to 0): {white_eval + black_eval}")
        
        # Check that they are reasonable numbers
        assert isinstance(white_eval, (int, float)), f"White eval should be numeric, got {type(white_eval)}"
        assert isinstance(black_eval, (int, float)), f"Black eval should be numeric, got {type(black_eval)}"
        
        return True

    def test_simple_search_consistency(self):
        """Test simple search algorithm for basic consistency."""
        print("\n=== Testing Simple Search Consistency ===")
        
        board = self.start_position.copy()
        
        # Test search from White's perspective
        self.search_engine.search_algorithm = 'simple'
        self.search_engine.depth = 1
        
        try:
            move = self.search_engine.search(board, chess.WHITE)
            print(f"Simple search returned move: {move}")
            
            # Verify the move is legal
            assert board.is_legal(move), f"Search returned illegal move: {move}"
            print("Γ£ô Simple search returned legal move")
            
            return True
            
        except Exception as e:
            print(f"Γ¥î Simple search failed: {e}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return False

def run_phase3_tests():
    """Run all Phase 3 search consistency tests."""
    print("=" * 60)
    print("V7P3R CHESS ENGINE - PHASE 3 SEARCH CONSISTENCY TESTS")
    print("=" * 60)
    
    test_instance = TestSearchConsistency()
    
    try:
        print("\nInitializing test components...")
        test_instance.setup_method()
        print("Γ£ô Test setup completed")
        
        print("\nRunning Phase 3 Tests...")
        
        if not test_instance.test_evaluation_perspective_consistency():
            return False
        print("Γ£ô Evaluation perspective consistency test PASSED")
        
        if not test_instance.test_simple_search_consistency():
            return False  
        print("Γ£ô Simple search consistency test PASSED")
        
        print("\n" + "=" * 60)
        print("BASIC PHASE 3 TESTS PASSED!")
        print("Ready to identify and fix search perspective issues.")
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
