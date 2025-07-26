#!/usr/bin/env python3
"""
Simple test for Phase 1: Perspective Evaluation Fix
Tests the critical bug fix in evaluate_position_from_perspective()
"""

import sys
import os
import chess

# Add the parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

def test_perspective_fix():
    """Simple test to verify the perspective evaluation fix"""
    print("=== Testing Phase 1: Perspective Evaluation Fix ===")
    
    try:
        # Test that we can import everything
        from v7p3r_score import v7p3rScore
        from v7p3r_config import v7p3rConfig
        from v7p3r_rules import v7p3rRules
        from v7p3r_pst import v7p3rPST
        
        print("Γ£à All imports successful")
        
        # Create required components
        config = v7p3rConfig()
        rules = v7p3rRules()
        pst = v7p3rPST()
        
        print("Γ£à All components created")
        
        # Create scorer with proper constructor
        scorer = v7p3rScore(rules_manager=rules, pst=pst)
        
        print("Γ£à Scorer created successfully")
        
        # Test basic position
        board = chess.Board()
        print(f"Testing position: {board.fen()}")
        
        # This should work now without the bug
        evaluation = scorer.evaluate_position_from_perspective(board, chess.WHITE)
        stored_eval = scorer.score_dataset.get('evaluation', 'NOT_SET')
        
        print(f"White perspective evaluation: {evaluation:.3f}")
        print(f"Stored evaluation: {stored_eval}")
        
        # The fix means stored_eval should equal evaluation (not call general evaluation)
        if abs(evaluation - stored_eval) < 0.001:
            print("Γ£à SUCCESS: Stored evaluation matches perspective evaluation")
            print("Γ£à Phase 1 fix is working correctly!")
            return True
        else:
            print("Γ¥î FAILURE: Stored evaluation doesn't match perspective")
            return False
            
    except Exception as e:
        print(f"Γ¥î ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_perspective_fix()
    
    if success:
        print("\n≡ƒÄë Phase 1 fix verification PASSED!")
        print("The perspective evaluation bug has been fixed.")
    else:
        print("\nΓ¥î Phase 1 fix verification FAILED!")
        print("Further investigation needed.")
