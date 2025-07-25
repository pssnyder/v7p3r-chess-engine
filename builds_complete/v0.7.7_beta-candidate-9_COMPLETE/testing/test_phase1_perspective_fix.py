#!/usr/bin/env python3
"""
Test for Phase 1: Perspective Evaluation Fix
Tests the critical bug fix in evaluate_position_from_perspective()
"""

import sys
import os
import chess

# Add the parent directory to Python path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from v7p3r_score import v7p3rScore
from v7p3r_config import v7p3rConfig

def test_perspective_evaluation_fix():
    """Test that perspective evaluation returns correct values"""
    print("=== Phase 1: Testing Perspective Evaluation Fix ===")
    
    # Initialize scoring calculator
    config = v7p3rConfig()
    scorer = v7p3rScore(
        config=config, 
        monitoring_enabled=True,
        verbose_output=False
    )
    
    # Test position: Starting position (should be roughly equal)
    board = chess.Board()
    
    print(f"Testing position: {board.fen()}")
    print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")
    
    # Test White's perspective (should be roughly 0 for starting position)
    white_perspective = scorer.evaluate_position_from_perspective(board, chess.WHITE)
    print(f"White's perspective evaluation: {white_perspective:.3f}")
    
    # Test general evaluation (should be roughly 0 for starting position)  
    general_evaluation = scorer.evaluate_position(board)
    print(f"General evaluation (White's perspective): {general_evaluation:.3f}")
    
    # Verify the score_dataset was updated with the perspective score
    stored_evaluation = scorer.score_dataset.get('evaluation', 'NOT_SET')
    print(f"Stored evaluation in score_dataset: {stored_evaluation}")
    
    # The bug was that stored_evaluation would equal general_evaluation 
    # instead of white_perspective. Now they should match.
    print(f"\n=== Results ===")
    print(f"White perspective score: {white_perspective:.3f}")
    print(f"Stored score (should match): {stored_evaluation:.3f}")
    print(f"General evaluation: {general_evaluation:.3f}")
    
    if abs(white_perspective - stored_evaluation) < 0.001:
        print("Γ£à SUCCESS: Stored evaluation matches perspective evaluation")
        return True
    else:
        print("Γ¥î FAILURE: Stored evaluation does not match perspective evaluation")
        return False

def test_position_with_advantage():
    """Test a position where one side has a clear advantage"""
    print("\n=== Testing Position with Material Advantage ===")
    
    config = v7p3rConfig()
    scorer = v7p3rScoringCalculator(
        config=config, 
        monitoring_enabled=True,
        verbose_output=False
    )
    
    # Position where White has extra material (missing Black queen)
    # This should strongly favor White
    board = chess.Board("rnb1kbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    
    print(f"Testing position (White up a queen): {board.fen()}")
    
    # White's perspective should be very positive
    white_perspective = scorer.evaluate_position_from_perspective(board, chess.WHITE)
    stored_evaluation = scorer.score_dataset.get('evaluation', 0)
    
    print(f"White's perspective evaluation: {white_perspective:.3f}")
    print(f"Stored evaluation: {stored_evaluation:.3f}")
    
    if white_perspective > 500 and abs(white_perspective - stored_evaluation) < 0.001:
        print("Γ£à SUCCESS: White shows large positive advantage and storage is correct")
        return True
    else:
        print("Γ¥î FAILURE: Either advantage not detected or storage mismatch")
        return False

if __name__ == "__main__":
    print("Testing Phase 1 Fix: Perspective Evaluation Bug")
    print("=" * 50)
    
    try:
        test1_result = test_perspective_evaluation_fix()
        test2_result = test_position_with_advantage()
        
        print(f"\n=== Final Results ===")
        print(f"Basic perspective test: {'PASS' if test1_result else 'FAIL'}")
        print(f"Advantage position test: {'PASS' if test2_result else 'FAIL'}")
        
        if test1_result and test2_result:
            print("Γ£à Phase 1 fix appears to be working correctly!")
        else:
            print("Γ¥î Phase 1 fix needs further investigation")
            
    except Exception as e:
        print(f"Γ¥î ERROR during testing: {e}")
        import traceback
        traceback.print_exc()
