#!/usr/bin/env python3
"""Quick test of Phase 1 fix"""

import chess
import sys
import os

# Add parent directory for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from v7p3r import v7p3rEngine

print("=== Testing Phase 1: Perspective Evaluation Fix ===")

try:
    # Create engine
    engine = v7p3rEngine()
    print("✅ Engine created successfully")

    # Test basic position
    board = chess.Board()
    print(f"Testing position: {board.fen()}")

    # Test evaluation from perspective  
    evaluation = engine.scoring_calculator.evaluate_position_from_perspective(board, chess.WHITE)
    stored_eval = engine.scoring_calculator.score_dataset.get('evaluation', 'NOT_SET')

    print(f"White perspective evaluation: {evaluation:.3f}")
    print(f"Stored evaluation: {stored_eval}")

    # The fix means these should match now
    if abs(evaluation - stored_eval) < 0.001:
        print("✅ SUCCESS: Phase 1 fix is working!")
        print("✅ Stored evaluation matches perspective evaluation")
        success = True
    else:
        print("❌ FAILURE: Stored evaluation does not match perspective")
        print(f"Difference: {abs(evaluation - stored_eval):.6f}")
        success = False
        
    print(f"\nResult: {'PASS' if success else 'FAIL'}")
    
except Exception as e:
    print(f"❌ ERROR: {e}")
    import traceback
    traceback.print_exc()
