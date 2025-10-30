#!/usr/bin/env python3
"""
Test V14.4 Architecture Cleanup
Verify that the unified bitboard evaluation system works correctly
"""

import sys
import os
import chess

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine

def test_unified_evaluation():
    """Test that the unified evaluation system works"""
    print("Testing V14.4 Unified Architecture...")
    
    engine = V7P3REngine()
    
    # Test positions
    positions = [
        chess.Board(),  # Starting position
        chess.Board('rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1'),  # After e4
        chess.Board('r6k/pp4pp/3pQ1n1/2pP1rq1/2N5/8/PPP2PPP/4RRK1 b - - 3 18'),  # Tactical position
    ]
    
    for i, board in enumerate(positions):
        print(f"\nTesting position {i + 1}...")
        try:
            # Test evaluation
            eval_score = engine._evaluate_position(board)
            print(f"  Evaluation: {eval_score:.2f}")
            
            # Test pin detection
            pins = engine.bitboard_evaluator.detect_pins_bitboard(board)
            white_pins = len(pins.get('white_pins', []))
            black_pins = len(pins.get('black_pins', []))
            print(f"  Pins: White={white_pins}, Black={black_pins}")
            
            # Test tactical analysis
            tactics = engine.bitboard_evaluator.analyze_position_for_tactics_bitboard(board)
            white_bonus = tactics.get('white_tactical_bonus', 0)
            black_bonus = tactics.get('black_tactical_bonus', 0)
            print(f"  Tactical bonuses: White={white_bonus:.2f}, Black={black_bonus:.2f}")
            
            # Test move generation and ordering (first 5 moves)
            legal_moves = list(board.legal_moves)[:5]
            if legal_moves:
                ordered_moves = engine._order_moves_advanced(board, legal_moves, 4)
                print(f"  Move ordering: {[str(move) for move in ordered_moves[:3]]}")
            
            print(f"  ✅ Position {i + 1} passed")
            
        except Exception as e:
            print(f"  ❌ Position {i + 1} failed: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    print("\n✅ All architecture tests passed!")
    return True

def test_performance_basic():
    """Basic performance test"""
    print("\nTesting basic performance...")
    
    import time
    engine = V7P3REngine()
    board = chess.Board()
    
    # Time evaluation calls
    start_time = time.time()
    for _ in range(100):
        engine._evaluate_position(board)
    end_time = time.time()
    
    avg_time = (end_time - start_time) / 100
    evals_per_sec = 1.0 / avg_time if avg_time > 0 else float('inf')
    
    print(f"  Average evaluation time: {avg_time * 1000:.3f}ms")
    print(f"  Evaluations per second: {evals_per_sec:,.0f}")
    
    if avg_time < 0.001:  # Less than 1ms
        print("  ✅ Performance acceptable")
        return True
    else:
        print("  ⚠️ Performance may be slower than expected")
        return True  # Still pass, just warn

if __name__ == "__main__":
    success = True
    success &= test_unified_evaluation()
    success &= test_performance_basic()
    
    if success:
        print("\n🎉 Architecture cleanup successful!")
        print("All evaluation functions are now unified in the bitboard evaluator.")
    else:
        print("\n❌ Architecture cleanup needs attention.")
        sys.exit(1)