#!/usr/bin/env python3
"""
Test V6.2 optimization features:
1. Opening phase TT injection (only first 4 moves)
2. Threshold-based scoring early exits
3. Performance comparison between optimized and traditional evaluation
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import chess
import time
from v7p3r import V7P3REvaluationEngine

def test_opening_injection_phase_limit():
    """Test that opening injection only works in opening phase"""
    print("=== Testing Opening Phase Injection Limits ===")
    
    engine = V7P3REvaluationEngine()
    engine.set_optimization_mode(use_optimized_scoring=True, use_opening_injection=True)
    
    # Test positions at different phases
    positions = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Opening", True),
        ("rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2", "Early Opening", True),
        ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", "Mid Opening", True),
        ("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5", "Late Opening", False),
        ("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 6 7", "Middlegame", False)
    ]
    
    for fen, phase, should_inject in positions:
        board = chess.Board(fen)
        ply = board.ply()
        print(f"{phase} (ply {ply}): {'Should inject' if should_inject else 'Should NOT inject'}")
        
        # The injection happens during initialization based on current position
        engine.board = board
        original_tt_size = len(engine.transposition_table)
        
        # Clear and re-inject to test
        engine.transposition_table.clear()
        engine._inject_opening_knowledge()
        new_tt_size = len(engine.transposition_table)
        
        injected = new_tt_size > 0
        print(f"  Actual: {'Injected' if injected else 'Not injected'} ({new_tt_size} moves)")
        print(f"  Result: {'✓' if injected == should_inject else '✗'}")
        print()

def test_scoring_optimization_performance():
    """Compare performance between optimized and traditional scoring"""
    print("=== Testing Scoring Optimization Performance ===")
    
    # Test positions with different characteristics
    test_positions = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Balanced Opening"),
        ("8/8/8/8/8/8/8/7K w - - 0 1", "Minimal Material"),
        ("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 2 2", "Tactical Position"),
        ("rnbqkb1r/pppp1Qpp/5n2/4p3/4P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4", "Mate Threat"),
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", "Material Imbalance")
    ]
    
    for fen, description in test_positions:
        print(f"\nTesting: {description}")
        print(f"FEN: {fen}")
        
        board = chess.Board(fen)
        
        # Test with optimized scoring
        engine_opt = V7P3REvaluationEngine()
        engine_opt.board = board
        engine_opt.set_optimization_mode(use_optimized_scoring=True, use_opening_injection=False)
        
        start_time = time.time()
        score_opt = engine_opt.evaluate_position(board)
        time_opt = time.time() - start_time
        
        # Test with traditional scoring
        engine_trad = V7P3REvaluationEngine()
        engine_trad.board = board
        engine_trad.set_optimization_mode(use_optimized_scoring=False, use_opening_injection=False)
        
        start_time = time.time()
        score_trad = engine_trad.evaluate_position(board)
        time_trad = time.time() - start_time
        
        # Results
        speedup = time_trad / time_opt if time_opt > 0 else float('inf')
        score_diff = abs(score_opt - score_trad)
        
        print(f"  Optimized:   {score_opt:.3f} ({time_opt*1000:.2f}ms)")
        print(f"  Traditional: {score_trad:.3f} ({time_trad*1000:.2f}ms)")
        print(f"  Speedup:     {speedup:.2f}x")
        print(f"  Score diff:  {score_diff:.3f}")
        
        # Check if optimization maintained reasonable accuracy
        accuracy_ok = score_diff < 100  # Allow some difference due to early exits
        speed_ok = speedup >= 0.8  # Should be at least as fast (allowing for measurement noise)
        
        print(f"  Accuracy:    {'✓' if accuracy_ok else '✗'}")
        print(f"  Speed:       {'✓' if speed_ok else '✗'}")

def test_fast_evaluation_thresholds():
    """Test that fast evaluation early exits work correctly"""
    print("\n=== Testing Fast Evaluation Thresholds ===")
    
    engine = V7P3REvaluationEngine()
    
    # Test positions that should trigger different exit thresholds
    threshold_tests = [
        ("7K/8/8/8/8/8/8/7k w - - 0 1", "Minimal material - should exit early"),
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Balanced - full evaluation"),
        ("8/8/8/8/8/5k2/8/4K2Q w - - 0 1", "King in check - early exit"),
        ("rnbqkb1r/pppp1Qpp/5n2/4p3/4P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4", "Mate threat - immediate exit")
    ]
    
    for fen, description in threshold_tests:
        board = chess.Board(fen)
        start_time = time.time()
        score = engine.evaluate_position(board)
        eval_time = time.time() - start_time
        
        print(f"{description}")
        print(f"  Score: {score:.3f}")
        print(f"  Time:  {eval_time*1000:.2f}ms")
        print()

if __name__ == "__main__":
    print("V7P3R V6.2 Optimization Tests")
    print("=" * 40)
    
    try:
        test_opening_injection_phase_limit()
        test_scoring_optimization_performance()
        test_fast_evaluation_thresholds()
        
        print("\n" + "=" * 40)
        print("Optimization tests completed!")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
