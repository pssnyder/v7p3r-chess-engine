#!/usr/bin/env python3
"""
Performance Comparison: V7.0 vs V9.3 Scoring
Direct NPS and evaluation speed comparison
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
import time
from v7p3r_scoring_calculation import V7P3RScoringCalculationClean
from v7p3r_scoring_calculation_v93 import V7P3RScoringCalculationV93

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0
}

def benchmark_evaluation_speed(scorer, name, positions, iterations=1000):
    """Benchmark evaluation speed for a scorer"""
    print(f"\nBenchmarking {name}:")
    print(f"Positions: {len(positions)}, Iterations per position: {iterations}")
    
    total_evaluations = len(positions) * iterations
    start_time = time.time()
    
    for _ in range(iterations):
        for board in positions:
            white_score = scorer.calculate_score_optimized(board, chess.WHITE)
            black_score = scorer.calculate_score_optimized(board, chess.BLACK)
    
    end_time = time.time()
    elapsed = end_time - start_time
    
    evaluations_per_second = total_evaluations / elapsed
    
    print(f"  Total evaluations: {total_evaluations}")
    print(f"  Time taken: {elapsed:.3f} seconds")
    print(f"  Evaluations per second: {evaluations_per_second:.0f}")
    print(f"  Time per evaluation: {(elapsed / total_evaluations) * 1000:.3f} ms")
    
    return evaluations_per_second

def create_test_positions():
    """Create variety of test positions"""
    fens = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",  # Starting
        "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq e6 0 2",  # King's pawn
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4",  # Development
        "r1bq1rk1/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQ1RK1 w - - 0 6",  # Castled
        "8/1P6/8/8/8/8/k7/K7 w - - 0 1",  # Endgame
        "r3k2r/8/8/8/8/8/8/4Q2K w kq - 0 1",  # Material imbalance
        "rnbqkb1r/ppp1pppp/5n2/3p4/3P4/5N2/PPP1PPPP/RNBQKB1R w KQkq - 2 3",  # Center control
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 5",  # Pin setup
    ]
    
    return [chess.Board(fen) for fen in fens]

def compare_evaluation_results(positions):
    """Compare evaluation results between v7.0 and v9.3"""
    print("\nEvaluation Result Comparison:")
    print("=" * 80)
    
    v70_scorer = V7P3RScoringCalculationClean(PIECE_VALUES)
    v93_scorer = V7P3RScoringCalculationV93(PIECE_VALUES)
    
    for i, board in enumerate(positions):
        print(f"\nPosition {i+1}: {board.fen()}")
        
        # V7.0 evaluation
        v70_white = v70_scorer.calculate_score_optimized(board, chess.WHITE)
        v70_black = v70_scorer.calculate_score_optimized(board, chess.BLACK)
        v70_eval = v70_white - v70_black
        
        # V9.3 evaluation
        v93_white = v93_scorer.calculate_score_optimized(board, chess.WHITE)
        v93_black = v93_scorer.calculate_score_optimized(board, chess.BLACK)
        v93_eval = v93_white - v93_black
        
        print(f"  V7.0 - White: {v70_white:.2f}, Black: {v70_black:.2f}, Eval: {v70_eval:.2f}")
        print(f"  V9.3 - White: {v93_white:.2f}, Black: {v93_black:.2f}, Eval: {v93_eval:.2f}")
        print(f"  Difference: {v93_eval - v70_eval:.2f}")

def main():
    """Main performance comparison"""
    print("V7P3R Scoring Calculation Performance Comparison")
    print("=" * 60)
    
    # Create test positions
    positions = create_test_positions()
    print(f"Created {len(positions)} test positions")
    
    # Initialize scorers
    v70_scorer = V7P3RScoringCalculationClean(PIECE_VALUES)
    v93_scorer = V7P3RScoringCalculationV93(PIECE_VALUES)
    
    # Benchmark V7.0 scoring
    v70_speed = benchmark_evaluation_speed(v70_scorer, "V7.0 Scoring (Simple)", positions, 500)
    
    # Benchmark V9.3 scoring  
    v93_speed = benchmark_evaluation_speed(v93_scorer, "V9.3 Scoring (Complex)", positions, 500)
    
    # Performance comparison
    print(f"\n{'='*60}")
    print("PERFORMANCE ANALYSIS")
    print(f"{'='*60}")
    print(f"V7.0 Speed: {v70_speed:.0f} evaluations/second")
    print(f"V9.3 Speed: {v93_speed:.0f} evaluations/second")
    
    if v70_speed > v93_speed:
        speedup = v70_speed / v93_speed
        print(f"V7.0 is {speedup:.1f}x FASTER than V9.3")
        print(f"V9.3 is {(1/speedup)*100:.1f}% the speed of V7.0")
    else:
        speedup = v93_speed / v70_speed
        print(f"V9.3 is {speedup:.1f}x faster than V7.0")
    
    print(f"\nEstimated NPS Impact:")
    print(f"If search does 10,000 evaluations per second:")
    print(f"  V7.0 estimated NPS: ~{(v70_speed/8):.0f}")  # Rough estimate 
    print(f"  V9.3 estimated NPS: ~{(v93_speed/8):.0f}")
    
    # Show evaluation differences
    compare_evaluation_results(positions[:3])  # Just first 3 positions
    
    print(f"\n🎯 CONCLUSION:")
    if v70_speed > v93_speed * 2:
        print("✓ V7.0 scoring is SIGNIFICANTLY faster - this explains the NPS problem!")
        print("✓ V9.3's complex evaluation is killing performance")
        print("→ Recommendation: Use V7.0 as base, add minimal improvements")
    elif v70_speed > v93_speed:
        print("✓ V7.0 scoring is faster but not dramatically")
        print("→ V9.3 complexity may be worth it for stronger play")
    else:
        print("? Unexpected: V9.3 is not slower than V7.0")
        print("→ Performance issues may be elsewhere")

if __name__ == "__main__":
    main()
