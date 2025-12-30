#!/usr/bin/env python3
"""
Test actual speedup from skipping strategic evaluation

Measures:
1. Time per evaluation call (full vs. fast path)
2. Speedup ratio
3. Whether it translates to depth improvement
"""

import chess
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r_fast_evaluator import V7P3RFastEvaluator

def benchmark_full_evaluation(evaluator, board, iterations=10000):
    """Benchmark full evaluation (material + PST + strategic)"""
    start = time.time()
    for _ in range(iterations):
        score = evaluator.evaluate(board)
    elapsed = time.time() - start
    return elapsed, score

def benchmark_fast_path(evaluator, board, iterations=10000):
    """Benchmark fast path (material + PST only, skip strategic)"""
    start = time.time()
    for _ in range(iterations):
        material = evaluator.evaluate_material(board)
        pst = evaluator.evaluate_pst(board)
        score = int(pst * 0.6 + material * 0.4)
    elapsed = time.time() - start
    return elapsed, score

def main():
    print("="*80)
    print("EVALUATION SPEED BENCHMARK")
    print("="*80)
    
    evaluator = V7P3RFastEvaluator()
    
    # Test position: middlegame with strategic elements
    board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1")
    
    iterations = 10000
    
    # Benchmark full evaluation
    print(f"\n1. Full Evaluation (Material + PST + Strategic)")
    print(f"   Running {iterations:,} evaluations...")
    full_time, full_score = benchmark_full_evaluation(evaluator, board, iterations)
    time_per_eval_full = (full_time / iterations) * 1000  # ms
    print(f"   Total time: {full_time:.3f}s")
    print(f"   Time per eval: {time_per_eval_full:.4f}ms")
    print(f"   Score: {full_score}cp")
    
    # Benchmark fast path
    print(f"\n2. Fast Path (Material + PST only)")
    print(f"   Running {iterations:,} evaluations...")
    fast_time, fast_score = benchmark_fast_path(evaluator, board, iterations)
    time_per_eval_fast = (fast_time / iterations) * 1000  # ms
    print(f"   Total time: {fast_time:.3f}s")
    print(f"   Time per eval: {time_per_eval_fast:.4f}ms")
    print(f"   Score: {fast_score}cp")
    
    # Calculate speedup
    speedup = full_time / fast_time if fast_time > 0 else 0
    time_saved = time_per_eval_full - time_per_eval_fast
    percent_saved = (time_saved / time_per_eval_full * 100) if time_per_eval_full > 0 else 0
    
    print(f"\n{'='*80}")
    print("SPEEDUP ANALYSIS")
    print(f"{'='*80}")
    print(f"Fast path speedup: {speedup:.2f}x")
    print(f"Time saved per eval: {time_saved:.4f}ms ({percent_saved:.1f}%)")
    print(f"Score difference: {abs(full_score - fast_score)}cp")
    
    # Estimate depth impact
    print(f"\n{'='*80}")
    print("DEPTH IMPACT ESTIMATE")
    print(f"{'='*80}")
    print(f"Assumptions:")
    print(f"  - Baseline depth 6.0 at 10s time limit")
    print(f"  - Average nodes at depth 6: ~20,000")
    print(f"  - Branching factor: ~35")
    print(f"")
    
    # Simplified depth calculation
    # If eval is X% faster and eval is Y% of total time, then total speedup is:
    # speedup_total = 1 / (1 - Y * (1 - 1/X))
    
    # Assume evaluation is ~20% of total search time (rest is move gen, ordering, etc.)
    eval_fraction = 0.20
    total_speedup = 1 / (1 - eval_fraction * (1 - 1/speedup))
    
    # Depth scales with log of speedup (roughly)
    # If we're 1.5x faster, we can search ~0.5-0.7 more plies
    import math
    depth_gain = math.log(total_speedup) / math.log(35) * 1.2  # Adjusted for chess branching
    new_depth = 6.0 + depth_gain
    
    print(f"Total search speedup (estimated): {total_speedup:.2f}x")
    print(f"Expected depth gain: +{depth_gain:.1f} plies")
    print(f"Projected depth: {new_depth:.1f} (from baseline 6.0)")
    print(f"")
    
    if new_depth >= 8.0:
        print(f"✓ ACHIEVES TARGET: Depth {new_depth:.1f} >= 8.0")
    else:
        print(f"✗ FALLS SHORT: Depth {new_depth:.1f} < 8.0 target")
        print(f"  Need {8.0 - new_depth:.1f} more plies")
        print(f"  Would require {pow(35, 8.0 - new_depth):.1f}x additional speedup")
    
    print(f"\n{'='*80}")

if __name__ == "__main__":
    main()
