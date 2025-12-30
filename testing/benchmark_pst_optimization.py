#!/usr/bin/env python3
"""
Benchmark PST Optimization (v18.3)

Compare old vs new PST implementation
Expected: 30-40% faster PST evaluation
"""

import chess
import sys
import os
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r_fast_evaluator import V7P3RFastEvaluator

def benchmark_pst_speed(positions, iterations=20000):
    """Benchmark PST evaluation speed"""
    
    evaluator = V7P3RFastEvaluator()
    
    print("="*80)
    print("PST OPTIMIZATION BENCHMARK (v18.3)")
    print("="*80)
    print(f"\nTesting {len(positions)} positions x {iterations:,} iterations each\n")
    
    total_time_full = 0
    total_time_pst = 0
    total_time_material = 0
    
    for name, fen in positions:
        board = chess.Board(fen)
        
        # Benchmark full evaluation
        start = time.time()
        for _ in range(iterations):
            score = evaluator.evaluate(board)
        full_time = time.time() - start
        
        # Benchmark PST only (optimized)
        start = time.time()
        for _ in range(iterations):
            score = evaluator.evaluate_pst(board)
        pst_time = time.time() - start
        
        # Benchmark material only
        start = time.time()
        for _ in range(iterations):
            score = evaluator.evaluate_material(board)
        material_time = time.time() - start
        
        total_time_full += full_time
        total_time_pst += pst_time
        total_time_material += material_time
        
        print(f"{name[:35]:35s}")
        print(f"  Full eval:     {full_time*1000/iterations:.4f}ms ({full_time:.3f}s total)")
        print(f"  PST only:      {pst_time*1000/iterations:.4f}ms ({pst_time:.3f}s total)")
        print(f"  Material only: {material_time*1000/iterations:.4f}ms ({material_time:.3f}s total)")
        print()
    
    # Calculate averages
    n = len(positions) * iterations
    avg_full = total_time_full / n * 1000
    avg_pst = total_time_pst / n * 1000
    avg_material = total_time_material / n * 1000
    
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Average time per evaluation:")
    print(f"  Full eval:     {avg_full:.4f}ms")
    print(f"  PST only:      {avg_pst:.4f}ms ({avg_pst/avg_full*100:.1f}% of full)")
    print(f"  Material only: {avg_material:.4f}ms ({avg_material/avg_full*100:.1f}% of full)")
    print()
    
    # Component breakdown
    strategic_time = avg_full - avg_pst - avg_material
    print(f"Component breakdown:")
    print(f"  Material:  {avg_material:.4f}ms ({avg_material/avg_full*100:.1f}%)")
    print(f"  PST:       {avg_pst:.4f}ms ({avg_pst/avg_full*100:.1f}%)")
    print(f"  Strategic: {strategic_time:.4f}ms ({strategic_time/avg_full*100:.1f}%)")
    print()
    
    # Compare to v17.1 baseline (from profiling)
    v171_pst_time = 0.0256  # ms (from profiling data)
    v171_full_time = 0.0460  # ms (from profiling data)
    
    pst_speedup = v171_pst_time / avg_pst
    full_speedup = v171_full_time / avg_full
    
    print("="*80)
    print("COMPARISON TO v17.1 BASELINE")
    print("="*80)
    print(f"v17.1 PST time:       {v171_pst_time:.4f}ms")
    print(f"v18.3 PST time:       {avg_pst:.4f}ms")
    print(f"PST speedup:          {pst_speedup:.2f}x ({(pst_speedup-1)*100:+.1f}%)")
    print()
    print(f"v17.1 Full eval time: {v171_full_time:.4f}ms")
    print(f"v18.3 Full eval time: {avg_full:.4f}ms")
    print(f"Full eval speedup:    {full_speedup:.2f}x ({(full_speedup-1)*100:+.1f}%)")
    print()
    
    # Project search impact
    # PST is ~55% of eval, eval is ~20% of search
    pst_contribution = 0.55 * 0.20  # 11% of total search time
    search_speedup = 1.0 / (1.0 - pst_contribution * (1.0 - 1.0/pst_speedup))
    
    print("="*80)
    print("PROJECTED SEARCH IMPACT")
    print("="*80)
    print(f"PST contribution to total search: {pst_contribution*100:.1f}%")
    print(f"Expected total search speedup:    {search_speedup:.3f}x ({(search_speedup-1)*100:+.1f}%)")
    print()
    
    # Depth gain estimation
    # Assuming branching factor of ~2.5 (from profiling)
    import math
    branching = 2.5
    depth_gain = math.log(search_speedup) / math.log(branching)
    
    print(f"Estimated depth gain: +{depth_gain:.2f} plies")
    print(f"Projected depth:      {6.0 + depth_gain:.1f} (from baseline 6.0)")
    print()
    
    if depth_gain >= 0.3:
        print(f"✓ SIGNIFICANT IMPROVEMENT: +{depth_gain:.2f} plies")
    elif depth_gain >= 0.1:
        print(f"✓ MEASURABLE IMPROVEMENT: +{depth_gain:.2f} plies")
    else:
        print(f"⚠ MINOR IMPROVEMENT: +{depth_gain:.2f} plies")
    
    print("\n" + "="*80)


def main():
    positions = [
        ("Starting position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Middlegame tactical", "r1bq1rk1/ppp2ppp/2n2n2/3p4/1b1P4/2NBP3/PPP2PPP/R1BQK2R w KQ - 0 1"),
        ("Middlegame strategic", "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1"),
        ("Endgame R+P", "6k1/5ppp/8/8/8/8/r4PPP/4R1K1 w - - 0 1"),
        ("Endgame pawn race", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
    ]
    
    benchmark_pst_speed(positions)


if __name__ == "__main__":
    main()
