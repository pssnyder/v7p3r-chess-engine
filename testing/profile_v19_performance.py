#!/usr/bin/env python3
"""
V19 Performance Profiling - Find The Bottleneck

WHY: Engine allocates 8-10s but only reaches depth 3-5 (should be 8-10)
GOAL: Identify which part of the code is slow
"""

import chess
import sys
import time
import cProfile
import pstats
from pathlib import Path
from io import StringIO

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import v7p3r


def profile_single_search():
    """Profile a single search to find bottlenecks"""
    
    print("=" * 80)
    print("V19 PERFORMANCE PROFILING")
    print("=" * 80)
    
    # Test position: Complex middlegame
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1")
    
    engine = v7p3r.V7P3REngine()
    
    print("\nTest Position: Complex Middlegame")
    print(f"FEN: {board.fen()}")
    print(f"Legal moves: {board.legal_moves.count()}")
    
    # Profile the search
    print("\nProfiling 5-second search...")
    profiler = cProfile.Profile()
    
    start = time.time()
    profiler.enable()
    
    result = engine.search(board, time_limit=5.0)
    
    profiler.disable()
    elapsed = time.time() - start
    
    print(f"\n✓ Search completed in {elapsed:.2f}s")
    print(f"  Best move: {result[1]}")
    print(f"  Score: {result[0]:.2f}cp")
    print(f"  Nodes: {engine.nodes_searched:,}")
    print(f"  NPS: {int(engine.nodes_searched / elapsed):,} nodes/sec")
    
    # Analyze profile
    print("\n" + "=" * 80)
    print("TOP 20 SLOWEST FUNCTIONS")
    print("=" * 80)
    
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('cumulative')
    stats.print_stats(20)
    
    print(stream.getvalue())
    
    # Also sort by total time
    print("\n" + "=" * 80)
    print("TOP 20 BY TOTAL TIME")
    print("=" * 80)
    
    stream = StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.sort_stats('tottime')
    stats.print_stats(20)
    
    print(stream.getvalue())


def benchmark_components():
    """Benchmark individual engine components"""
    
    print("\n\n" + "=" * 80)
    print("COMPONENT BENCHMARKS")
    print("=" * 80)
    
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1")
    engine = v7p3r.V7P3REngine()
    
    # Benchmark move generation
    print("\n1. Move Generation")
    start = time.time()
    for _ in range(1000):
        moves = list(board.legal_moves)
    elapsed = time.time() - start
    print(f"   1000 iterations: {elapsed*1000:.2f}ms ({elapsed:.4f}ms per call)")
    
    # Benchmark evaluation
    print("\n2. Position Evaluation")
    start = time.time()
    for _ in range(1000):
        score = engine._evaluate_position(board)
    elapsed = time.time() - start
    print(f"   1000 evaluations: {elapsed*1000:.2f}ms ({elapsed:.4f}ms per call)")
    
    # Benchmark quiescence search
    print("\n3. Quiescence Search (depth 4)")
    start = time.time()
    iterations = 100
    for _ in range(iterations):
        score = engine._quiescence_search(board, -10000, 10000, 4)
    elapsed = time.time() - start
    print(f"   {iterations} iterations: {elapsed*1000:.2f}ms ({elapsed/iterations*1000:.2f}ms per call)")
    
    # Benchmark move safety check
    print("\n4. Move Safety Check")
    moves = list(board.legal_moves)
    start = time.time()
    for _ in range(100):
        for move in moves:
            safety = engine.move_safety.evaluate_move_safety(board, move)
    elapsed = time.time() - start
    print(f"   100 iterations x {len(moves)} moves: {elapsed*1000:.2f}ms ({elapsed/len(moves)/100*1000:.4f}ms per move)")


def test_depth_progression():
    """Test how deep the engine can search in various time limits"""
    
    print("\n\n" + "=" * 80)
    print("DEPTH PROGRESSION TEST")
    print("=" * 80)
    
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1")
    engine = v7p3r.V7P3REngine()
    
    time_limits = [1.0, 2.0, 3.0, 5.0, 10.0]
    
    print("\nTime Limit | Depth Reached | Nodes | NPS")
    print("-" * 60)
    
    for time_limit in time_limits:
        engine.nodes_searched = 0  # Reset
        start = time.time()
        result = engine.search(board, time_limit=time_limit)
        elapsed = time.time() - start
        
        nps = int(engine.nodes_searched / elapsed) if elapsed > 0 else 0
        
        # Estimate depth (would need to instrument search to know exactly)
        print(f"{time_limit:6.1f}s    | ~depth ?      | {engine.nodes_searched:6,} | {nps:,}")


def main():
    print("Starting v19 performance profiling...")
    print("This will take ~30 seconds\n")
    
    # Component benchmarks first (fast)
    benchmark_components()
    
    # Depth progression
    test_depth_progression()
    
    # Full profile (slow but detailed)
    profile_single_search()
    
    print("\n\n" + "=" * 80)
    print("DIAGNOSIS")
    print("=" * 80)
    print("""
Based on profiling results, look for:
1. Evaluation taking >0.1ms per call (should be ~0.01ms)
2. Quiescence search taking >10ms per call
3. Move safety checks taking excessive time
4. Low NPS (<50,000 nodes/sec is concerning)

Expected performance for depth 8-10 in 5s:
- NPS: 100,000+ nodes/sec
- Evaluation: <0.05ms per call
- Total nodes: 500,000+ in 5 seconds

If actual performance is far below this, we have a bottleneck to fix.
""")


if __name__ == "__main__":
    main()
