#!/usr/bin/env python3
"""
V17.1 Performance Profiler

Comprehensive profiling to identify:
1. Actual bottlenecks in search
2. Expensive vs. cheap evaluation components
3. Redundant calculations
4. Time budget distribution
5. Optimization opportunities

Goal: Data-driven decisions for v18 improvements
"""

import chess
import sys
import os
import time
import cProfile
import pstats
import io
from collections import defaultdict
from typing import Dict, List, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine


class PerformanceProfiler:
    """Profile v17.1 to find optimization opportunities"""
    
    def __init__(self):
        self.results = {
            'search_breakdown': defaultdict(float),
            'eval_breakdown': defaultdict(float),
            'function_calls': defaultdict(int),
            'time_per_function': defaultdict(float)
        }
    
    def profile_search_depth(self, positions: List[Tuple[str, str]], depth_range=(3, 7)):
        """
        Profile search at different depths to understand scaling
        """
        print("\n" + "="*80)
        print("SEARCH DEPTH SCALING ANALYSIS")
        print("="*80)
        
        results = []
        
        for depth in range(depth_range[0], depth_range[1] + 1):
            print(f"\nDepth {depth}:")
            total_nodes = 0
            total_time = 0
            
            for name, fen in positions:
                board = chess.Board(fen)
                engine = V7P3REngine(use_fast_evaluator=True)
                engine.default_depth = depth
                
                start = time.time()
                move = engine.search(board, time_limit=30.0)
                elapsed = time.time() - start
                
                total_nodes += engine.nodes_searched
                total_time += elapsed
                
                print(f"  {name[:30]:30s} | Nodes: {engine.nodes_searched:8,} | Time: {elapsed*1000:6.1f}ms | NPS: {int(engine.nodes_searched/max(elapsed, 0.001)):8,}")
            
            avg_nodes = total_nodes / len(positions)
            avg_time = total_time / len(positions)
            nps = int(total_nodes / max(total_time, 0.001))
            
            results.append({
                'depth': depth,
                'avg_nodes': avg_nodes,
                'avg_time_ms': avg_time * 1000,
                'nps': nps
            })
            
            print(f"  {'AVERAGE':30s} | Nodes: {avg_nodes:8,.0f} | Time: {avg_time*1000:6.1f}ms | NPS: {nps:8,}")
        
        # Calculate branching factor
        print(f"\n" + "-"*80)
        print("BRANCHING FACTOR ANALYSIS")
        print("-"*80)
        for i in range(len(results) - 1):
            d1, d2 = results[i], results[i+1]
            branching = d2['avg_nodes'] / max(d1['avg_nodes'], 1)
            print(f"Depth {d1['depth']}->{d2['depth']}: {branching:.1f}x nodes (effective branching factor)")
        
        return results
    
    def profile_function_hotspots(self, positions: List[Tuple[str, str]]):
        """
        Use cProfile to identify function-level bottlenecks
        """
        print("\n" + "="*80)
        print("FUNCTION-LEVEL PROFILING")
        print("="*80)
        
        # Profile a representative search
        name, fen = positions[0]
        board = chess.Board(fen)
        engine = V7P3REngine(use_fast_evaluator=True)
        
        # Create profiler
        profiler = cProfile.Profile()
        profiler.enable()
        
        # Run search
        move = engine.search(board, time_limit=5.0)
        
        profiler.disable()
        
        # Analyze results
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.strip_dirs()
        ps.sort_stats('cumulative')
        
        print(f"\nTop 30 functions by cumulative time:")
        print("-"*80)
        ps.print_stats(30)
        print(s.getvalue())
        
        # Get top time consumers
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.strip_dirs()
        ps.sort_stats('time')
        
        print(f"\nTop 30 functions by self time:")
        print("-"*80)
        ps.print_stats(30)
        print(s.getvalue())
        
        return ps
    
    def profile_evaluation_components(self, positions: List[Tuple[str, str]]):
        """
        Measure time spent in each evaluation component
        """
        print("\n" + "="*80)
        print("EVALUATION COMPONENT BREAKDOWN")
        print("="*80)
        
        from v7p3r_fast_evaluator import V7P3RFastEvaluator
        
        evaluator = V7P3RFastEvaluator()
        iterations = 10000
        
        results = {}
        
        for name, fen in positions[:3]:  # Test on 3 positions
            board = chess.Board(fen)
            print(f"\nPosition: {name}")
            print(f"FEN: {fen}")
            
            # Time full evaluation
            start = time.time()
            for _ in range(iterations):
                score = evaluator.evaluate(board)
            full_time = time.time() - start
            
            # Time material only
            start = time.time()
            for _ in range(iterations):
                score = evaluator.evaluate_material(board)
            material_time = time.time() - start
            
            # Time PST only
            start = time.time()
            for _ in range(iterations):
                score = evaluator.evaluate_pst(board)
            pst_time = time.time() - start
            
            # Time strategic only
            start = time.time()
            for _ in range(iterations):
                score = evaluator.evaluate_strategic(board)
            strategic_time = time.time() - start
            
            # Calculate percentages
            total = material_time + pst_time + strategic_time
            
            print(f"  Full eval:     {full_time*1000/iterations:.4f}ms per call")
            print(f"  Material:      {material_time*1000/iterations:.4f}ms ({material_time/total*100:.1f}%)")
            print(f"  PST:           {pst_time*1000/iterations:.4f}ms ({pst_time/total*100:.1f}%)")
            print(f"  Strategic:     {strategic_time*1000/iterations:.4f}ms ({strategic_time/total*100:.1f}%)")
            
            overhead = full_time - total
            print(f"  Overhead:      {overhead*1000/iterations:.4f}ms ({overhead/full_time*100:.1f}%)")
            
            results[name] = {
                'full': full_time,
                'material': material_time,
                'pst': pst_time,
                'strategic': strategic_time,
                'overhead': overhead
            }
        
        return results
    
    def profile_move_ordering_impact(self, positions: List[Tuple[str, str]]):
        """
        Measure impact of move ordering on search efficiency
        """
        print("\n" + "="*80)
        print("MOVE ORDERING IMPACT ANALYSIS")
        print("="*80)
        
        for name, fen in positions[:3]:
            board = chess.Board(fen)
            print(f"\nPosition: {name}")
            
            # Search with move ordering
            engine = V7P3REngine(use_fast_evaluator=True)
            engine.default_depth = 5
            start = time.time()
            move = engine.search(board, time_limit=10.0)
            with_ordering_time = time.time() - start
            with_ordering_nodes = engine.nodes_searched
            
            # Search without killer moves (crude way to reduce ordering quality)
            engine2 = V7P3REngine(use_fast_evaluator=True)
            engine2.default_depth = 5
            engine2.killer_moves.killers = {}  # Disable killer moves
            start = time.time()
            move2 = engine2.search(board, time_limit=10.0)
            without_ordering_time = time.time() - start
            without_ordering_nodes = engine2.nodes_searched
            
            improvement = (without_ordering_nodes / max(with_ordering_nodes, 1) - 1) * 100
            
            print(f"  With ordering:    {with_ordering_nodes:8,} nodes in {with_ordering_time*1000:6.1f}ms")
            print(f"  Without killers:  {without_ordering_nodes:8,} nodes in {without_ordering_time*1000:6.1f}ms")
            print(f"  Improvement:      {improvement:+.1f}% fewer nodes with killer moves")
    
    def analyze_cache_efficiency(self, positions: List[Tuple[str, str]]):
        """
        Analyze transposition table and eval cache hit rates
        """
        print("\n" + "="*80)
        print("CACHE EFFICIENCY ANALYSIS")
        print("="*80)
        
        for name, fen in positions[:3]:
            board = chess.Board(fen)
            engine = V7P3REngine(use_fast_evaluator=True)
            
            move = engine.search(board, time_limit=5.0)
            
            # Analyze cache statistics
            tt_hits = engine.search_stats.get('tt_hits', 0)
            tt_misses = engine.search_stats.get('tt_misses', 0)
            tt_total = tt_hits + tt_misses
            tt_rate = (tt_hits / max(tt_total, 1)) * 100
            
            eval_hits = engine.search_stats.get('cache_hits', 0)
            eval_misses = engine.search_stats.get('cache_misses', 0)
            eval_total = eval_hits + eval_misses
            eval_rate = (eval_hits / max(eval_total, 1)) * 100
            
            print(f"\n{name}:")
            print(f"  Transposition Table:")
            print(f"    Hits:   {tt_hits:8,} ({tt_rate:.1f}%)")
            print(f"    Misses: {tt_misses:8,}")
            print(f"  Evaluation Cache:")
            print(f"    Hits:   {eval_hits:8,} ({eval_rate:.1f}%)")
            print(f"    Misses: {eval_misses:8,}")


def main():
    print("="*80)
    print("V17.1 PERFORMANCE PROFILER")
    print("Identifying optimization opportunities for v18")
    print("="*80)
    
    # Test positions covering different game phases
    positions = [
        ("Opening - Italian Game", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"),
        ("Middlegame - Tactical", "r1bq1rk1/ppp2ppp/2n2n2/3p4/1b1P4/2NBP3/PPP2PPP/R1BQK2R w KQ - 0 1"),
        ("Middlegame - Strategic", "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1"),
        ("Endgame - R+P vs R", "6k1/5ppp/8/8/8/8/r4PPP/4R1K1 w - - 0 1"),
        ("Endgame - Pawn Race", "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1"),
        ("Desperate - Down Material", "rnbqkbnr/pppppppp/8/8/8/8/PPPP1PPP/RNBK1BNR w KQkq - 0 1"),
    ]
    
    profiler = PerformanceProfiler()
    
    # 1. Search depth scaling
    depth_results = profiler.profile_search_depth(positions[:3])
    
    # 2. Function-level hotspots
    function_stats = profiler.profile_function_hotspots(positions[:1])
    
    # 3. Evaluation component breakdown
    eval_results = profiler.profile_evaluation_components(positions)
    
    # 4. Move ordering impact
    profiler.profile_move_ordering_impact(positions)
    
    # 5. Cache efficiency
    profiler.analyze_cache_efficiency(positions)
    
    # Summary and recommendations
    print("\n" + "="*80)
    print("OPTIMIZATION RECOMMENDATIONS")
    print("="*80)
    print("""
Based on profiling data, prioritize:

1. SEARCH OPTIMIZATIONS (biggest impact on depth):
   - Late Move Reduction (LMR) tuning
   - Null Move Pruning
   - Principal Variation Search
   - Aspiration Windows

2. MOVE ORDERING (better cutoffs = fewer nodes):
   - History heuristic improvements
   - Killer move enhancements
   - SEE (Static Exchange Evaluation)
   - MVV-LVA for captures

3. EVALUATION (only if spending >30% of time):
   - Cache expensive components
   - Lazy evaluation (defer until needed)
   - Incremental updates

4. CACHING (if hit rates <50%):
   - Larger transposition table
   - Better replacement scheme
   - Zobrist hashing improvements

Run this profiler on v17.1 vs v18 to ensure no regressions!
    """)


if __name__ == "__main__":
    main()
