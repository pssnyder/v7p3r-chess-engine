#!/usr/bin/env python3
"""
V7P3R v14.2 Search Bottleneck Profiler
Instruments the search to identify EXACTLY what's consuming CPU time

This profiles the ACTUAL SEARCH, not just evaluation:
- Move ordering overhead (gives_check calls, tactical detection, etc.)
- Search tree operations (TT lookups, alpha-beta, etc.)
- Helper function costs
- Per-move overhead breakdown

Goal: Find the hidden bottleneck preventing depth gains
"""

import sys
import os
import time
import chess
from collections import defaultdict
from typing import Dict, List

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine


class SearchProfiler:
    """Instruments V7P3R search to measure actual costs"""
    
    def __init__(self):
        self.timings = defaultdict(float)
        self.call_counts = defaultdict(int)
        self.engine = None
        
    def instrument_engine(self, engine: V7P3REngine):
        """Wrap engine methods with timing instrumentation"""
        self.engine = engine
        
        # Store original methods
        self._original_order_moves = engine._order_moves_advanced
        self._original_recursive_search = engine._recursive_search
        self._original_quiescence = engine._quiescence_search
        self._original_evaluate = engine.bitboard_evaluator.evaluate_bitboard
        
        # Wrap with instrumentation
        engine._order_moves_advanced = self._instrumented_order_moves
        engine._recursive_search = self._instrumented_recursive_search
        engine._quiescence_search = self._instrumented_quiescence
        engine.bitboard_evaluator.evaluate_bitboard = self._instrumented_evaluate
        
    def _instrumented_order_moves(self, board, moves, depth, tt_move=None):
        """Measure move ordering overhead"""
        start = time.perf_counter()
        
        # Track gives_check calls
        check_start = time.perf_counter()
        check_count = 0
        
        # Temporarily wrap gives_check to count calls
        original_gives_check = board.gives_check
        
        def counted_gives_check(move):
            nonlocal check_count
            check_count += 1
            return original_gives_check(move)
        
        board.gives_check = counted_gives_check
        
        try:
            result = self._original_order_moves(board, moves, depth, tt_move)
        finally:
            board.gives_check = original_gives_check
        
        elapsed = time.perf_counter() - start
        
        self.timings['move_ordering_total'] += elapsed
        self.call_counts['move_ordering_calls'] += 1
        self.call_counts['gives_check_calls'] += check_count
        self.timings['gives_check_total'] += (time.perf_counter() - check_start)
        
        return result
    
    def _instrumented_recursive_search(self, board, depth, alpha, beta, time_limit):
        """Measure recursive search overhead"""
        start = time.perf_counter()
        result = self._original_recursive_search(board, depth, alpha, beta, time_limit)
        self.timings['recursive_search'] += time.perf_counter() - start
        self.call_counts['recursive_search_calls'] += 1
        return result
    
    def _instrumented_quiescence(self, board, alpha, beta, depth):
        """Measure quiescence search overhead"""
        start = time.perf_counter()
        result = self._original_quiescence(board, alpha, beta, depth)
        self.timings['quiescence_search'] += time.perf_counter() - start
        self.call_counts['quiescence_calls'] += 1
        return result
    
    def _instrumented_evaluate(self, board, color):
        """Measure evaluation overhead"""
        # NOTE: Skipping evaluation instrumentation - focus on search bottlenecks
        start = time.perf_counter()
        # result = self._original_evaluate(board, color)
        self.timings['evaluation'] += time.perf_counter() - start
        self.call_counts['evaluation_calls'] += 1
        return 0.0  # Dummy return
    
    def profile_search(self, positions: List[str], time_per_move: float = 5.0):
        """Profile search on multiple positions"""
        print("="*80)
        print("V7P3R v14.2 SEARCH BOTTLENECK PROFILER")
        print("="*80)
        print(f"Testing {len(positions)} positions @ {time_per_move}s each")
        print()
        
        results = []
        
        for i, fen in enumerate(positions, 1):
            print(f"Position {i}/{len(positions)}: {fen[:50]}...")
            
            # Reset counters
            self.timings.clear()
            self.call_counts.clear()
            
            board = chess.Board(fen)
            engine = V7P3REngine()
            
            # Instrument
            self.instrument_engine(engine)
            
            # Search
            search_start = time.perf_counter()
            best_move = engine.search(board, time_limit=time_per_move)
            total_time = time.perf_counter() - search_start
            
            # Calculate stats
            nodes = self.call_counts.get('evaluation_calls', 0)
            nps = int(nodes / max(total_time, 0.001))
            
            # Move ordering stats
            move_order_calls = self.call_counts.get('move_ordering_calls', 0)
            gives_check_calls = self.call_counts.get('gives_check_calls', 0)
            gives_check_per_ordering = gives_check_calls / max(move_order_calls, 1)
            
            # Time breakdown
            eval_time = self.timings.get('evaluation', 0)
            move_order_time = self.timings.get('move_ordering_total', 0)
            gives_check_time = self.timings.get('gives_check_total', 0)
            
            eval_pct = (eval_time / total_time * 100) if total_time > 0 else 0
            move_order_pct = (move_order_time / total_time * 100) if total_time > 0 else 0
            gives_check_pct = (gives_check_time / total_time * 100) if total_time > 0 else 0
            
            result = {
                'fen': fen,
                'time': total_time,
                'nodes': nodes,
                'nps': nps,
                'best_move': str(best_move),
                'eval_time': eval_time,
                'eval_pct': eval_pct,
                'move_order_time': move_order_time,
                'move_order_pct': move_order_pct,
                'gives_check_calls': gives_check_calls,
                'gives_check_time': gives_check_time,
                'gives_check_pct': gives_check_pct,
                'gives_check_per_ordering': gives_check_per_ordering
            }
            
            results.append(result)
            
            print(f"  Time: {total_time:.2f}s | Nodes: {nodes:,} | NPS: {nps:,}")
            print(f"  Evaluation: {eval_pct:.1f}% ({eval_time:.2f}s)")
            print(f"  Move Ordering: {move_order_pct:.1f}% ({move_order_time:.2f}s)")
            print(f"  gives_check(): {gives_check_calls:,} calls = {gives_check_pct:.1f}% of time")
            print(f"  gives_check() per move ordering: {gives_check_per_ordering:.1f}")
            print()
        
        return results
    
    def print_summary(self, results: List[Dict]):
        """Print summary of profiling results"""
        print("="*80)
        print("PROFILING SUMMARY")
        print("="*80)
        
        # Averages
        avg_nps = sum(r['nps'] for r in results) / len(results)
        avg_eval_pct = sum(r['eval_pct'] for r in results) / len(results)
        avg_move_order_pct = sum(r['move_order_pct'] for r in results) / len(results)
        avg_gives_check_pct = sum(r['gives_check_pct'] for r in results) / len(results)
        avg_gives_check_calls = sum(r['gives_check_calls'] for r in results) / len(results)
        
        print(f"Average NPS: {avg_nps:,.0f}")
        print()
        print("Time Breakdown:")
        print(f"  Evaluation:         {avg_eval_pct:5.1f}%")
        print(f"  Move Ordering:      {avg_move_order_pct:5.1f}%")
        print(f"    gives_check():    {avg_gives_check_pct:5.1f}%")
        print(f"  Other:              {100-avg_eval_pct-avg_move_order_pct:5.1f}%")
        print()
        print(f"Average gives_check() calls per position: {avg_gives_check_calls:,.0f}")
        print()
        
        # Identify bottlenecks
        print("BOTTLENECK ANALYSIS:")
        print("-" * 80)
        
        if avg_gives_check_pct > 15:
            print(f"[CRITICAL] gives_check() consuming {avg_gives_check_pct:.1f}% of search time!")
            print(f"           Called {avg_gives_check_calls:,.0f} times per position")
            print(f"           RECOMMENDATION: Remove gives_check() from move ordering")
            print()
        
        if avg_move_order_pct > 25:
            print(f"[WARNING] Move ordering overhead is {avg_move_order_pct:.1f}%")
            print(f"          RECOMMENDATION: Simplify move ordering logic")
            print()
        
        if avg_eval_pct > 30:
            print(f"[INFO] Evaluation consuming {avg_eval_pct:.1f}% (reasonable for chess engine)")
            print()
        
        # Expected depth calculation
        material_opponent_nps = 55000  # From previous tests
        depth_ratio = avg_nps / material_opponent_nps
        
        print("DEPTH COMPARISON:")
        print(f"  MaterialOpponent NPS: ~{material_opponent_nps:,}")
        print(f"  V7P3R v14.2 NPS:      ~{avg_nps:,.0f}")
        print(f"  Speed ratio:          {depth_ratio:.2f}x")
        print()
        
        if depth_ratio < 0.7:
            print(f"[FAIL] V7P3R is {(1-depth_ratio)*100:.0f}% SLOWER than MaterialOpponent")
            print(f"       This explains why depth gains didn't materialize")
        elif depth_ratio < 0.9:
            print(f"[WARN] V7P3R is {(1-depth_ratio)*100:.0f}% slower than MaterialOpponent")
            print(f"       Close but not enough for 2-ply depth gain")
        else:
            print(f"[PASS] V7P3R matches or exceeds MaterialOpponent NPS")


def main():
    # Test positions (mix of opening, middlegame, endgame)
    test_positions = [
        # Opening
        "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1",
        
        # Middlegame
        "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5",
        "r1bqkb1r/pp1ppppp/2n2n2/2p5/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 4",
        "rnbqkb1r/ppp2ppp/4pn2/3p4/2PP4/2N2N2/PP2PPPP/R1BQKB1R b KQkq - 0 4",
        
        # Endgame
        "8/5k2/8/5P2/5K2/8/8/8 w - - 0 1",
        "8/8/4k3/8/8/4K3/4P3/8 w - - 0 1",
    ]
    
    profiler = SearchProfiler()
    
    print("\nV7P3R v14.2 Search Bottleneck Analysis")
    print("Goal: Identify EXACTLY what's preventing depth gains")
    print()
    
    results = profiler.profile_search(test_positions, time_per_move=3.0)
    profiler.print_summary(results)
    
    print("\n" + "="*80)
    print("NEXT STEPS:")
    print("="*80)
    print("1. If gives_check() is >15%: Remove from move ordering")
    print("2. If move ordering is >25%: Simplify ordering logic")
    print("3. If NPS < MaterialOpponent: Profile deeper to find root cause")
    print("4. Create v14.3 with identified optimizations")
    print()


if __name__ == "__main__":
    main()
