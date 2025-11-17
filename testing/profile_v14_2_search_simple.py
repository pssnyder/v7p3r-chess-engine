#!/usr/bin/env python3
"""
V7P3R v14.2 SIMPLE Search Bottleneck Profiler
Focus: gives_check() call count and move ordering overhead

This is the KEY investigation: how many times does V7P3R call gives_check()?
"""

import sys
import os
import time
import chess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))
from v7p3r import V7P3REngine


class SimpleSearchProfiler:
    """Count gives_check() calls during search"""
    
    def __init__(self):
        self.gives_check_count = 0
        self.move_ordering_time = 0.0
        self.search_time = 0.0
        
    def count_gives_check_in_search(self, fen: str, time_limit: float = 5.0):
        """Profile a single position focusing on gives_check() calls"""
        board = chess.Board(fen)
        engine = V7P3REngine()
        
        # Wrap gives_check to count calls
        original_gives_check = chess.Board.gives_check
        gives_check_calls = [0]  # Use list to allow modification in closure
        
        def counted_gives_check(self, move):
            gives_check_calls[0] += 1
            return original_gives_check(self, move)
        
        chess.Board.gives_check = counted_gives_check
        
        try:
            search_start = time.perf_counter()
            best_move = engine.search(board, time_limit=time_limit)
            search_time = time.perf_counter() - search_start
            
            nodes = engine.nodes_searched
            nps = int(nodes / max(search_time, 0.001))
            
            return {
                'fen': fen,
                'best_move': str(best_move) if best_move else 'None',
                'time': search_time,
                'nodes': nodes,
                'nps': nps,
                'gives_check_calls': gives_check_calls[0],
                'gives_check_per_node': gives_check_calls[0] / max(nodes, 1)
            }
        finally:
            chess.Board.gives_check = original_gives_check


def main():
    print("="*80)
    print("V7P3R v14.2 gives_check() CALL PROFILER")
    print("="*80)
    print("CRITICAL QUESTION: How many times does V7P3R call gives_check()?")
    print("MaterialOpponent: Only for checkmate verification")
    print("V7P3R: For EVERY move in ordering (suspect bottleneck)")
    print()
    
    # Test positions
    test_positions = [
        ("Starting Position", chess.STARTING_FEN),
        ("Italian Game", "r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3"),
        ("Ruy Lopez", "r1bqkbnr/1ppp1ppp/p1n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 4"),
        ("French Defense", "rnbqkbnr/ppp2ppp/4p3/3p4/3PP3/8/PPP2PPP/RNBQKBNR w KQkq d6 0 3"),
        ("Middlegame", "r2qkb1r/ppp2ppp/2n1pn2/3p4/2PP4/2N2NP1/PP2PP1P/R1BQKB1R w KQkq - 0 7"),
        ("Endgame", "8/5k2/3p4/1p1Pp2p/pP2Pp1P/P4P1K/8/8 w - - 0 1"),
    ]
    
    profiler = SimpleSearchProfiler()
    results = []
    
    for name, fen in test_positions:
        print(f"Testing: {name}")
        print(f"FEN: {fen[:60]}...")
        
        result = profiler.count_gives_check_in_search(fen, time_limit=3.0)
        results.append((name, result))
        
        print(f"  Time: {result['time']:.2f}s")
        print(f"  Nodes: {result['nodes']:,}")
        print(f"  NPS: {result['nps']:,}")
        print(f"  gives_check() calls: {result['gives_check_calls']:,}")
        print(f"  Ratio: {result['gives_check_per_node']:.2f} gives_check/node")
        print()
    
    # Summary
    print("="*80)
    print("SUMMARY")
    print("="*80)
    
    total_nodes = sum(r['nodes'] for _, r in results)
    total_checks = sum(r['gives_check_calls'] for _, r in results)
    total_time = sum(r['time'] for _, r in results)
    
    print(f"Total nodes: {total_nodes:,}")
    print(f"Total gives_check() calls: {total_checks:,}")
    print(f"Average ratio: {total_checks / max(total_nodes, 1):.2f} checks/node")
    print(f"Total time: {total_time:.2f}s")
    print()
    
    # Analysis
    print("="*80)
    print("BOTTLENECK ANALYSIS")
    print("="*80)
    print()
    print(f"If MaterialOpponent only checks for checkmate (1-2 calls per 100 nodes)")
    print(f"and V7P3R calls gives_check() {total_checks / max(total_nodes, 1):.2f} times per node,")
    print(f"that's approximately {(total_checks / max(total_nodes, 1)) / 0.02:.0f}x more calls!")
    print()
    
    # Cost estimate
    avg_check_cost = 20  # μs estimate for gives_check()
    check_overhead_time = (total_checks * avg_check_cost) / 1_000_000
    check_overhead_percent = (check_overhead_time / total_time) * 100
    
    print(f"Estimated gives_check() cost:")
    print(f"  Calls: {total_checks:,}")
    print(f"  @ ~{avg_check_cost}μs per call")
    print(f"  Total overhead: {check_overhead_time:.2f}s ({check_overhead_percent:.1f}% of search time)")
    print()
    
    if check_overhead_percent > 20:
        print("❌ CRITICAL BOTTLENECK FOUND!")
        print(f"gives_check() is consuming {check_overhead_percent:.1f}% of search time")
        print("Recommendation: Remove gives_check() from general move ordering")
        print("Alternative: Use MaterialOpponent's approach (checkmate threats only)")
    elif check_overhead_percent > 10:
        print("⚠️  SIGNIFICANT OVERHEAD")
        print(f"gives_check() is consuming {check_overhead_percent:.1f}% of search time")
        print("Consider optimization")
    else:
        print("✓ gives_check() overhead acceptable")
        print("Look for other bottlenecks")


if __name__ == '__main__':
    main()
