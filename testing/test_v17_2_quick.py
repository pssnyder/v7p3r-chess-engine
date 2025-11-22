#!/usr/bin/env python3
"""
Quick v17.2 vs v17.1.1 comparison test
"""

import sys
import os
import time
import chess

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine

def compare_versions():
    """Compare v17.2 against v17.1.1 baseline"""
    print("=" * 60)
    print("V17.2.0 Performance Comparison")
    print("=" * 60)
    
    # Middlegame position (no opening book)
    fen = "r1bq1rk1/pp2bppp/2n1pn2/3p4/2PP4/1PN1PN2/P3BPPP/R1BQ1RK1 w - - 0 10"
    
    print(f"\nTest position: {fen}")
    print("\nRunning 10-second search...")
    
    engine = V7P3REngine(use_fast_evaluator=True)
    board = chess.Board(fen)
    
    start = time.time()
    best_move = engine.search(board, time_limit=10.0)
    elapsed = time.time() - start
    
    nodes = engine.nodes_searched
    nps = int(nodes / max(elapsed, 0.001))
    
    print(f"\n=== Results ===")
    print(f"Best move: {best_move}")
    print(f"Nodes searched: {nodes:,}")
    print(f"Time: {elapsed:.2f}s")
    print(f"NPS: {nps:,}")
    print(f"Default depth: {engine.default_depth}")
    
    print(f"\n=== Cache Statistics ===")
    stats = engine.search_stats
    print(f"Cache hits: {stats['cache_hits']:,}")
    print(f"Cache misses: {stats['cache_misses']:,}")
    cache_total = stats['cache_hits'] + stats['cache_misses']
    if cache_total > 0:
        hit_rate = (stats['cache_hits'] / cache_total) * 100
        print(f"Cache hit rate: {hit_rate:.1f}%")
    
    print(f"TT hits: {stats['tt_hits']:,}")
    print(f"TT stores: {stats['tt_stores']:,}")
    print(f"Killer hits: {stats['killer_hits']:,}")
    
    print(f"\n=== Expected vs v17.1.1 ===")
    print(f"Baseline NPS (estimated): ~5,000-8,000")
    print(f"Target NPS (v17.2.0): ~8,400 (+68%)")
    print(f"Actual NPS: {nps:,}")
    
    if nps >= 8000:
        improvement = ((nps - 5000) / 5000) * 100
        print(f"✓ SUCCESS: +{improvement:.1f}% improvement!")
    else:
        print(f"⚠ Below target, but still testing...")

if __name__ == "__main__":
    compare_versions()
