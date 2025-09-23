#!/usr/bin/env python3
"""
V11.5 Deep Search Tactical Cache Test
====================================

Test tactical cache with deeper search where repeated positions
should yield much higher cache hit rates.
"""

import sys
import os
import time

# Add the source path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.v7p3r import V7P3REngine
import chess

def test_deep_tactical_cache():
    print("V11.5 Deep Search Tactical Cache Test")
    print("=====================================")
    print()
    
    # Create engine
    engine = V7P3REngine()
    
    # Test position with tactical complexity
    test_fen = "r1bq1rk1/ppp2ppp/2n1bn2/2bpp3/2PP4/2NBPN2/PP3PPP/R1BQKR2 w K - 0 8"
    board = chess.Board(test_fen)
    
    print(f"Test Position: {test_fen}")
    print(f"Turn: {'White' if board.turn else 'Black'}")
    print()
    
    # Test with depth 5 for more transpositions
    print("Running deep search (depth 5) to maximize cache effectiveness...")
    start_time = time.time()
    
    result = engine.search(board, depth=5)
    
    search_time = time.time() - start_time
    
    print(f"Search completed in {search_time:.3f}s")
    print(f"Best move: {result[0] if result else 'None'}")
    print(f"Nodes searched: {engine.nodes_searched}")
    
    if engine.nodes_searched > 0:
        nps = engine.nodes_searched / search_time
        print(f"NPS: {nps:.0f}")
    
    # Check tactical cache statistics
    if hasattr(engine, 'tactical_cache'):
        cache_stats = engine.tactical_cache.get_stats()
        print()
        print("Tactical Cache Statistics:")
        print(f"- Cache hits: {cache_stats['hits']}")
        print(f"- Cache misses: {cache_stats['misses']}")
        print(f"- Hit rate: {cache_stats['hit_rate_percent']:.1f}%")
        print(f"- Cache size: {cache_stats['cache_size']} positions")
        
        # Calculate performance improvement
        total_tactical_calls = cache_stats['hits'] + cache_stats['misses']
        tactical_savings = cache_stats['hits'] / total_tactical_calls * 100 if total_tactical_calls > 0 else 0
        
        print()
        print("Performance Analysis:")
        print(f"- Total tactical evaluations: {total_tactical_calls}")
        print(f"- Tactical calculations saved: {tactical_savings:.1f}%")
        print(f"- Estimated time savings: {tactical_savings * 0.8:.1f}% (assuming tactics = 80% of search time)")
        
        if cache_stats['hit_rate_percent'] > 40:
            print("✅ Tactical cache is providing significant performance boost!")
        elif cache_stats['hit_rate_percent'] > 20:
            print("✅ Tactical cache is working well")
        else:
            print("⚠️  Cache hit rate could be improved")
    
    print()
    print("Comparison vs V11.4:")
    print("- V11.4 NPS: 300-600 (without cache)")
    print(f"- V11.5 NPS: {nps:.0f} (with tactical cache)")
    improvement_factor = nps / 450  # Using 450 as average of 300-600
    print(f"- Speed improvement: {improvement_factor:.1f}x faster")

if __name__ == "__main__":
    test_deep_tactical_cache()