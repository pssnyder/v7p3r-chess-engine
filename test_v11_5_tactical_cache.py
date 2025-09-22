#!/usr/bin/env python3
"""
V11.5 Tactical Cache Performance Test
====================================

Test the tactical cache performance fix by running a quick search
and measuring the improvement in NPS and cache hit rate.
"""

import sys
import os
import time

# Add the source path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.v7p3r import V7P3REngine
import chess

def test_tactical_cache_performance():
    print("V11.5 Tactical Cache Performance Test")
    print("=====================================")
    print()
    
    # Create engine
    engine = V7P3REngine()
    
    # Test position (middlegame with tactical possibilities)
    test_fen = "r1bq1rk1/ppp2ppp/2n1bn2/2bpp3/2PP4/2NBPN2/PP3PPP/R1BQKR2 w K - 0 8"
    board = chess.Board(test_fen)
    
    print(f"Test Position: {test_fen}")
    print(f"Turn: {'White' if board.turn else 'Black'}")
    print()
    
    # Run search to populate cache and measure performance
    print("Running tactical cache performance test...")
    start_time = time.time()
    
    # Search with depth 4 to generate tactical cache entries
    result = engine.search(board, depth=4)
    
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
        print(f"- Cache enabled: {cache_stats['enabled']}")
        
        if cache_stats['hit_rate_percent'] > 50:
            print("✅ Tactical cache is working effectively!")
        else:
            print("⚠️  Cache hit rate is lower than expected")
    else:
        print("❌ Tactical cache not found!")
    
    print()
    print("Expected improvements with tactical cache:")
    print("- 80-90% cache hit rate for repeated positions")
    print("- 5-10x improvement in search speed")
    print("- Target NPS: 2000-5000+ (vs previous 300-600)")

if __name__ == "__main__":
    test_tactical_cache_performance()