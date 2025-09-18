#!/usr/bin/env python3
"""
V7P3R Chess Engine - Phase 3A Integration Performance Test
Tests that defensive analysis integration maintains high NPS performance
"""

import sys
import os
import time
import chess

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine

def test_performance_integration():
    """Test that the full engine with defensive integration maintains NPS targets"""
    
    print("V7P3R Phase 3A Integration Performance Test")
    print("=" * 60)
    
    # Initialize engine
    engine = V7P3REngine()
    
    # Test positions - standard middlegame positions
    test_positions = [
        "rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 3",  # Italian Game
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 4 4",  # Scotch Game
        "rnbqk2r/pppp1ppp/5n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 5",  # Italian Variation
        "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 3",  # Four Knights
        "rnbqkb1r/ppp2ppp/3p1n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 4",  # Caro-Kann Defense
    ]
    
    total_nodes = 0
    total_time = 0
    position_count = 0
    
    for i, fen in enumerate(test_positions):
        print(f"\nTesting position {i+1}: {fen[:30]}...")
        
        board = chess.Board(fen)
        
        # Set a reasonable search depth for performance testing
        depth = 4
        
        start_time = time.time()
        
        # Perform search
        best_move = engine.search(board, depth=depth)
        
        end_time = time.time()
        search_time = end_time - start_time
        
        # Get node count
        nodes = engine.nodes_searched
        nps = nodes / search_time if search_time > 0 else 0
        
        print(f"  Best move: {best_move}")
        print(f"  Nodes: {nodes:,}")
        print(f"  Time: {search_time:.3f}s")
        print(f"  NPS: {nps:,.0f}")
        
        # Accumulate stats
        total_nodes += nodes
        total_time += search_time
        position_count += 1
        
        # Reset engine for next test
        engine.nodes_searched = 0
        engine.depth_reached = 0
    
    # Calculate overall performance
    average_nps = total_nodes / total_time if total_time > 0 else 0
    
    print("\n" + "=" * 60)
    print("OVERALL PERFORMANCE RESULTS:")
    print(f"Total positions tested: {position_count}")
    print(f"Total nodes searched: {total_nodes:,}")
    print(f"Total time: {total_time:.3f}s")
    print(f"Average NPS: {average_nps:,.0f}")
    
    # Performance targets
    target_nps = 25000  # Conservative target
    performance_pass = average_nps >= target_nps
    
    print(f"\nPerformance Target: {target_nps:,} NPS")
    print(f"Performance Result: {'PASS' if performance_pass else 'FAIL'}")
    
    if not performance_pass:
        print(f"WARNING: Performance below target by {target_nps - average_nps:,.0f} NPS")
    else:
        print(f"Excellent: Performance exceeds target by {average_nps - target_nps:,.0f} NPS")
    
    return performance_pass

def test_defensive_cache_efficiency():
    """Test the cache efficiency of the integrated defensive analysis"""
    
    print("\n" + "=" * 60)
    print("DEFENSIVE CACHE EFFICIENCY TEST:")
    print("=" * 60)
    
    engine = V7P3REngine()
    
    # Use a position that will trigger defensive analysis
    board = chess.Board("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 4 3")
    
    # Perform multiple searches to test cache
    search_count = 10
    depth = 3
    
    print(f"Performing {search_count} searches at depth {depth}...")
    
    cache_hits_before = engine.lightweight_defense.performance_stats['cache_hits']
    cache_requests_before = engine.lightweight_defense.performance_stats['calls']
    
    for i in range(search_count):
        engine.search(board, depth=depth)
        engine.nodes_searched = 0  # Reset for consistency
    
    cache_hits_after = engine.lightweight_defense.performance_stats['cache_hits']
    cache_requests_after = engine.lightweight_defense.performance_stats['calls']
    
    new_hits = cache_hits_after - cache_hits_before
    new_requests = cache_requests_after - cache_requests_before
    
    if new_requests > 0:
        hit_rate = (new_hits / new_requests) * 100
        print(f"Cache requests: {new_requests}")
        print(f"Cache hits: {new_hits}")
        print(f"Hit rate: {hit_rate:.1f}%")
        
        target_hit_rate = 70.0  # Reasonable target
        cache_pass = hit_rate >= target_hit_rate
        print(f"Cache Target: {target_hit_rate:.1f}%")
        print(f"Cache Result: {'PASS' if cache_pass else 'FAIL'}")
        
        return cache_pass
    else:
        print("No cache requests detected - defensive analysis may not be triggering")
        return False

if __name__ == "__main__":
    print("Starting V7P3R Phase 3A Integration Performance Test...")
    
    try:
        # Test overall performance
        performance_result = test_performance_integration()
        
        # Test cache efficiency
        cache_result = test_defensive_cache_efficiency()
        
        print("\n" + "=" * 60)
        print("FINAL TEST SUMMARY:")
        print(f"Performance Test: {'PASS' if performance_result else 'FAIL'}")
        print(f"Cache Test: {'PASS' if cache_result else 'FAIL'}")
        
        overall_pass = performance_result and cache_result
        print(f"Overall Result: {'PASS' if overall_pass else 'FAIL'}")
        
        if overall_pass:
            print("\n✓ Phase 3A integration is performance-ready!")
        else:
            print("\n✗ Phase 3A integration needs optimization before proceeding.")
            
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()