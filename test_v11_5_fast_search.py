#!/usr/bin/env python3
"""
V11.5 Fast Search Performance Test
==================================

Test the new high-performance search implementation
Target: 10,000+ NPS (vs current 300-600)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import time
import chess
from v7p3r import V7P3REngine

def test_fast_search_performance():
    """Test the new fast search performance"""
    print("=== V11.5 FAST SEARCH PERFORMANCE TEST ===")
    
    # Test positions
    positions = [
        ("Starting Position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("After e4", "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),
        ("Tactical Position", "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 0 4"),
    ]
    
    engine = V7P3REngine()
    
    total_nodes = 0
    total_time = 0
    
    for name, fen in positions:
        print(f"\n--- {name} ---")
        board = chess.Board(fen)
        
        # Test fast search
        start_time = time.time()
        try:
            result = engine.search(board, depth=4, time_limit=10.0)
            move, score, search_info = result
            
            elapsed = time.time() - start_time
            nodes = search_info.get('nodes', 0)
            nps = nodes / max(elapsed, 0.001)
            
            print(f"Move: {move}")
            print(f"Score: {score}")
            print(f"Nodes: {nodes:,}")
            print(f"Time: {elapsed:.3f}s")
            print(f"NPS: {nps:,.0f}")
            print(f"Tactical cache stats: {engine.tactical_cache.get_stats()}")
            
            total_nodes += nodes
            total_time += elapsed
            
        except Exception as e:
            print(f"Search failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Overall performance
    if total_time > 0:
        overall_nps = total_nodes / total_time
        print(f"\n=== OVERALL PERFORMANCE ===")
        print(f"Total nodes: {total_nodes:,}")
        print(f"Total time: {total_time:.3f}s")
        print(f"Average NPS: {overall_nps:,.0f}")
        
        if overall_nps > 10000:
            print("✅ SUCCESS: NPS target achieved!")
        elif overall_nps > 5000:
            print("⚠️  GOOD: Significant improvement, but can do better")
        elif overall_nps > 2000:
            print("📈 BETTER: Improvement over v11.4, but still slow")
        else:
            print("❌ POOR: Still too slow, need more optimization")

def compare_with_v11_4():
    """Quick comparison with v11.4 performance"""
    print("\n=== COMPARISON WITH v11.4 ===")
    print("v11.4 NPS: 300-600")
    print("v11.5 Target: 10,000+")
    print("Expected improvement: 20x faster")

if __name__ == "__main__":
    test_fast_search_performance()
    compare_with_v11_4()