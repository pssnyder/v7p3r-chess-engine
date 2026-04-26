"""
Quick performance validation for v19.1 emergency fix

This will benchmark the key metrics:
1. NPS (nodes per second) - target 100,000+ (was 6,800)
2. Depth reached in 5 seconds - target 8-10 (was 3-6)
3. Move ordering time - target <0.2ms per position (was 2.9ms)
"""

import sys
import os
import chess
import time

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine

def quick_depth_test():
    """Test how deep we can search in 5 seconds"""
    print("=" * 80)
    print("V19.1 QUICK PERFORMANCE TEST")
    print("=" * 80)
    
    engine = V7P3REngine()
    board = chess.Board()
    
    # Complex middlegame position
    board.set_fen("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1")
    
    print(f"\nTest Position: {board.fen()}")
    print(f"Legal moves: {len(list(board.legal_moves))}")
    print("\nSearching for 5 seconds...\n")
    
    # Search with 5 second time limit
    start = time.time()
    result = engine.search(board, time_limit=5.0)
    elapsed = time.time() - start
    
    print(f"\n{'=' * 80}")
    print(f"RESULTS:")
    print(f"  Time: {elapsed:.2f}s")
    print(f"  Best move: {result}")
    print(f"  Search stats: {engine.search_stats}")
    
    # Calculate NPS
    nodes = engine.search_stats.get('nodes_searched', 0)
    if nodes > 0 and elapsed > 0:
        nps = nodes / elapsed
        print(f"\n  Nodes: {nodes:,}")
        print(f"  NPS: {nps:,.0f}")
        
        # Determine success
        print(f"\n{'=' * 80}")
        if nps >= 80000:
            print("✓ SUCCESS: NPS is 80,000+ (target achieved!)")
        elif nps >= 50000:
            print("✓ GOOD: NPS is 50,000+ (significant improvement)")
        elif nps >= 20000:
            print("⚠ PARTIAL: NPS is 20,000+ (better but not target)")
        else:
            print(f"✗ FAIL: NPS is {nps:,.0f} (still too slow)")
            
        print(f"{'=' * 80}\n")

if __name__ == '__main__':
    quick_depth_test()
