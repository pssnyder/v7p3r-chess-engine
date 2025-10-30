#!/usr/bin/env python3
"""
Debug test to see why search is stopping early
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine
import time

def test_search_depth():
    """Test why search stops at depth 2"""
    print("Debugging V14.5 Search Depth Issue")
    print("=" * 60)
    
    engine = V7P3REngine()
    board = chess.Board()
    
    print(f"\nStarting position")
    print(f"Time limit: 3.0 seconds")
    
    # Calculate what the time allocations should be (updated for v14.5)
    base_time = 3.0
    if base_time <= 3.0:
        target_time = base_time * 0.7  # 2.1s
        max_time = base_time * 0.9     # 2.7s
    
    print(f"Expected target_time: {target_time}s")
    print(f"Expected max_time: {max_time}s")
    print(f"Expected 85% threshold: {base_time * 0.85}s (2.55s)")
    print(f"Expected max_time * 0.85: {max_time * 0.85}s (2.04s)")
    
    print("\n" + "="*60)
    print("Running search...")
    start = time.time()
    
    best_move = engine.search(board, time_limit=3.0)
    
    elapsed = time.time() - start
    print(f"\nSearch completed in {elapsed:.3f}s")
    print(f"Best move: {best_move}")
    
    print(f"\nWhy did it stop?")
    print(f"- Used {elapsed:.3f}s of 3.0s available ({elapsed/3.0*100:.1f}%)")
    print(f"- 85% threshold would be {3.0*0.85:.2f}s")
    print(f"- max_time * 0.85 threshold would be {max_time * 0.85:.2f}s")

if __name__ == "__main__":
    test_search_depth()
