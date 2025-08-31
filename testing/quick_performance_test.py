#!/usr/bin/env python3
"""
Quick Performance Test
Check NPS after removing expensive PV extraction
"""

import chess
import sys
import os
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from v7p3r import V7P3RCleanEngine

def quick_performance_test():
    """Quick test to check NPS"""
    
    print("⚡ QUICK PERFORMANCE TEST")
    print("=" * 40)
    
    engine = V7P3RCleanEngine()
    board = chess.Board()
    
    print("Testing 2-second search on starting position...")
    
    start_time = time.time()
    move = engine.search(board, 2.0)
    actual_time = time.time() - start_time
    
    nps = engine.nodes_searched / actual_time
    
    print(f"\nResults:")
    print(f"Time: {actual_time:.2f}s")
    print(f"Nodes: {engine.nodes_searched}")
    print(f"NPS: {nps:.0f}")
    
    if nps > 10000:
        print(f"✅ Performance looks good!")
    elif nps > 5000:
        print(f"⚠️  Performance OK but could be better")
    else:
        print(f"❌ Performance issue - too slow!")
    
    return nps

if __name__ == "__main__":
    nps = quick_performance_test()
    print(f"\nExpected: >10,000 NPS")
    print(f"Actual: {nps:.0f} NPS")
