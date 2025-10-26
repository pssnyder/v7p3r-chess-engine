#!/usr/bin/env python3

"""
Simple time test to debug V14.3 time management issues
"""

import chess
import time
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine

def simple_time_test():
    """Test basic time management with simple positions"""
    
    engine = V7P3REngine()
    
    # Test 1: Simple opening position with 1.0s limit
    print("=== Simple Time Test 1: 1.0s limit ===")
    board = chess.Board()
    
    start_time = time.time()
    try:
        move = engine.search(board, time_limit=1.0)
        actual_time = time.time() - start_time
        print(f"Move: {move}")
        print(f"Actual time: {actual_time:.3f}s ({actual_time*100:.1f}% of limit)")
        
        if actual_time > 1.0:
            print(f"❌ FAILED: Exceeded time limit by {actual_time-1.0:.3f}s")
        else:
            print(f"✅ PASSED: Within time limit")
            
    except Exception as e:
        actual_time = time.time() - start_time
        print(f"❌ EXCEPTION after {actual_time:.3f}s: {e}")
    
    print()
    
    # Test 2: Even shorter time limit
    print("=== Simple Time Test 2: 0.5s limit ===")
    
    start_time = time.time()
    try:
        move = engine.search(board, time_limit=0.5)
        actual_time = time.time() - start_time
        print(f"Move: {move}")
        print(f"Actual time: {actual_time:.3f}s ({actual_time*200:.1f}% of limit)")
        
        if actual_time > 0.5:
            print(f"❌ FAILED: Exceeded time limit by {actual_time-0.5:.3f}s")
        else:
            print(f"✅ PASSED: Within time limit")
            
    except Exception as e:
        actual_time = time.time() - start_time
        print(f"❌ EXCEPTION after {actual_time:.3f}s: {e}")

if __name__ == "__main__":
    simple_time_test()