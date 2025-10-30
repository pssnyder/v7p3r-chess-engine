#!/usr/bin/env python3
"""
Debug script to see what time allocations V14.5 is actually using
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine
import chess

def test_time_allocations():
    """Test what time allocations are being calculated"""
    engine = V7P3REngine()
    board = chess.Board()
    
    # Test scenarios matching our test
    scenarios = [
        ("Opening move 1", 0, 300.0, 5.0),
        ("Move 5", 4, 285.0, 5.0),
        ("Move 10 middlegame", 10, 270.0, 5.0),
    ]
    
    print("V14.5 Time Allocation Debug")
    print("=" * 70)
    
    for name, moves, remaining, inc in scenarios:
        print(f"\n{name}: {moves} moves played, {remaining}s remaining, {inc}s increment")
        print("-" * 70)
        
        # Simulate UCI time calculation (from v7p3r_uci.py logic)
        if moves < 8:
            time_factor = 80.0
            uci_time_limit = (remaining / time_factor) + (inc * 0.8)
        elif moves < 15:
            time_factor = 50.0
            uci_time_limit = (remaining / time_factor) + inc
        elif moves < 30:
            time_factor = 30.0
            uci_time_limit = (remaining / time_factor) + inc
        else:
            time_factor = 20.0
            uci_time_limit = (remaining / time_factor) + inc
        
        print(f"  UCI calculated time limit: {uci_time_limit:.2f}s")
        
        # Engine's emergency allocation
        target, max_time = engine._calculate_emergency_time_allocation(uci_time_limit, moves)
        print(f"  Engine target_time: {target:.2f}s")
        print(f"  Engine max_time: {max_time:.2f}s")
        print(f"  Emergency stop at: {uci_time_limit * 0.85:.2f}s")
        
        # Minimum depth
        min_depth = engine._calculate_minimum_depth(uci_time_limit)
        print(f"  Minimum depth: {min_depth}")

if __name__ == "__main__":
    test_time_allocations()
