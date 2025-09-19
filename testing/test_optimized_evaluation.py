#!/usr/bin/env python3
"""
Quick test of optimized evaluation selection
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
import time
from v7p3r import V7P3REngine

def test_optimized_evaluation():
    """Test optimized evaluation selection for speed"""
    print("TESTING OPTIMIZED EVALUATION SELECTION")
    print("=" * 50)
    
    engine = V7P3REngine()
    
    # Test complex middlegame position
    board = chess.Board("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")
    
    print("Testing complex middlegame position...")
    print("Position: r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")
    print()
    
    # Test depths 1-6 with time tracking
    for depth in range(1, 7):
        engine.default_depth = depth
        
        start_time = time.time()
        try:
            move = engine.search(board, time_limit=30.0)
            elapsed = time.time() - start_time
            nps = int(engine.nodes_searched / max(elapsed, 0.001))
            
            print(f"Depth {depth}: {elapsed:6.2f}s, {engine.nodes_searched:8d} nodes, {nps:6d} NPS")
            
            if elapsed > 25:  # Stop if getting too slow
                print("  -> Stopping due to time limit")
                break
                
        except Exception as e:
            print(f"Depth {depth}: ERROR - {e}")
            break
    
    print("\nTesting opening position for comparison...")
    board = chess.Board()
    
    for depth in range(1, 8):
        engine.default_depth = depth
        
        start_time = time.time()
        try:
            move = engine.search(board, time_limit=30.0)
            elapsed = time.time() - start_time
            nps = int(engine.nodes_searched / max(elapsed, 0.001))
            
            print(f"Depth {depth}: {elapsed:6.2f}s, {engine.nodes_searched:8d} nodes, {nps:6d} NPS")
            
            if elapsed > 25:
                print("  -> Stopping due to time limit")
                break
                
        except Exception as e:
            print(f"Depth {depth}: ERROR - {e}")
            break

if __name__ == "__main__":
    test_optimized_evaluation()