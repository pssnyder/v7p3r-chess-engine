#!/usr/bin/env python3
"""
Quick test of enhanced dynamic move selector
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
import time
from v7p3r import V7P3REngine

def test_enhanced_selector():
    """Quick test of enhanced move selector"""
    print("TESTING ENHANCED DYNAMIC MOVE SELECTOR")
    print("=" * 50)
    
    engine = V7P3REngine()
    
    # Test the problematic middlegame position
    board = chess.Board("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")
    
    print("Complex middlegame position:")
    print("Testing depth 5 performance...")
    
    engine.default_depth = 5
    start_time = time.time()
    
    try:
        move = engine.search(board, time_limit=20.0)
        elapsed = time.time() - start_time
        nodes = engine.nodes_searched
        nps = int(nodes / max(elapsed, 0.001))
        
        print(f"  Search time: {elapsed:6.2f}s")
        print(f"  Nodes: {nodes:8,}")
        print(f"  NPS: {nps:8,}")
        print(f"  Best move: {move}")
        
        # Compare to previous performance (was ~49s)
        improvement = "✅ IMPROVED" if elapsed < 30 else "❌ STILL SLOW"
        print(f"  Performance: {improvement}")
        
    except Exception as e:
        print(f"  ERROR: {e}")

if __name__ == "__main__":
    test_enhanced_selector()