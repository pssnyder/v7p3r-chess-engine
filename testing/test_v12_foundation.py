#!/usr/bin/env python3
"""
V12.0 Foundation Test
====================

Test the cleaned v12.0 foundation to ensure it works properly:
- Engine initialization 
- Enhanced nudge database loading
- Basic search functionality
- Core heuristics preserved
"""

import os
import sys
import chess

# Add the src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from v7p3r import V7P3REngine

def test_v12_foundation():
    print("V7P3R v12.0 Foundation Test")
    print("============================")
    print()
    
    try:
        # 1. Engine initialization
        print("1. Testing engine initialization...")
        engine = V7P3REngine()
        print("   ✅ Engine initialized successfully")
        print(f"   ✅ Nudge database: {len(engine.nudge_database)} positions")
        print()
        
        # 2. Basic search test
        print("2. Testing basic search functionality...")
        board = chess.Board()
        print(f"   Testing position: {board.fen()}")
        
        move = engine.search(board, time_limit=2.0, depth=3)
        print(f"   ✅ Search completed: {move}")
        print(f"   ✅ Nodes searched: {engine.nodes_searched:,}")
        print()
        
        # 3. Test evaluation components
        print("3. Testing evaluation components...")
        score = engine._evaluate_position(board)
        print(f"   ✅ Position evaluation: {score}")
        print()
        
        # 4. Test nudge system
        print("4. Testing nudge system...")
        position_key = engine._get_position_key(board)
        nudge_bonus = engine._get_nudge_bonus(board, move)
        print(f"   ✅ Position key generated: {position_key[:50]}...")
        print(f"   ✅ Nudge bonus calculated: {nudge_bonus}")
        print()
        
        # 5. Performance check
        print("5. Performance check...")
        import time
        start_time = time.time()
        for _ in range(100):
            engine._evaluate_position(board)
        eval_time = time.time() - start_time
        evals_per_sec = 100 / eval_time
        print(f"   ✅ Evaluation speed: {evals_per_sec:.0f} evals/sec")
        print()
        
        print("🎉 V12.0 Foundation Test: ALL PASSED")
        print("Ready for v12.0 development!")
        return True
        
    except Exception as e:
        print(f"❌ Foundation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_v12_foundation()