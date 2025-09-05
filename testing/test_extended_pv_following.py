#!/usr/bin/env python3
"""
Extended PV Following Test - Multi-move sequences
"""

import os
import sys
import time
import chess

# Add the src directory to the path so we can import the engine
src_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src')
sys.path.insert(0, src_dir)

import v7p3r
V7P3REngine = v7p3r.V7P3REngine

def test_extended_pv_following():
    """Test PV following for multiple consecutive moves"""
    print("🔍 Testing Extended PV Following - Multiple Move Sequence")
    print("=" * 70)
    
    engine = V7P3REngine()
    board = chess.Board()
    
    print("Step 1: Engine searches and establishes deep PV")
    print("-" * 50)
    
    engine_move_1 = engine.search(board, time_limit=5.0)
    original_pv = engine.principal_variation
    
    if len(original_pv) < 6:
        print(f"PV too short for extended test: {len(original_pv)} moves")
        return False
    
    print(f"Original PV ({len(original_pv)} moves): {' '.join(str(m) for m in original_pv)}")
    
    # Play first 6 moves as predicted
    moves_to_test = original_pv[:6]  # Test first 6 moves
    total_instant_moves = 0
    
    for i, move in enumerate(moves_to_test):
        print(f"\nStep {i+2}: Playing move {i+1}: {move}")
        print("-" * 30)
        
        if i == 0:
            # First move - engine should play it normally
            board.push(move)
            print(f"✓ Engine's first move: {move}")
        elif i % 2 == 1:
            # Odd moves - opponent moves (we simulate)
            board.push(move)
            print(f"✓ Opponent plays predicted move: {move}")
        else:
            # Even moves - engine should follow PV instantly
            start_time = time.time()
            engine_move = engine.search(board, time_limit=3.0)
            search_time = time.time() - start_time
            
            print(f"Engine played: {engine_move} in {search_time:.3f}s")
            
            if engine_move == move:
                print(f"✅ CORRECT: Engine played expected PV move")
                if search_time < 0.05:
                    print(f"✅ INSTANT: Response time {search_time:.3f}s")
                    total_instant_moves += 1
                else:
                    print(f"⚠️  SLOW: Response time {search_time:.3f}s")
            else:
                print(f"❌ WRONG: Expected {move}, got {engine_move}")
                break
                
            board.push(engine_move)
    
    print(f"\n" + "=" * 70)
    print(f"EXTENDED PV FOLLOWING RESULTS:")
    print(f"Instant moves played: {total_instant_moves} out of {len(moves_to_test)//2}")
    
    if total_instant_moves >= 2:
        print("🎉 SUCCESS: Engine demonstrated extended PV following!")
        return True
    else:
        print("❌ FAILED: Engine did not consistently follow PV")
        return False

if __name__ == "__main__":
    test_extended_pv_following()
