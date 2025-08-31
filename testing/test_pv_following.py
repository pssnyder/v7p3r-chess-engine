#!/usr/bin/env python3
"""
Test PV Following Logic
Verify that the engine uses stored PV when opponent plays expected moves
"""

import chess
import sys
import os
import time

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3RCleanEngine

def test_pv_following():
    """Test PV following when opponent cooperates"""
    
    print("🎯 PV FOLLOWING TEST")
    print("=" * 40)
    
    engine = V7P3RCleanEngine()
    board = chess.Board()
    
    print("Step 1: Engine calculates initial PV")
    start_time = time.time()
    move1 = engine.search(board, 2.0)
    search_time_1 = time.time() - start_time
    
    print(f"First search took: {search_time_1:.2f}s")
    print(f"Best move: {move1}")
    print(f"Full PV: {[str(m) for m in engine.last_pv]}")
    
    if not engine.last_pv or len(engine.last_pv) < 2:
        print("❌ No PV stored, can't test following")
        return
    
    # Play the engine's move
    board.push(move1)
    engine.update_pv_after_move(move1)  # Update the PV
    print(f"\nStep 2: Engine plays {move1}")
    print(f"PV after engine move: {[str(m) for m in engine.last_pv]}")
    
    # Play the opponent's expected move (first move in remaining PV)
    if engine.last_pv and len(engine.last_pv) > 0:
        expected_response = engine.last_pv[0]
        if expected_response in board.legal_moves:
            board.push(expected_response)
            engine.update_pv_after_move(expected_response)  # Update PV after opponent move
            print(f"Step 3: Opponent cooperates and plays {expected_response}")
            print(f"Remaining PV: {[str(m) for m in engine.last_pv]}")
        
        # Now test if engine follows PV quickly
        print("\nStep 4: Engine should use PV following...")
        start_time = time.time()
        move2 = engine.search(board, 2.0)
        search_time_2 = time.time() - start_time
        
        print(f"Second search took: {search_time_2:.3f}s")
        
        if search_time_2 < 0.01:
            print(f"Speed improvement: INSTANT (>100x faster)")
            print("🚀 SUCCESS! PV following is working perfectly!")
        else:
            speed_improvement = search_time_1 / max(search_time_2, 0.001)
            print(f"Speed improvement: {speed_improvement:.1f}x faster")
            
            if search_time_2 < 0.1:
                print("🚀 SUCCESS! PV following is working!")
            elif search_time_2 < search_time_1 * 0.5:
                print("✅ GOOD! Significant speedup from PV following")
            else:
                print("⚠️  PV following may not be working optimally")
    else:
        print("❌ No remaining PV after engine move")

if __name__ == "__main__":
    test_pv_following()
