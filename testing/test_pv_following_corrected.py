#!/usr/bin/env python3
"""
Test PV Following Functionality - Proper Implementation
Tests that the engine follows PV when opponent plays predicted moves
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

def test_pv_following_proper():
    """Test proper PV following with a 5-move sequence"""
    print("🔍 Testing PV Following - Proper Game Sequence")
    print("=" * 70)
    
    engine = V7P3REngine()
    board = chess.Board()
    
    print("Step 1: Engine searches from starting position")
    print("-" * 50)
    
    # Let engine search and get a deep PV
    start_time = time.time()
    engine_move_1 = engine.search(board, time_limit=6.0)  # Longer search for deeper PV
    search_time_1 = time.time() - start_time
    
    original_pv = engine.principal_variation
    print(f"   Engine's first move: {engine_move_1}")
    print(f"   Search time: {search_time_1:.2f}s")
    print(f"   PV length: {len(original_pv)}")
    
    if len(original_pv) < 5:
        print(f"   ❌ PV too short ({len(original_pv)} moves). Need at least 5 moves for test.")
        print(f"   PV: {' '.join(str(m) for m in original_pv)}")
        return False
    
    print(f"   Complete PV: {' '.join(str(m) for m in original_pv)}")
    
    # The PV should be: [engine_move_1, opponent_move_1, engine_move_2, opponent_move_2, engine_move_3, ...]
    expected_opponent_move_1 = original_pv[1]  # What opponent should play
    expected_engine_move_2 = original_pv[2]    # What engine should play next if opponent follows PV
    
    print(f"   Expected sequence:")
    print(f"     1. Engine plays: {engine_move_1} (✓ about to happen)")
    print(f"     2. Opponent should play: {expected_opponent_move_1}")
    print(f"     3. Engine should quickly play: {expected_engine_move_2}")
    
    print(f"\nStep 2: Make engine's first move")
    print("-" * 50)
    board.push(engine_move_1)
    print(f"   Position after {engine_move_1}: {board.fen()}")
    
    print(f"\nStep 3: Opponent plays the PREDICTED move ({expected_opponent_move_1})")
    print("-" * 50)
    
    # Simulate opponent playing the expected move
    if expected_opponent_move_1 not in board.legal_moves:
        print(f"   ❌ Expected opponent move {expected_opponent_move_1} is not legal!")
        print(f"   Legal moves: {list(board.legal_moves)}")
        return False
    
    # Notify engine that opponent is about to play the expected move
    engine.notify_move_played(expected_opponent_move_1, board)
    board.push(expected_opponent_move_1)
    print(f"   Opponent played: {expected_opponent_move_1}")
    print(f"   Position after opponent move: {board.fen()}")
    
    print(f"\nStep 4: Engine should now INSTANTLY play {expected_engine_move_2}")
    print("-" * 50)
    
    # This is the critical test - engine should follow PV without full search
    start_time = time.time()
    engine_move_2 = engine.search(board, time_limit=3.0)
    search_time_2 = time.time() - start_time
    
    print(f"   Engine's second move: {engine_move_2}")
    print(f"   Time taken: {search_time_2:.3f}s")
    
    # Evaluate results
    success = True
    if engine_move_2 == expected_engine_move_2:
        print(f"   ✅ MOVE CORRECT: Engine played expected PV move!")
    else:
        print(f"   ❌ MOVE WRONG: Expected {expected_engine_move_2}, got {engine_move_2}")
        success = False
    
    if search_time_2 < 0.1:  # Should be nearly instant
        print(f"   ✅ SPEED PERFECT: Instant response ({search_time_2:.3f}s) - PV following worked!")
    elif search_time_2 < 0.5:
        print(f"   ⚠️  SPEED GOOD: Fast response ({search_time_2:.3f}s) - likely PV following")
    else:
        print(f"   ❌ SPEED SLOW: Took {search_time_2:.3f}s - probably did full search")
        success = False
    
    print(f"\nStep 5: Test PV breaking scenario")
    print("-" * 50)
    
    # Reset and test what happens when opponent breaks PV
    engine.new_game()
    board = chess.Board()
    
    # Repeat first move
    engine_move_1b = engine.search(board, time_limit=3.0)
    original_pv_b = engine.principal_variation
    
    if len(original_pv_b) >= 2:
        board.push(engine_move_1b)
        expected_opponent_move = original_pv_b[1]
        
        # Find a different legal move for opponent
        different_move = None
        for move in board.legal_moves:
            if move != expected_opponent_move:
                different_move = move
                break
        
        if different_move:
            print(f"   Expected opponent move: {expected_opponent_move}")
            print(f"   Opponent actually plays: {different_move}")
            
            # Notify engine of the unexpected move
            engine.notify_move_played(different_move, board)
            board.push(different_move)
            
            # Engine should now do a full search
            start_time = time.time()
            engine_move_break = engine.search(board, time_limit=3.0)
            search_time_break = time.time() - start_time
            
            print(f"   Engine response: {engine_move_break}")
            print(f"   Time taken: {search_time_break:.3f}s")
            
            if search_time_break > 1.0:
                print(f"   ✅ PV BREAK HANDLED: Full search performed ({search_time_break:.3f}s)")
            else:
                print(f"   ⚠️  Response was fast ({search_time_break:.3f}s) - might still be cached")
        else:
            print(f"   ⚠️  Could not find alternative move to test PV breaking")
    
    print(f"\n" + "=" * 70)
    if success:
        print("🎉 PV FOLLOWING TEST: SUCCESS!")
        print("   Engine correctly follows PV when opponent plays predicted moves")
    else:
        print("❌ PV FOLLOWING TEST: FAILED!")
        print("   Engine is not properly following the principal variation")
    print("=" * 70)
    
    return success

if __name__ == "__main__":
    test_pv_following_proper()
