#!/usr/bin/env python3
"""
Showcase UCI Output - Test various search and PV following scenarios
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

def showcase_uci_output():
    """Demonstrate clean UCI output in different scenarios"""
    print("🎭 V7P3R Chess Engine - UCI Output Showcase")
    print("=" * 70)
    
    engine = V7P3REngine()
    board = chess.Board()
    
    print("\n📊 SCENARIO 1: Normal Search from Starting Position")
    print("-" * 50)
    print("Position: rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
    print("\nUCI Output:")
    print("-" * 30)
    
    move1 = engine.search(board, time_limit=3.0)
    pv = engine.principal_variation
    
    print(f"bestmove {move1}")
    print(f"\n✅ Result: Engine chose {move1}")
    print(f"   PV Length: {len(pv)} moves")
    print(f"   Full PV: {' '.join(str(m) for m in pv)}")
    
    # Set up for PV following test
    if len(pv) >= 3:
        predicted_opponent_move = pv[1]
        predicted_our_response = pv[2]
        
        print(f"\n📊 SCENARIO 2: PV Following (Opponent plays predicted {predicted_opponent_move})")
        print("-" * 50)
        
        # Make our first move
        board.push(move1)
        print(f"Position after {move1}: {board.fen()}")
        
        # Opponent plays predicted move
        board.push(predicted_opponent_move)
        print(f"Position after {predicted_opponent_move}: {board.fen()}")
        
        print("\nUCI Output:")
        print("-" * 30)
        
        start_time = time.time()
        move2 = engine.search(board, time_limit=3.0)
        search_time = time.time() - start_time
        
        print(f"bestmove {move2}")
        print(f"\n✅ Result: Engine chose {move2} in {search_time:.3f}s")
        
        if move2 == predicted_our_response and search_time < 0.1:
            print(f"   🚀 PV FOLLOWING SUCCESS! Instant response")
        elif move2 == predicted_our_response:
            print(f"   ✅ Correct move, but took {search_time:.3f}s")
        else:
            print(f"   ⚠️  Different move chosen: expected {predicted_our_response}")
    
    print(f"\n📊 SCENARIO 3: PV Break (Opponent plays unexpected move)")
    print("-" * 50)
    
    # Reset and test PV break
    engine.new_game()
    board = chess.Board()
    
    # Get a fresh PV
    move1b = engine.search(board, time_limit=3.0)
    pv = engine.principal_variation
    
    if len(pv) >= 2:
        board.push(move1b)
        expected_move = pv[1]
        
        # Find a different legal move
        different_move = None
        for move in board.legal_moves:
            if move != expected_move:
                different_move = move
                break
        
        if different_move:
            print(f"Expected: {expected_move}, Playing: {different_move}")
            board.push(different_move)
            print(f"Position after unexpected {different_move}: {board.fen()}")
            
            print("\nUCI Output:")
            print("-" * 30)
            
            start_time = time.time()
            move3 = engine.search(board, time_limit=3.0)
            search_time = time.time() - start_time
            
            print(f"bestmove {move3}")
            print(f"\n✅ Result: Engine chose {move3} in {search_time:.3f}s")
            print(f"   🔍 Full search performed after PV break")
    
    print(f"\n📊 SCENARIO 4: Quick Search (1 second limit)")
    print("-" * 50)
    
    engine.new_game()
    board = chess.Board()
    print("Position: Starting position")
    
    print("\nUCI Output:")
    print("-" * 30)
    
    start_time = time.time()
    move4 = engine.search(board, time_limit=1.0)
    search_time = time.time() - start_time
    
    print(f"bestmove {move4}")
    print(f"\n✅ Result: Engine chose {move4} in {search_time:.3f}s")
    print(f"   ⚡ Quick search completed")
    
    print(f"\n" + "=" * 70)
    print("🎉 UCI OUTPUT SHOWCASE COMPLETE!")
    print("   Clean, professional output with essential information only")
    print("   PV following displays standard UCI depth/pv format")
    print("   No debug clutter in production output")
    print("=" * 70)

if __name__ == "__main__":
    showcase_uci_output()
