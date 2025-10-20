#!/usr/bin/env python3
"""
TAL-BOT PV Following Test

Test the Principal Variation following system for instant moves in bullet/blitz.
"""

import time
import chess
import sys
import os

# Add the src directory to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from vpr import VPREngine


def test_pv_collection():
    """Test PV collection and display"""
    print("=== PV Collection Test ===\n")
    
    engine = VPREngine()
    board = chess.Board()
    
    print("Testing PV collection on starting position...")
    best_move = engine.search(board, time_limit=2.0)
    
    print(f"\nResults:")
    print(f"Best move: {best_move}")
    print(f"Principal Variation: {[str(move) for move in engine.principal_variation[:5]]}")
    print(f"PV length: {len(engine.principal_variation)}")
    print(f"Last search PV: {[str(move) for move in engine.last_search_pv[:5]]}")
    
    if len(engine.principal_variation) >= 2:
        print("✓ PV collection working - multiple moves collected")
    else:
        print("✗ PV collection issue - only single move")
    

def test_pv_following():
    """Test PV following for instant moves"""
    print("\n=== PV Following Test ===\n")
    
    engine = VPREngine()
    board = chess.Board()
    
    # First search to establish PV
    print("Step 1: Initial search to establish PV...")
    first_move = engine.search(board, time_limit=2.0)
    initial_pv = engine.last_search_pv[:]
    
    print(f"Initial best move: {first_move}")
    print(f"Initial PV: {[str(move) for move in initial_pv[:3]]}")
    
    if len(initial_pv) >= 2:
        # Play the first move
        board.push(first_move)
        print(f"\nStep 2: Played {first_move}, checking for PV following...")
        
        # Simulate opponent playing random legal move
        opponent_moves = list(board.legal_moves)
        if opponent_moves:
            opponent_move = opponent_moves[0]
            board.push(opponent_move)
            print(f"Opponent played: {opponent_move}")
            
            # Now check if we can instantly follow PV
            start_time = time.time()
            instant_move = engine._check_pv_following(board)
            follow_time = time.time() - start_time
            
            print(f"PV following check time: {follow_time*1000:.1f}ms")
            
            if instant_move:
                print(f"✓ Instant PV move found: {instant_move}")
                print(f"⚡ BLITZ MODE: No search needed!")
            else:
                print("No PV following possible (position diverged)")
                
                # Try regular search to compare
                start_time = time.time()
                regular_move = engine.search(board, time_limit=1.0)
                search_time = time.time() - start_time
                
                print(f"Regular search time: {search_time*1000:.1f}ms")
                print(f"Time saved by PV following: {(search_time - follow_time)*1000:.1f}ms")
    

def test_bullet_simulation():
    """Simulate bullet game scenario where PV following is crucial"""
    print("\n=== Bullet Game Simulation ===\n")
    
    engine = VPREngine()
    board = chess.Board()
    
    move_times = []
    pv_follows = 0
    total_moves = 0
    
    print("Simulating 5 moves with bullet time pressure...")
    
    for move_num in range(1, 6):
        print(f"\nMove {move_num}:")
        
        start_time = time.time()
        
        # Check for instant PV move first
        instant_move = engine._check_pv_following(board)
        
        if instant_move and instant_move in board.legal_moves:
            move = instant_move
            move_time = time.time() - start_time
            pv_follows += 1
            print(f"  ⚡ INSTANT PV follow: {move} ({move_time*1000:.1f}ms)")
        else:
            # Regular search with bullet time limit
            move = engine.search(board, time_limit=0.5)  # 500ms bullet time
            move_time = time.time() - start_time
            print(f"  🔍 Regular search: {move} ({move_time*1000:.1f}ms)")
        
        move_times.append(move_time)
        total_moves += 1
        
        if move != chess.Move.null():
            board.push(move)
        else:
            break
    
    print(f"\nBullet Performance Summary:")
    print(f"Total moves: {total_moves}")
    print(f"PV follows: {pv_follows}")
    print(f"PV follow rate: {pv_follows/total_moves*100:.1f}%")
    print(f"Average move time: {sum(move_times)/len(move_times)*1000:.1f}ms")
    print(f"Fastest move: {min(move_times)*1000:.1f}ms")
    print(f"Slowest move: {max(move_times)*1000:.1f}ms")
    
    if pv_follows > 0:
        print("✓ PV Following active - bullet advantage confirmed!")
    else:
        print("ℹ  No PV follows in this sequence (normal in short test)")


if __name__ == "__main__":
    print("TAL-BOT PV System Test")
    print("=" * 50)
    
    test_pv_collection()
    test_pv_following()
    test_bullet_simulation()
    
    print("\n🔥 TAL-BOT PV system ready for bullet domination!")
    print("The entropy engine now has instant-move capability! ⚡")