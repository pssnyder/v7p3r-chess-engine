#!/usr/bin/env python3
"""
Time Management Analysis Tool
Diagnose why V7P3R is only reaching depth 2-3 instead of target depth 6+
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r_time_manager import V7P3RTimeManager

def analyze_time_management():
    """Analyze current time management behavior"""
    
    print("V7P3R Time Management Analysis")
    print("=" * 50)
    
    # Test different time controls
    time_controls = [
        (30, 0, "30 seconds per game"),
        (120, 1, "2:1 (2 min + 1 sec increment)"),
        (300, 3, "5:3 (5 min + 3 sec increment)"),
        (600, 0, "10 minutes per game"),
        (1800, 0, "30 minutes per game (tournament)"),
    ]
    
    test_positions = [
        ("Starting position", chess.Board()),
        ("Complex middlegame", chess.Board("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")),
        ("Simple endgame", chess.Board("8/8/8/4k3/4K3/8/8/8 w - - 0 1")),
    ]
    
    for time_control, increment, description in time_controls:
        print(f"\n{description}")
        print("-" * 40)
        
        time_manager = V7P3RTimeManager(base_time=time_control, increment=increment)
        
        for move_number in [1, 10, 20, 30]:
            print(f"\n  Move {move_number}:")
            time_manager.update_time_info(0, move_number)
            
            for pos_name, board in test_positions:
                time_remaining = time_control * 0.8  # Assume 80% time remaining
                allocated_time, target_depth = time_manager.calculate_time_allocation(board, time_remaining)
                
                # Calculate what percentage of total time this represents
                percentage = (allocated_time / time_control) * 100
                
                print(f"    {pos_name:15}: {allocated_time:6.2f}s (depth {target_depth}, {percentage:4.1f}% of total)")
    
    print(f"\n\nTarget Goals:")
    print("- Depth 6+ for routine positions")
    print("- Depth 10+ for 30-minute games")
    print("- Complete games with >60s remaining")
    print("- No timeouts")

def test_search_depth_vs_time():
    """Test how much time different depths actually take"""
    
    print("\n" + "=" * 50)
    print("SEARCH DEPTH vs TIME ANALYSIS")
    print("=" * 50)
    
    # Import the engine
    from v7p3r import V7P3REngine
    
    engine = V7P3REngine()
    test_board = chess.Board()
    
    print("Testing actual search times for different depths...")
    print("Position: Starting position")
    print()
    
    # Test different fixed depths
    for test_depth in range(1, 8):
        import time
        start_time = time.time()
        
        # Temporarily set default depth
        original_depth = engine.default_depth
        engine.default_depth = test_depth
        
        try:
            # Search with a very long time limit to not be time-constrained
            move = engine.search(test_board, time_limit=30.0)
            elapsed = time.time() - start_time
            
            print(f"Depth {test_depth}: {elapsed:6.2f}s, {engine.nodes_searched:8d} nodes, {int(engine.nodes_searched/max(elapsed, 0.001)):6d} NPS")
            
        except Exception as e:
            print(f"Depth {test_depth}: ERROR - {e}")
        finally:
            engine.default_depth = original_depth
    
    print("\nAnalysis:")
    print("- If depth 6 takes <5 seconds, time management is too conservative")
    print("- If depth 6 takes >30 seconds, we need performance optimization")

def analyze_time_allocation_logic():
    """Analyze the specific time allocation calculations"""
    
    print("\n" + "=" * 50)
    print("TIME ALLOCATION LOGIC ANALYSIS")
    print("=" * 50)
    
    # Test 30-minute game scenario
    time_manager = V7P3RTimeManager(base_time=1800, increment=0)
    board = chess.Board()
    
    print("30-minute game analysis:")
    print()
    
    for move_num in [1, 10, 20, 30, 35]:
        time_remaining = 1800 - (move_num * 20)  # Simulate 20s per move used
        if time_remaining <= 0:
            time_remaining = 300  # Emergency time
        
        time_manager.update_time_info(1800 - time_remaining, move_num)
        allocated_time, target_depth = time_manager.calculate_time_allocation(board, time_remaining)
        
        # Calculate the internal values
        estimated_moves_left = max(10, 40 - move_num)
        base_allocation = time_remaining / estimated_moves_left
        
        print(f"Move {move_num:2d}: {time_remaining:4.0f}s remaining")
        print(f"         Estimated moves left: {estimated_moves_left}")
        print(f"         Base allocation: {base_allocation:.2f}s")
        print(f"         Final allocation: {allocated_time:.2f}s")
        print(f"         Target depth: {target_depth}")
        print()

if __name__ == "__main__":
    analyze_time_management()
    test_search_depth_vs_time()
    analyze_time_allocation_logic()