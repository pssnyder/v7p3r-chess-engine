#!/usr/bin/env python3
"""
Test the enhanced LMR implementation and time manager integration
"""
import sys
import os
import chess
import time

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from v7p3r import V7P3REngine
from v7p3r_time_manager import V7P3RTimeManager

def test_v11_phase1_enhancements():
    """Test V11 Phase 1 enhancements: Time Management + LMR"""
    
    print("V7P3R v11 Phase 1 Enhancement Test")
    print("=" * 50)
    
    # Initialize engine and time manager
    engine = V7P3REngine()
    time_manager = V7P3RTimeManager(base_time=180.0, increment=2.0)  # 3+2 time control
    
    # Test positions
    test_positions = [
        ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1", "Starting Position"),
        ("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1", "Complex Kiwipete"),
        ("8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1", "Tactical Position")
    ]
    
    print("\nTesting Time Management Integration:")
    print("-" * 40)
    
    for i, (fen, name) in enumerate(test_positions):
        board = chess.Board(fen)
        
        # Get time allocation
        time_remaining = 120.0  # 2 minutes remaining
        allocated_time, target_depth = time_manager.calculate_time_allocation(board, time_remaining)
        
        print(f"\n{i+1}. {name}")
        print(f"   Allocated Time: {allocated_time:.2f}s")
        print(f"   Target Depth: {target_depth}")
        
        # Test search with time limit
        start_time = time.time()
        try:
            best_move = engine.search(board, allocated_time)
            search_time = time.time() - start_time
            
            print(f"   Search Time: {search_time:.2f}s")
            print(f"   Best Move: {best_move}")
            print(f"   Nodes Searched: {engine.nodes_searched:,}")
            
            if search_time > 0:
                nps = engine.nodes_searched / search_time
                print(f"   NPS: {nps:,.0f}")
            
            # Check if time allocation was respected
            time_efficiency = search_time / allocated_time
            if time_efficiency <= 1.1:  # Within 10% of allocation
                print("   ✅ Time management: GOOD")
            else:
                print(f"   ⚠️ Time management: OVER by {(time_efficiency-1)*100:.1f}%")
        
        except Exception as e:
            print(f"   ❌ Search failed: {e}")
    
    # Test LMR statistics
    print(f"\nTime Manager Statistics:")
    stats = time_manager.get_statistics()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n🎯 V11 Phase 1 Enhancement Test Complete!")
    
    return True

if __name__ == "__main__":
    test_v11_phase1_enhancements()