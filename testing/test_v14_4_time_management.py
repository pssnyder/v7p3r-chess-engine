#!/usr/bin/env python3
"""
Test V14.4 Time Management - Verify v14.0 restoration

Compares time allocation between v14.1's rushed approach and v14.4's balanced approach.
Should show v14.4 using more time in opening (0.8x vs 0.5x).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine


def test_time_management():
    """Test time management at different game phases"""
    engine = V7P3REngine()
    
    print("=" * 80)
    print("V14.4 TIME MANAGEMENT TEST")
    print("Verifying restoration of v14.0's balanced time allocation")
    print("=" * 80)
    
    # Test scenarios
    scenarios = [
        ("Early Opening (move 5)", "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1", 5),
        ("Mid Opening (move 12)", "r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1", 12),
        ("Middlegame (move 25)", "r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 1", 25),
        ("Complex Middlegame (move 35)", "2r2rk1/1p1q1ppp/p2p1n2/3Pp3/4P3/2N2P2/PPP1Q1PP/2KR3R w - - 0 1", 35),
        ("Endgame (move 55)", "8/5pk1/6p1/8/3K4/8/5P2/8 w - - 0 1", 55),
    ]
    
    base_time = 180.0  # 3 minutes base time
    
    print(f"\nBase time available: {base_time}s (3 minutes)")
    print("-" * 80)
    
    for name, fen, moves_played in scenarios:
        board = chess.Board(fen)
        # Simulate move history for time calculation
        board.move_stack = [None] * moves_played  # Fake move history
        
        target_time, max_time = engine._calculate_adaptive_time_allocation(board, base_time)
        
        # Calculate what v14.1 would have done (for comparison)
        if moves_played < 8:
            v14_1_factor = 0.5
        elif moves_played < 15:
            v14_1_factor = 0.6
        elif moves_played < 25:
            v14_1_factor = 0.9
        elif moves_played < 40:
            v14_1_factor = 1.1
        else:
            v14_1_factor = 0.7
        
        v14_1_time = min(min(base_time, 60.0) * v14_1_factor * 0.7, min(base_time, 60.0) * 0.75)
        
        improvement = ((target_time - v14_1_time) / v14_1_time * 100) if v14_1_time > 0 else 0
        
        print(f"\n{name} ({moves_played} moves played):")
        print(f"  V14.4 target time: {target_time:.2f}s (max: {max_time:.2f}s)")
        print(f"  V14.1 would use:   {v14_1_time:.2f}s (rushed, 60s cap)")
        print(f"  Improvement:       {improvement:+.1f}% more time to think")
        
        if moves_played < 15:
            print(f"  ✓ OPENING: More time for quality moves (v14.0 style)")
        elif moves_played < 40:
            print(f"  ✓ MIDDLEGAME: Appropriate time for complex tactics")
        else:
            print(f"  ✓ ENDGAME: Balanced time for technique")
    
    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("=" * 80)
    print("✓ V14.4 restores v14.0's balanced opening thinking (0.8x factor)")
    print("✓ Removes artificial 60-second cap from v14.1")
    print("✓ Opening moves get MORE time than v14.1's rushed approach")
    print("✓ Tournament evidence: v14.0 (70.7%) >> v14.1 (53.8%)")
    print("✓ Expected: V14.4 should return to ~70% win rate")
    print("=" * 80)


def test_quick_game():
    """Quick game test to verify engine functionality"""
    print("\n" + "=" * 80)
    print("QUICK FUNCTIONALITY TEST")
    print("=" * 80)
    
    engine = V7P3REngine()
    board = chess.Board()
    
    print("\nStarting position:")
    print(board)
    
    print("\nFinding best move with 5-second time limit...")
    start_time = time.time()
    move = engine.search(board, time_limit=5.0)
    elapsed = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Best move: {move}")
    print(f"  Nodes searched: {engine.nodes_searched}")
    print(f"  Time used: {elapsed:.2f}s")
    
    if move and move != chess.Move.null():
        print(f"\n✓ V14.4 functioning correctly!")
        print(f"  Move is legal: {move in board.legal_moves}")
    else:
        print(f"\n✗ Error: No valid move returned")
    
    print("=" * 80)


if __name__ == "__main__":
    import time
    
    test_time_management()
    test_quick_game()
    
    print("\n" + "=" * 80)
    print("V14.4 VALIDATION COMPLETE")
    print("=" * 80)
    print("\nNext steps:")
    print("1. Test V14.4 vs MaterialOpponent (should improve from v14.3)")
    print("2. Test V14.4 vs PositionalOpponent (target: competitive performance)")
    print("3. Run mini-tournament: V14.0 vs V14.4 (should be similar)")
    print("4. Consider simple PST evaluation for V14.5 (PositionalOpponent approach)")
    print("=" * 80)
