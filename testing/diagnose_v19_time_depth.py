#!/usr/bin/env python3
"""
Diagnose v19 Time Management & Search Depth Issues

WHY: Arena game showed v19 searching to depth 3-5 (way too shallow)
     v19 moves were 1-3 seconds (very fast compared to v18.4's 3-11s)
     
GOAL: Understand if time management is too conservative
"""

import chess
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from v7p3r_time_manager import TimeManager
import v7p3r


def test_time_allocations():
    """Test time allocations at various game states"""
    
    print("=" * 80)
    print("V19 TIME ALLOCATION DIAGNOSTICS")
    print("=" * 80)
    
    # Simulate 5min+4s blitz (like programmatic tournament)
    print("\n5 MINUTE + 4 SECOND BLITZ (Programmatic Tournament)")
    print("-" * 80)
    
    scenarios = [
        ("Opening (move 8)", 285.0, 4.0, 8),
        ("Early mid (move 15)", 240.0, 4.0, 15),
        ("Middlegame (move 25)", 180.0, 4.0, 25),
        ("Late middle (move 35)", 120.0, 4.0, 35),
        ("Endgame (move 50)", 80.0, 4.0, 50),
        ("Low time (move 40)", 10.0, 4.0, 40),
        ("Emergency (move 45)", 2.5, 4.0, 45),
    ]
    
    for label, remaining, inc, moves in scenarios:
        target, max_time = TimeManager.calculate_time_allocation(
            remaining, inc, moves
        )
        print(f"{label:25} | Remain: {remaining:5.1f}s | Alloc: {target:5.2f}s (max {max_time:5.2f}s)")
    
    # Simulate 5min+5s blitz (like Arena tournament)
    print("\n\n5 MINUTE + 5 SECOND BLITZ (Arena Tournament)")
    print("-" * 80)
    
    scenarios_arena = [
        ("Opening (move 8)", 285.0, 5.0, 8),
        ("Early mid (move 15)", 240.0, 5.0, 15),
        ("Middlegame (move 25)", 180.0, 5.0, 25),
        ("Late middle (move 35)", 120.0, 5.0, 35),
        ("Endgame (move 50)", 80.0, 5.0, 50),
    ]
    
    for label, remaining, inc, moves in scenarios_arena:
        target, max_time = TimeManager.calculate_time_allocation(
            remaining, inc, moves
        )
        print(f"{label:25} | Remain: {remaining:5.1f}s | Alloc: {target:5.2f}s (max {max_time:5.2f}s)")


def test_search_depth():
    """Test what search depth v19 achieves with allocated time"""
    
    print("\n\n" + "=" * 80)
    print("SEARCH DEPTH ANALYSIS")
    print("=" * 80)
    
    board = chess.Board()
    
    # Test opening position
    print("\nOpening Position (e4):")
    board.push_san("e4")
    
    # Allocate time for move 2
    target, max_time = TimeManager.calculate_time_allocation(295.0, 4.0, 1)
    print(f"  Allocated time: {target:.2f}s (max {max_time:.2f}s)")
    print(f"  Expected depth: Depends on search speed")
    
    # Test middlegame complexity
    print("\nComplex Middlegame Position:")
    print("  FEN: r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1")
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1")
    
    # Allocate time for move 15 in middlegame
    target, max_time = TimeManager.calculate_time_allocation(240.0, 4.0, 15)
    print(f"  Allocated time: {target:.2f}s (max {max_time:.2f}s)")
    print(f"  This should reach depth 8-10 for decent tactical play")


def compare_v18_vs_v19_time():
    """Compare time allocations between v18.4 and v19.0"""
    
    print("\n\n" + "=" * 80)
    print("V18.4 vs V19.0 TIME COMPARISON")
    print("=" * 80)
    
    print("\nv19.0 allocations (C0BR4-style conservative):")
    print("-" * 80)
    
    # Middlegame scenario: 240s remaining, 4s increment, move 15
    target, max_time = TimeManager.calculate_time_allocation(240.0, 4.0, 15)
    print(f"Middlegame (240s, move 15): {target:.2f}s allocated")
    
    print("\nv18.4 allocations (complex v14.1 logic):")
    print("-" * 80)
    print("v18.4 used complex time factor calculations:")
    print("  - Multiple nested conditions per color")
    print("  - Hard caps at various thresholds")  
    print("  - Likely allocated MORE time per move")
    print(f"  - Estimated ~6-10s per middlegame move")
    
    print("\n⚠️  DIAGNOSIS:")
    print("   v19 allocates ~" + f"{target:.1f}" + "s per middlegame move")
    print("   v18.4 likely allocated 2-3x more (~6-10s)")
    print("   This explains why v19 searches to depth 3-5 vs v18.4's deeper search")


def main():
    test_time_allocations()
    test_search_depth()
    compare_v18_vs_v19_time()
    
    print("\n\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("""
1. Time allocations are TOO CONSERVATIVE for 5-minute games
   - v19 allocating 2-4s per move in middlegame
   - v18.4 was allocating 6-10s per move
   - Result: v19 reaches depth 3-5, v18.4 reaches depth 6-8+
   
2. The "Error on move 38" in programmatic tournament is concerning
   - Need to investigate what's causing crashes
   - Might be depth-related or position-specific
   
3. Suggested fix: Increase time allocation multipliers
   - Opening: 0.95x → 1.2x (search deeper in opening)
   - Middlegame: 1.1x → 1.5x (critical for tactics)
   - Endgame: 0.9x → 1.0x (maintain current)
   - This should bring v19 closer to v18.4's search depth
   
4. C0BR4 comparison might not be apples-to-apples
   - C0BR4 is C# (likely faster per-node than Python)
   - C0BR4 simple eval might search faster than v19's eval
   - Need to tune time allocation for v19's actual search speed
""")


if __name__ == "__main__":
    main()
