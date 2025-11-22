#!/usr/bin/env python3
"""
Test V17.0 PV Following Blunder

This test recreates the exact game sequence from Loss #1 to determine if PV-following
(instant move without search) caused the critical f6 blunder.

Test Strategy:
1. Play moves leading up to the critical position (moves 1-9)
2. Let v17.0 search normally and build PV
3. Check if f6 appears in the PV
4. Compare PV-following move vs fresh search at critical position

Expected Findings:
- If PV contains f6 and engine plays it instantly (depth 1, 0 nodes): PV blunder confirmed
- If fresh search at position avoids f6: Confirms PV-following is harmful
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine
import time

def print_section(title):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)

def play_game_sequence_with_pv_tracking():
    """Play the losing game sequence, tracking PV at each step"""
    
    print_section("V17.0 PV BLUNDER ANALYSIS - LOSS #1 RECREATION")
    
    # The losing game moves
    game_moves = [
        ("e2e3", "White's 1.e3"),
        ("b8c6", "Black's 1...Nc6"),
        ("g1f3", "White's 2.Nf3"),
        ("g8f6", "Black's 2...Nf6"),
        ("b1c3", "White's 3.Nc3"),
        ("d7d5", "Black's 3...d5"),
        ("f1b5", "White's 4.Bb5"),
        ("a7a6", "Black's 4...a6"),
        ("b5c6", "White's 5.Bxc6+"),
        ("b7c6", "Black's 5...bxc6"),
        ("f3e5", "White's 6.Ne5"),
        ("d8d6", "Black's 6...Qd6"),
        ("d2d4", "White's 7.d4"),
        ("f6e4", "Black's 7...Ne4"),
        ("c3e4", "White's 8.Nxe4"),
        ("d5e4", "Black's 8...dxe4"),
        ("d1h5", "White's 9.Qh5"),
        ("g7g6", "Black's 9...g6 (v17.0 search, builds PV)"),
    ]
    
    board = chess.Board()
    engine = V7P3REngine()
    
    # Track if we're at Black's moves (v17.0's perspective)
    move_number = 1
    is_black_move = False
    
    print("\nPlaying through game sequence...")
    print("Note: We're tracking v17.0 (Black's) PV after each of its moves\n")
    
    for move_uci, description in game_moves:
        move = chess.Move.from_uci(move_uci)
        
        # Before making the move, check if this is Black's turn (v17.0)
        if board.turn == chess.BLACK:
            print(f"\n{'-' * 80}")
            print(f"MOVE {move_number}b: {description}")
            print(f"Position before v17.0's move:")
            print(board)
            print()
            
            # Let v17.0 search this position (5 seconds to build good PV)
            print("v17.0 searching (5 seconds)...")
            start_time = time.time()
            best_move = engine.search(board, 5.0)
            search_time = time.time() - start_time
            
            print(f"v17.0 selected: {best_move} (expected: {move_uci})")
            print(f"Search time: {search_time:.2f}s")
            
            # Check PV tracker state
            if engine.pv_tracker.following_pv:
                print(f"\n[!] PV FOLLOWING ACTIVE:")
                print(f"  - Predicted position FEN: {engine.pv_tracker.predicted_position_fen[:50]}...")
                print(f"  - Next planned move: {engine.pv_tracker.next_our_move}")
                print(f"  - Remaining PV: {engine.pv_tracker.pv_display_string}")
            else:
                print("  - PV following: NOT ACTIVE")
            
            if engine.pv_tracker.original_pv:
                pv_str = ' '.join(str(m) for m in engine.pv_tracker.original_pv[:6])
                print(f"  - Last search PV (first 6): {pv_str}")
            
            is_black_move = True
        else:
            print(f"\nMOVE {move_number}w: {description}")
            is_black_move = False
        
        # Make the move
        board.push(move)
        
        if not is_black_move:
            move_number += 1
    
    # NOW THE CRITICAL MOMENT - Position after 9.Qh5 g6, Black to move
    print_section("CRITICAL POSITION REACHED - BLACK TO MOVE (v17.0)")
    print(f"\nCurrent position:")
    print(board)
    print(f"\nFEN: {board.fen()}")
    
    # Check PV tracker state
    print("\n" + "-" * 80)
    print("PV TRACKER STATE:")
    if engine.pv_tracker.following_pv:
        print("  [!] PV FOLLOWING IS ACTIVE")
        print(f"  - Predicted position matches: {board.fen() == engine.pv_tracker.predicted_position_fen}")
        print(f"  - Next planned instant move: {engine.pv_tracker.next_our_move}")
        print(f"  - Will play with 0 nodes, depth 1: YES")
    else:
        print("  [OK] PV following is NOT active")
        print("  - Will perform full search: YES")
    
    return board, engine

def test_critical_position_with_pv_following():
    """Test the critical position with PV-following active"""
    
    board, engine = play_game_sequence_with_pv_tracking()
    
    print_section("TEST 1: PV-FOLLOWING MOVE (Simulates Tournament Behavior)")
    
    print("\nLetting v17.0 make its move (may use PV instant move)...")
    start_time = time.time()
    move_with_pv = engine.search(board, 10.0)  # Give it time, but it may instant-move
    time_with_pv = time.time() - start_time
    
    print(f"\nMove selected: {move_with_pv}")
    print(f"Time taken: {time_with_pv:.3f}s")
    print(f"Was this a PV instant move? {'YES' if time_with_pv < 0.1 else 'NO'}")
    
    return board, move_with_pv, time_with_pv

def test_critical_position_without_pv():
    """Test the same position with fresh engine (no PV following)"""
    
    print_section("TEST 2: FRESH SEARCH (No PV in Memory)")
    
    # Set up the exact same position, but with a fresh engine
    position_fen = "r1b1kb1r/2p1p2p/p1nq2p1/4N2Q/3Pp3/4P3/PPP2PPP/R1B1K2R b KQkq - 2 9"
    board = chess.Board(position_fen)
    
    print(f"\nSame position, fresh engine:")
    print(board)
    print(f"\nFEN: {position_fen}")
    
    # Create fresh engine with no PV history
    fresh_engine = V7P3REngine()
    
    print("\n" + "-" * 80)
    print("PV TRACKER STATE (Fresh Engine):")
    print(f"  - PV following: {fresh_engine.pv_tracker.following_pv}")
    print(f"  - Will perform full search: YES")
    
    print("\nLetting v17.0 search with full thinking time (10 seconds)...")
    start_time = time.time()
    move_without_pv = fresh_engine.search(board, 10.0)
    time_without_pv = time.time() - start_time
    
    print(f"\nMove selected: {move_without_pv}")
    print(f"Time taken: {time_without_pv:.3f}s")
    print(f"Nodes searched: Should be >> 0 (normal search)")
    
    return move_without_pv, time_without_pv

def analyze_results(move_with_pv, time_with_pv, move_without_pv, time_without_pv):
    """Compare results and determine if PV-following caused the blunder"""
    
    print_section("ANALYSIS RESULTS")
    
    print("\nCOMPARISON:")
    print(f"  PV-following move:  {move_with_pv} (time: {time_with_pv:.3f}s)")
    print(f"  Fresh search move:  {move_without_pv} (time: {time_without_pv:.3f}s)")
    print()
    
    # The historical blunder
    blunder_move = chess.Move.from_uci("f7f6")
    
    print("HISTORICAL DATA FROM TOURNAMENT:")
    print(f"  Actual move played: {blunder_move} (f6)")
    print(f"  Evaluation shown:   0.00/1 (depth 1, 0 nodes)")
    print(f"  Result:             After Nxg6, position became -1.86")
    print()
    
    print("-" * 80)
    print("FINDINGS:")
    print("-" * 80)
    
    # Determine verdict
    if move_with_pv == blunder_move and time_with_pv < 0.1:
        print("\n[!!!] PV BLUNDER CONFIRMED!")
        print("  [OK] PV-following produced f6 as instant move")
        print("  [OK] Time < 0.1s confirms instant move (no search)")
        print("  [OK] This matches tournament behavior (depth 1, 0 nodes)")
        
        if move_without_pv != blunder_move:
            print(f"\n[***] SMOKING GUN:")
            print(f"  - Fresh search selected: {move_without_pv} (NOT f6!)")
            print(f"  - PV instant move selected: f6 (the blunder)")
            print(f"  - CONCLUSION: PV-following caused the loss!")
            print()
            print("RECOMMENDATION:")
            print("  -> DISABLE PV instant moves in V17.1")
            print("  -> Always perform at least depth 3-4 search")
            print("  -> PV can be used for move ordering, not instant play")
        else:
            print(f"\n  [!] Fresh search also selected f6")
            print(f"  - This suggests deeper evaluation issue")
            print(f"  - But PV instant move prevented deeper analysis")
    
    elif move_with_pv != blunder_move:
        print("\n[OK] PV-following did NOT produce the blunder this time")
        print("  - May indicate inconsistent PV behavior")
        print("  - Or PV was different in this test run")
        print(f"  - PV move: {move_with_pv}")
        print(f"  - Expected blunder: {blunder_move}")
    
    print("\n" + "-" * 80)
    print("\nRECOMMENDATION FOR V17.1:")
    print("-" * 80)
    
    if move_with_pv == blunder_move and move_without_pv != blunder_move:
        print("""
CRITICAL FIX REQUIRED:

1. DISABLE PV instant moves completely, OR
2. Require minimum depth verification (depth >= 4) before PV instant move, OR  
3. Add sanity check: verify PV move doesn't lose material with quick search

The tournament data shows:
- All 3 losses: depth 1, 0 nodes at critical moments
- PV instant moves bypass tactical verification
- v14.1 exploited this predictability repeatedly

RECOMMENDED IMPLEMENTATION:
- Remove lines 301-310 in v7p3r.py (PV instant move check)
- Or add depth requirement: only instant-move if last PV depth >= 6
- Or add quick verification: 1-second search confirms PV move is best
""")
    else:
        print("""
Further investigation needed:
- PV behavior may be inconsistent between runs
- Evaluation function may have issues with this position type
- Consider both PV fix AND evaluation improvements
""")

def main():
    """Run the complete PV blunder analysis"""
    
    try:
        # Test 1: Play full sequence with PV tracking
        board, move_with_pv, time_with_pv = test_critical_position_with_pv_following()
        
        # Test 2: Test same position fresh
        move_without_pv, time_without_pv = test_critical_position_without_pv()
        
        # Analyze and report
        analyze_results(move_with_pv, time_with_pv, move_without_pv, time_without_pv)
        
    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())
