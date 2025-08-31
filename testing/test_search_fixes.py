#!/usr/bin/env python3
"""
Test V10 Search Stability and PV Display
"""

import chess
import sys
import os

# Add src directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from v7p3r import V7P3RCleanEngine

def test_search_stability():
    """Test that the engine doesn't change its mind on the last iteration"""
    
    print("🔍 SEARCH STABILITY TEST")
    print("=" * 50)
    
    engine = V7P3RCleanEngine()
    
    # Test with a tactical position where there's a clear best move
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 4")
    print(f"Position: {board.fen()}")
    print("This is the Italian Game - should have a consistent best move")
    
    # Run multiple searches to see if they're consistent
    moves = []
    for i in range(3):
        print(f"\nSearch {i+1}:")
        move = engine.search(board, 2.0)  # 2 second search
        moves.append(move)
        print(f"Best move: {move}")
        engine.new_game()  # Reset for next search
    
    # Check consistency
    if len(set(moves)) == 1:
        print(f"\n✅ STABLE: All searches returned {moves[0]}")
    else:
        print(f"\n⚠️  UNSTABLE: Different moves returned: {moves}")
    
    return len(set(moves)) == 1

def test_pv_extraction():
    """Test that PV shows the full line, not just the first move"""
    
    print(f"\n📋 PV EXTRACTION TEST")
    print("=" * 50)
    
    engine = V7P3RCleanEngine()
    
    # Test position where we can analyze the PV
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    print(f"Position: {board.fen()}")
    print("After 1.e4 - should show multi-move PV")
    
    print(f"\nRunning search with detailed output...")
    move = engine.search(board, 3.0)
    
    print(f"\nFinal best move: {move}")
    
    # Test PV extraction directly
    print(f"\nTesting PV extraction method:")
    pv_line = engine._extract_pv(board, 4)
    print(f"Extracted PV: {' '.join(str(m) for m in pv_line)}")
    
    if len(pv_line) > 1:
        print(f"✅ PV WORKING: Shows {len(pv_line)} moves")
        return True
    else:
        print(f"❌ PV ISSUE: Only shows {len(pv_line)} move(s)")
        return False

def test_iterative_deepening():
    """Test that iterative deepening works properly"""
    
    print(f"\n⏳ ITERATIVE DEEPENING TEST")
    print("=" * 50)
    
    engine = V7P3RCleanEngine()
    
    # Use a complex position
    board = chess.Board("r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1")
    print(f"Position: {board.fen()}")
    print("Complex tactical position - should show multiple depths")
    
    print(f"\nRunning search...")
    move = engine.search(board, 4.0)  # 4 second search
    
    print(f"\nFinal move: {move}")
    print(f"Nodes searched: {engine.nodes_searched}")
    
    # Check if we got reasonable node count (should be > 1000 in 4 seconds)
    if engine.nodes_searched > 1000:
        print(f"✅ ITERATIVE DEEPENING: Searched {engine.nodes_searched} nodes")
        return True
    else:
        print(f"❌ SEARCH ISSUE: Only {engine.nodes_searched} nodes")
        return False

def test_time_management():
    """Test that time management doesn't cause last-minute changes"""
    
    print(f"\n⏰ TIME MANAGEMENT TEST")
    print("=" * 50)
    
    engine = V7P3RCleanEngine()
    
    # Use starting position
    board = chess.Board()
    print("Position: Starting position")
    
    # Test different time limits
    time_limits = [1.0, 2.0, 3.0]
    moves = []
    
    for time_limit in time_limits:
        print(f"\nTime limit: {time_limit}s")
        move = engine.search(board, time_limit)
        moves.append(move)
        print(f"Move: {move}")
        engine.new_game()
    
    # Check if moves are consistent (they should be for starting position)
    unique_moves = set(moves)
    if len(unique_moves) <= 2:  # Allow for some variation
        print(f"\n✅ TIME MANAGEMENT: Consistent moves across time limits")
        return True
    else:
        print(f"\n⚠️  TIME MANAGEMENT: Too much variation: {moves}")
        return False

if __name__ == "__main__":
    print("🧪 V10 SEARCH FIXES VERIFICATION")
    print("=" * 60)
    
    stability_ok = test_search_stability()
    pv_ok = test_pv_extraction()
    iterative_ok = test_iterative_deepening()
    time_ok = test_time_management()
    
    print(f"\n📊 TEST RESULTS SUMMARY")
    print("=" * 60)
    print(f"✅ Search Stability: {'PASS' if stability_ok else 'FAIL'}")
    print(f"✅ PV Extraction: {'PASS' if pv_ok else 'FAIL'}")
    print(f"✅ Iterative Deepening: {'PASS' if iterative_ok else 'FAIL'}")
    print(f"✅ Time Management: {'PASS' if time_ok else 'FAIL'}")
    
    all_passed = all([stability_ok, pv_ok, iterative_ok, time_ok])
    
    if all_passed:
        print(f"\n🏆 ALL TESTS PASSED - V10 search issues fixed!")
    else:
        print(f"\n⚠️  Some tests failed - needs more work")
        
    print("\n🔧 Key Fixes Applied:")
    print("• Best move stability across iterations")
    print("• Full PV line extraction from transposition table")
    print("• Better time management with early termination")
    print("• Graceful handling of interrupted searches")
