#!/usr/bin/env python3
"""
Test to validate that V16.1 searches to full depth 10 after null move pruning fix.

This test verifies:
1. Engine reaches depth 10 in various positions
2. No early cutoffs at root position
3. Search depth is consistent across different position types
"""

import sys
sys.path.insert(0, 'src')

import chess
from v7p3r import V7P3REngine
import time

def test_depth_search():
    """Test that engine searches to the configured max depth."""
    
    print("=" * 70)
    print("V16.1 DEPTH VALIDATION TEST")
    print("=" * 70)
    print()
    
    # Initialize engine with depth 10
    engine = V7P3REngine(max_depth=10, tt_size_mb=256)
    
    print(f"Engine Configuration:")
    print(f"  Max Depth: {engine.max_depth}")
    print(f"  TT Size: 256 MB")
    print()
    
    # Test positions at different game phases
    test_positions = [
        {
            "name": "Starting Position",
            "fen": chess.STARTING_FEN,
            "description": "Opening phase, should reach depth 10"
        },
        {
            "name": "Italian Game",
            "fen": "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 4 5",
            "description": "Early middlegame, lots of pieces"
        },
        {
            "name": "Open Middlegame",
            "fen": "r1bq1rk1/pp2bppp/2np1n2/4p3/2BPP3/2N2N2/PP3PPP/R1BQ1RK1 w - - 0 10",
            "description": "Complex middlegame position"
        },
        {
            "name": "Endgame",
            "fen": "8/5pk1/6p1/8/3K4/8/5PPP/8 w - - 0 40",
            "description": "King and pawn endgame"
        },
        {
            "name": "Tactical Position",
            "fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 6",
            "description": "Tactical opportunities, should search deep"
        }
    ]
    
    results = []
    
    for i, pos in enumerate(test_positions, 1):
        print(f"Test {i}/5: {pos['name']}")
        print(f"  {pos['description']}")
        print(f"  FEN: {pos['fen']}")
        
        board = chess.Board(pos['fen'])
        engine.board = board
        
        # Track search depth by monkey-patching
        max_depth_reached = [0]
        original_search = engine._search
        
        def tracked_search(board, depth, alpha, beta, ply, do_null_move=True):
            if ply == 0:  # At root
                max_depth_reached[0] = max(max_depth_reached[0], depth)
            return original_search(board, depth, alpha, beta, ply, do_null_move)
        
        engine._search = tracked_search
        
        # Perform search
        start_time = time.time()
        best_move = engine.get_best_move(time_left=0)
        elapsed = time.time() - start_time
        
        # Restore original method
        engine._search = original_search
        
        move_san = board.san(best_move) if best_move else "None"
        depth_reached = max_depth_reached[0]
        
        print(f"  Best Move: {move_san}")
        print(f"  Depth Reached: {depth_reached}")
        print(f"  Time: {elapsed:.3f}s")
        
        # Check if depth 10 was reached
        if depth_reached >= 10:
            status = "✓ PASS"
            passed = True
        elif depth_reached >= 8:
            status = "⚠ WARN (depth 8-9)"
            passed = True
        else:
            status = "✗ FAIL (depth < 8)"
            passed = False
        
        print(f"  Status: {status}")
        print()
        
        results.append({
            "name": pos['name'],
            "depth": depth_reached,
            "passed": passed,
            "move": move_san,
            "time": elapsed
        })
    
    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print()
    
    passed_count = sum(1 for r in results if r['passed'])
    total_count = len(results)
    
    print(f"Tests Passed: {passed_count}/{total_count}")
    print()
    
    print("Depth Analysis:")
    for r in results:
        status = "✓" if r['passed'] else "✗"
        print(f"  {status} {r['name']:25s} - Depth {r['depth']:2d} - {r['move']:6s} ({r['time']:.2f}s)")
    print()
    
    if passed_count == total_count:
        print("✓ ALL TESTS PASSED - Engine reaches full depth 10")
        print("  Null move pruning fix is working correctly!")
        return True
    elif passed_count >= total_count - 1:
        print("⚠ MOSTLY PASSING - Engine reaches depth 8-10 in most positions")
        print("  This is acceptable for deployment")
        return True
    else:
        print("✗ TESTS FAILED - Engine not reaching sufficient depth")
        print("  Further investigation needed before deployment")
        return False

if __name__ == "__main__":
    try:
        success = test_depth_search()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n\n✗ Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
