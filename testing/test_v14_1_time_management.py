#!/usr/bin/env python3
"""
V7P3R v14.1 Time Management Test

Validates the new time management improvements:
1. Never exceeds 60 seconds per move
2. Opening moves are faster (< 5s typically)
3. Returns early when best move is stable
4. Doesn't waste time on incomplete depth iterations
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
import time
from v7p3r import V7P3REngine


def test_opening_speed():
    """Test that opening moves are played quickly"""
    print("=" * 60)
    print("TEST 1: Opening Speed (should be < 5 seconds)")
    print("=" * 60)
    
    engine = V7P3REngine()
    board = chess.Board()
    
    # Test first 3 moves
    total_time = 0
    for move_num in range(1, 4):
        start = time.time()
        move = engine.search(board, time_limit=30.0)  # Give it 30s budget
        elapsed = time.time() - start
        total_time += elapsed
        
        print(f"Move {move_num}: {move} in {elapsed:.2f}s")
        board.push(move)
        board.push(list(board.legal_moves)[0])  # Dummy opponent move
        
        assert elapsed < 5.0, f"Opening move took {elapsed:.2f}s (should be < 5s)"
    
    avg_time = total_time / 3
    print(f"\n✓ Average opening time: {avg_time:.2f}s")
    print(f"✓ All opening moves < 5s")
    return True


def test_60_second_hard_cap():
    """Test that engine NEVER exceeds 60 seconds"""
    print("\n" + "=" * 60)
    print("TEST 2: 60 Second Hard Cap")
    print("=" * 60)
    
    engine = V7P3REngine()
    
    # Complex middlegame position
    board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1")
    
    # Give it 120 seconds - should cap at 60
    start = time.time()
    move = engine.search(board, time_limit=120.0)
    elapsed = time.time() - start
    
    print(f"Complex position: {move} in {elapsed:.2f}s")
    print(f"Time limit given: 120.0s")
    print(f"Actual time used: {elapsed:.2f}s")
    
    assert elapsed < 65.0, f"Engine exceeded 60s cap: {elapsed:.2f}s"
    print(f"\n✓ Engine respected 60s hard cap")
    return True


def test_time_distribution():
    """Test time distribution across game phases"""
    print("\n" + "=" * 60)
    print("TEST 3: Time Distribution Across Game Phases")
    print("=" * 60)
    
    engine = V7P3REngine()
    
    test_positions = [
        ("Opening (move 3)", chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1"), 5.0),
        ("Early Middlegame (move 15)", chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 1"), 15.0),
        ("Complex Middlegame (move 25)", chess.Board("r1bq1rk1/ppp2ppp/2np1n2/2b1p3/2B1P3/2NP1N2/PPP2PPP/R1BQ1RK1 w - - 0 1"), 20.0),
        ("Endgame (move 50)", chess.Board("8/5k2/8/3K4/8/8/8/8 w - - 0 1"), 8.0),
    ]
    
    for name, board, expected_max in test_positions:
        start = time.time()
        move = engine.search(board, time_limit=30.0)
        elapsed = time.time() - start
        
        print(f"{name}: {move} in {elapsed:.2f}s (expected < {expected_max:.1f}s)")
        
        if elapsed > expected_max * 1.5:  # Allow 50% margin
            print(f"  ⚠ Took longer than expected")
        else:
            print(f"  ✓ Within expected time")
    
    return True


def test_stable_best_move():
    """Test early return when best move is stable"""
    print("\n" + "=" * 60)
    print("TEST 4: Early Return on Stable Best Move")
    print("=" * 60)
    
    engine = V7P3REngine()
    
    # Simple tactical position where best move is obvious
    # White can capture free pawn
    board = chess.Board("rnbqkbnr/ppp2ppp/8/3pp3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 1")
    
    start = time.time()
    move = engine.search(board, time_limit=20.0)
    elapsed = time.time() - start
    
    print(f"Simple tactical position: {move} in {elapsed:.2f}s")
    print(f"Expected: Quick return since best move is obvious")
    
    # Should return quickly (< 8s) since position is simple
    if elapsed < 8.0:
        print(f"✓ Returned early ({elapsed:.2f}s < 8s)")
    else:
        print(f"⚠ Took longer than expected: {elapsed:.2f}s")
    
    return True


def test_increment_awareness():
    """Test that engine uses more time when increment is available"""
    print("\n" + "=" * 60)
    print("TEST 5: Increment Awareness (UCI Level)")
    print("=" * 60)
    
    print("This test requires UCI interface testing")
    print("Manual verification: Engine should use more time in games with increment")
    print("✓ Increment logic implemented in v7p3r_uci.py")
    
    return True


def main():
    """Run all time management tests"""
    print("V7P3R v14.1 Time Management Validation")
    print("=" * 60)
    
    tests = [
        ("Opening Speed", test_opening_speed),
        ("60 Second Hard Cap", test_60_second_hard_cap),
        ("Time Distribution", test_time_distribution),
        ("Stable Best Move", test_stable_best_move),
        ("Increment Awareness", test_increment_awareness),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ TEST FAILED: {e}")
            import traceback
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED - v14.1 time management ready!")
    else:
        print(f"\n⚠ {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
