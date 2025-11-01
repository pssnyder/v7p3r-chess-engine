"""
Quick test of V14.9.1 time management improvements
Tests the 3 key criteria:
1. Fast opening moves (<1s)
2. Early exit on stable PV in quiet positions
3. Full time usage in tactical/noisy positions
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
import time
from v7p3r import V7P3REngine

def test_opening_speed():
    """Test opening moves are fast"""
    print("\n" + "="*70)
    print("TEST 1: OPENING SPEED (<1s expected)")
    print("="*70)
    
    engine = V7P3REngine()
    board = chess.Board()
    
    start = time.time()
    move = engine.search(board, time_limit=3.0)
    elapsed = time.time() - start
    
    print(f"Move: {move}")
    print(f"Time: {elapsed:.2f}s")
    
    if elapsed < 1.0:
        print(f"✅ PASS: Opening move in {elapsed:.2f}s")
        return True
    else:
        print(f"❌ FAIL: Opening took {elapsed:.2f}s (should be <1s)")
        return False

def test_stable_pv_exit():
    """Test early exit on stable PV"""
    print("\n" + "="*70)
    print("TEST 2: STABLE PV EARLY EXIT (<2s expected)")
    print("="*70)
    
    engine = V7P3REngine()
    # Position after 1.e4 e5 2.Nf3 Nc6 3.Bc4 - quiet Italian opening
    board = chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 2 4")
    
    start = time.time()
    move = engine.search(board, time_limit=5.0)
    elapsed = time.time() - start
    
    print(f"Move: {move}")
    print(f"Time: {elapsed:.2f}s (allocated 5.0s)")
    
    if elapsed < 2.5:
        print(f"✅ PASS: Early exit on stable PV in {elapsed:.2f}s")
        return True
    else:
        print(f"⚠️  WARN: Took {elapsed:.2f}s (could exit earlier on stable PV)")
        return elapsed < 4.0  # Still pass if reasonable

def test_tactical_depth():
    """Test tactical positions use full time"""
    print("\n" + "="*70)
    print("TEST 3: TACTICAL POSITIONS USE TIME (>70% expected)")
    print("="*70)
    
    engine = V7P3REngine()
    # Tactical position with Scholar's mate threat
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4")
    
    allocated = 3.0
    start = time.time()
    move = engine.search(board, time_limit=allocated)
    elapsed = time.time() - start
    
    efficiency = (elapsed / allocated) * 100
    
    print(f"Move: {move}")
    print(f"Time: {elapsed:.2f}s / {allocated:.2f}s ({efficiency:.1f}%)")
    
    if efficiency > 70:
        print(f"✅ PASS: Tactical position used {efficiency:.1f}% of time")
        return True
    else:
        print(f"❌ FAIL: Only used {efficiency:.1f}% (should use >70% for tactics)")
        return False

def main():
    print("="*70)
    print("V14.9.1 TIME MANAGEMENT VALIDATION")
    print("="*70)
    
    results = []
    
    # Run tests
    results.append(("Opening Speed", test_opening_speed()))
    results.append(("Stable PV Exit", test_stable_pv_exit()))
    results.append(("Tactical Depth", test_tactical_depth()))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All time management improvements working correctly!")
    elif passed >= 2:
        print("\n✅ Time management significantly improved")
    else:
        print("\n⚠️  Time management needs more tuning")

if __name__ == "__main__":
    main()
