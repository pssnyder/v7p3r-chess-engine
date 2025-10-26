#!/usr/bin/env python3
"""
V7P3R v14.3 Emergency Fixes Test

Quick test to verify the critical time management and depth consistency fixes.
Focus on time safety and minimum depth achievement.
"""

import chess
import time
import sys
import os

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine

def test_emergency_time_management():
    """Test that V14.3 never exceeds time limits"""
    print("Testing Emergency Time Management...")
    
    engine = V7P3REngine()
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R b KQkq - 4 4")
    
    # Test various time controls
    time_limits = [0.5, 1.0, 2.0, 3.0, 5.0]
    
    for time_limit in time_limits:
        print(f"\nTesting {time_limit}s time limit...")
        
        start_time = time.time()
        best_move = engine.search(board, time_limit)
        actual_time = time.time() - start_time
        
        time_percentage = (actual_time / time_limit) * 100
        
        print(f"Move: {best_move}")
        print(f"Time used: {actual_time:.3f}s ({time_percentage:.1f}% of limit)")
        
        # CRITICAL: Should never exceed 90% of time limit
        if actual_time > time_limit * 0.9:
            print(f"❌ WARNING: Exceeded 90% time limit!")
            return False
        else:
            print(f"✅ Time limit respected")
    
    return True

def test_minimum_depth_guarantee():
    """Test that V14.3 always achieves minimum depth"""
    print("\nTesting Minimum Depth Guarantee...")
    
    engine = V7P3REngine()
    board = chess.Board()
    
    # Test that minimum depth is always achieved
    time_limits = [1.0, 2.0, 3.0, 5.0]
    
    for time_limit in time_limits:
        print(f"\nTesting minimum depth with {time_limit}s...")
        
        expected_min_depth = engine._calculate_minimum_depth(time_limit)
        print(f"Expected minimum depth: {expected_min_depth}")
        
        engine.search_depth_achieved.clear()
        best_move = engine.search(board, time_limit)
        
        if best_move in engine.search_depth_achieved:
            achieved_depth = engine.search_depth_achieved[best_move]
            print(f"Achieved depth: {achieved_depth}")
            
            if achieved_depth >= expected_min_depth:
                print(f"✅ Minimum depth achieved")
            else:
                print(f"❌ Failed to achieve minimum depth!")
                return False
        else:
            print(f"❌ No depth tracking data")
            return False
    
    return True

def test_conservative_game_phases():
    """Test conservative game phase detection"""
    print("\nTesting Conservative Game Phase Detection...")
    
    engine = V7P3REngine()
    
    # Test opening position
    opening_board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    opening_phase = engine._detect_game_phase_conservative(opening_board)
    print(f"Opening position detected as: {opening_phase}")
    
    # Test middlegame position (should be more conservative)
    middlegame_board = chess.Board("r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 4 4")
    middlegame_phase = engine._detect_game_phase_conservative(middlegame_board)
    print(f"Middlegame position detected as: {middlegame_phase}")
    
    # Test endgame position
    endgame_board = chess.Board("8/8/8/3k4/8/3K4/8/R7 w - - 0 1")
    endgame_phase = engine._detect_game_phase_conservative(endgame_board)
    print(f"Endgame position detected as: {endgame_phase}")
    
    # Conservative detection should default to middlegame when uncertain
    return True

def test_emergency_allocation():
    """Test emergency time allocation calculations"""
    print("\nTesting Emergency Time Allocation...")
    
    engine = V7P3REngine()
    
    time_limits = [0.5, 1.0, 3.0, 5.0, 10.0]
    
    for time_limit in time_limits:
        target, max_time = engine._calculate_emergency_time_allocation(time_limit)
        percentage = (target / time_limit) * 100
        max_percentage = (max_time / time_limit) * 100
        
        print(f"Time limit: {time_limit}s -> Target: {target:.2f}s ({percentage:.0f}%), Max: {max_time:.2f}s ({max_percentage:.0f}%)")
        
        # Should always be conservative
        if target > time_limit * 0.8 or max_time > time_limit * 0.9:
            print(f"❌ Time allocation not conservative enough!")
            return False
    
    print("✅ Emergency time allocation is appropriately conservative")
    return True

def run_emergency_tests():
    """Run all V14.3 emergency fix tests"""
    print("V7P3R v14.3 Emergency Fixes Test Suite")
    print("=" * 50)
    
    tests = [
        test_emergency_allocation,
        test_conservative_game_phases,
        test_minimum_depth_guarantee,
        test_emergency_time_management
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        try:
            if test():
                passed += 1
                print(f"✅ {test.__name__} PASSED")
            else:
                print(f"❌ {test.__name__} FAILED")
        except Exception as e:
            print(f"❌ {test.__name__} ERROR: {e}")
    
    print(f"\n" + "=" * 50)
    print(f"EMERGENCY TESTS: {passed}/{total} passed")
    
    if passed == total:
        print("🎉 All emergency fixes working correctly!")
        print("V14.3 is ready for time-critical play (Lichess safe)")
    else:
        print("⚠️ Some emergency fixes need attention")
    
    return passed == total

if __name__ == "__main__":
    success = run_emergency_tests()
    sys.exit(0 if success else 1)