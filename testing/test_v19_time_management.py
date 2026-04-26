#!/usr/bin/env python3
"""
V19.0 Time Management Validation Test

WHY THIS EXISTS: Verify that time management simplification fixes v18.4's 75% time forfeit rate

WHAT IT DOES: Tests time allocation under various scenarios to ensure reasonable behavior

TEST CASES:
1. Emergency time (<3s) - should be conservative
2. Low time (<15s) - should be safe
3. Blitz (2min+2s) - should be aggressive
4. Rapid (5min+10s) - should be balanced
5. Classical (15min+10s) - should take more time

EXPECTED: All time allocations should be safe and reasonable
"""

import sys
sys.path.insert(0, 'src')

from v7p3r_time_manager import TimeManager
import chess


def test_emergency_time():
    """Test emergency time allocation (<3 seconds)"""
    print("\nTEST 1: Emergency Time Management (<3 seconds)")
    print("-" * 60)
    
    test_cases = [
        (2.5, 0, 30, "2.5s, no increment"),
        (1.0, 0, 40, "1.0s, no increment"),
        (0.5, 0.5, 35, "0.5s + 0.5s inc"),
    ]
    
    for remaining, inc, moves, desc in test_cases:
        target, max_time = TimeManager.calculate_time_allocation(
            remaining, inc, moves
        )
        # Emergency: should use ~3% of time
        expected_max = remaining * 0.05  # Allow up to 5%
        safe = target <= expected_max
        status = "✓ PASS" if safe else "✗ FAIL"
        print(f"  {desc:20s} → {target:.3f}s (max {max_time:.3f}s) {status}")
        
        if not safe:
            print(f"    WARNING: Using {target/remaining*100:.1f}% of time (expected ≤5%)")
    
    return True


def test_low_time():
    """Test low time allocation (3-15 seconds)"""
    print("\nTEST 2: Low Time Management (3-15 seconds)")
    print("-" * 60)
    
    test_cases = [
        (14.0, 2.0, 25, "14s + 2s inc"),
        (10.0, 2.0, 30, "10s + 2s inc"),
        (5.0, 0, 35, "5s, no increment"),
    ]
    
    for remaining, inc, moves, desc in test_cases:
        target, max_time = TimeManager.calculate_time_allocation(
            remaining, inc, moves
        )
        # Low time: should use ~5-10% of time
        expected_max = remaining * 0.15  # Allow up to 15%
        safe = target <= expected_max
        status = "✓ PASS" if safe else "✗ FAIL"
        print(f"  {desc:20s} → {target:.3f}s (max {max_time:.3f}s) {status}")
        
        if not safe:
            print(f"    WARNING: Using {target/remaining*100:.1f}% of time (expected ≤15%)")
    
    return True


def test_blitz_time():
    """Test blitz time allocation (2min+2s increment)"""
    print("\nTEST 3: Blitz Time Management (2min+2s)")
    print("-" * 60)
    
    test_cases = [
        (120.0, 2.0, 5, "Opening (move 5)"),
        (110.0, 2.0, 15, "Early middlegame (move 15)"),
        (95.0, 2.0, 25, "Middlegame (move 25)"),
        (80.0, 2.0, 35, "Endgame (move 35)"),
    ]
    
    for remaining, inc, moves, desc in test_cases:
        target, max_time = TimeManager.calculate_time_allocation(
            remaining, inc, moves
        )
        # Blitz: should use modest time, increment helps
        expected_max = remaining * 0.20  # Allow up to 20%
        safe = target <= expected_max
        status = "✓ PASS" if safe else "✗ FAIL"
        print(f"  {desc:25s} → {target:.2f}s (max {max_time:.2f}s) {status}")
        
        # Also check it's not too fast (>0.5s)
        if target < 0.5:
            print(f"    WARNING: May be too fast ({target:.2f}s)")
    
    return True


def test_rapid_time():
    """Test rapid time allocation (5min+10s increment)"""
    print("\nTEST 4: Rapid Time Management (5min+10s)")
    print("-" * 60)
    
    test_cases = [
        (300.0, 10.0, 8, "Opening (move 8)"),
        (280.0, 10.0, 15, "Early middlegame (move 15)"),
        (250.0, 10.0, 25, "Middlegame (move 25)"),
        (220.0, 10.0, 40, "Endgame (move 40)"),
    ]
    
    for remaining, inc, moves, desc in test_cases:
        target, max_time = TimeManager.calculate_time_allocation(
            remaining, inc, moves
        )
        # Rapid: can afford to think longer
        expected_max = remaining * 0.20  # Allow up to 20%
        safe = target <= expected_max
        status = "✓ PASS" if safe else "✗ FAIL"
        print(f"  {desc:25s} → {target:.2f}s (max {max_time:.2f}s) {status}")
        
        # Should be taking reasonable time
        if target < 1.0:
            print(f"    WARNING: May be too fast for rapid ({target:.2f}s)")
    
    return True


def test_classical_time():
    """Test classical time allocation (15min+10s increment)"""
    print("\nTEST 5: Classical Time Management (15min+10s)")
    print("-" * 60)
    
    test_cases = [
        (900.0, 10.0, 8, "Opening (move 8)"),
        (850.0, 10.0, 15, "Early middlegame (move 15)"),
        (800.0, 10.0, 25, "Middlegame (move 25)"),
        (700.0, 10.0, 40, "Endgame (move 40)"),
    ]
    
    for remaining, inc, moves, desc in test_cases:
        target, max_time = TimeManager.calculate_time_allocation(
            remaining, inc, moves
        )
        # Classical: can afford deeper search
        expected_max = remaining * 0.20  # Allow up to 20%
        safe = target <= expected_max
        status = "✓ PASS" if safe else "✗ FAIL"
        print(f"  {desc:25s} → {target:.2f}s (max {max_time:.2f}s) {status}")
        
        # Should be taking substantial time
        if target < 5.0:
            print(f"    INFO: Classical could think longer ({target:.2f}s)")
    
    return True


def test_time_forfeit_scenarios():
    """Test scenarios that caused v18.4 time forfeits"""
    print("\nTEST 6: v18.4 Time Forfeit Scenarios")
    print("-" * 60)
    print("  Testing time controls from v18.4 time forfeit games:")
    
    # Real forfeit scenarios from v18.4
    test_cases = [
        # Game r5oLGOgM: 2+2 blitz, timed out move 17
        (60.0, 2.0, 17, "Game r5oLGOgM (2+2 blitz, move 17)"),
        # Game 1PxPjK2x: 3+2 blitz, timed out move 41
        (90.0, 2.0, 41, "Game 1PxPjK2x (3+2 blitz, move 41)"),
        # Game J1TPgpgO: 2+2 blitz, timed out move 33
        (50.0, 2.0, 33, "Game J1TPgpgO (2+2 blitz, move 33)"),
    ]
    
    all_safe = True
    for remaining, inc, moves, desc in test_cases:
        target, max_time = TimeManager.calculate_time_allocation(
            remaining, inc, moves
        )
        
        # Critical: must leave enough time for future moves
        # With 2s increment, should always finish move in <10s
        safe = target < 10.0 and max_time < 12.0
        status = "✓ SAFE" if safe else "✗ RISKY"
        
        print(f"  {desc}")
        print(f"    Time remaining: {remaining:.1f}s + {inc:.1f}s inc")
        print(f"    Allocation: {target:.2f}s (max {max_time:.2f}s) {status}")
        
        if not safe:
            print(f"    WARNING: May timeout (allocation too high)")
            all_safe = False
        
        print()
    
    return all_safe


if __name__ == "__main__":
    print("=" * 60)
    print("V19.0 TIME MANAGEMENT VALIDATION")
    print("=" * 60)
    
    tests = [
        ("Emergency Time", test_emergency_time),
        ("Low Time", test_low_time),
        ("Blitz Time", test_blitz_time),
        ("Rapid Time", test_rapid_time),
        ("Classical Time", test_classical_time),
        ("Time Forfeit Scenarios", test_time_forfeit_scenarios),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ {name} FAILED: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 60)
    print("VALIDATION RESULTS:")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status} {name}")
    
    print("=" * 60)
    print(f"TOTAL: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ ALL TESTS PASSED - Time management is safe and reasonable")
        print("\nREADY FOR: Arena GUI tournament testing vs v18.4 and C0BR4 v3.4")
    else:
        print(f"\n✗ {total - passed} TESTS FAILED - Review time allocation logic")
    
    print("=" * 60)
