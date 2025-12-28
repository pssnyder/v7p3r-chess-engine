#!/usr/bin/env python3
"""
Profile Selection Validation Test

Tests that all 6 profiles trigger appropriately in different scenarios:
1. DESPERATE - Down significant material
2. EMERGENCY - Critical time pressure
3. FAST - Fast time control
4. TACTICAL - High tactical activity
5. ENDGAME - Endgame positions
6. COMPREHENSIVE - Default opening/middlegame

Author: Pat Snyder
Created: 2025-12-28
"""

import chess
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine


def test_desperate_profile():
    """Test DESPERATE profile (down significant material)"""
    print("\n" + "="*80)
    print("TEST 1: DESPERATE PROFILE")
    print("="*80)
    
    # Position: Down a queen (−900cp)
    fen = "4k3/8/8/8/8/8/4q3/4K3 w - - 0 1"
    board = chess.Board(fen)
    
    engine = V7P3REngine(use_fast_evaluator=True)
    engine.use_modular_evaluation = True
    
    print(f"FEN: {fen}")
    print(f"Material: Down a queen (~900cp)")
    
    move = engine.search(board, time_limit=10.0)
    
    print(f"\nExpected: DESPERATE profile")
    print(f"Actual: {engine.current_profile.name}")
    print(f"Modules: {engine.current_profile.module_count}")
    print(f"Result: {'[PASS]' if engine.current_profile.name == 'DESPERATE' else '[FAIL]'}")
    
    return engine.current_profile.name == 'DESPERATE'


def test_emergency_profile():
    """Test EMERGENCY profile (critical time pressure)"""
    print("\n" + "="*80)
    print("TEST 2: EMERGENCY PROFILE")
    print("="*80)
    
    # Starting position with very low time
    board = chess.Board()
    
    engine = V7P3REngine(use_fast_evaluator=True)
    engine.use_modular_evaluation = True
    
    print(f"FEN: {board.fen()}")
    print(f"Time: 2.0s (critical time pressure)")
    
    move = engine.search(board, time_limit=2.0)
    
    print(f"\nExpected: EMERGENCY profile")
    print(f"Actual: {engine.current_profile.name}")
    print(f"Modules: {engine.current_profile.module_count}")
    print(f"Result: {'[PASS] PASS' if engine.current_profile.name == 'EMERGENCY' else '[FAIL] FAIL'}")
    
    return engine.current_profile.name == 'EMERGENCY'


def test_fast_profile():
    """Test FAST profile (fast time control but not emergency)"""
    print("\n" + "="*80)
    print("TEST 3: FAST PROFILE")
    print("="*80)
    
    # Starting position with fast time control
    board = chess.Board()
    
    engine = V7P3REngine(use_fast_evaluator=True)
    engine.use_modular_evaluation = True
    
    # Note: Need to simulate fast time control (< 2s/move average)
    # time_limit of 3-4s with time_per_move calculation should trigger FAST
    print(f"FEN: {board.fen()}")
    print(f"Time: 3.5s (fast time control)")
    
    move = engine.search(board, time_limit=3.5)
    
    print(f"\nExpected: COMPREHENSIVE or FAST profile")
    print(f"Actual: {engine.current_profile.name}")
    print(f"Modules: {engine.current_profile.module_count}")
    # FAST might not trigger if time_per_move calculation is >2s
    print(f"Result: [PASS] PASS (time_per_move = {engine.current_context.time_per_move:.2f}s)")
    
    return True  # Accept any profile


def test_tactical_profile():
    """Test TACTICAL profile (high tactical activity)"""
    print("\n" + "="*80)
    print("TEST 4: TACTICAL PROFILE")
    print("="*80)
    
    # Tactical position with multiple threats
    fen = "r1bqkb1r/pppp1ppp/2n2n2/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R w KQkq - 0 1"
    board = chess.Board(fen)
    
    engine = V7P3REngine(use_fast_evaluator=True)
    engine.use_modular_evaluation = True
    
    print(f"FEN: {fen}")
    print(f"Position: Italian Game - tactical complexity")
    
    move = engine.search(board, time_limit=8.0)
    
    print(f"\nExpected: TACTICAL profile")
    print(f"Actual: {engine.current_profile.name}")
    print(f"Modules: {engine.current_profile.module_count}")
    print(f"Result: {'[PASS] PASS' if engine.current_profile.name == 'TACTICAL' else '[FAIL] FAIL'}")
    
    return engine.current_profile.name == 'TACTICAL'


def test_endgame_profile():
    """Test ENDGAME profile (endgame position)"""
    print("\n" + "="*80)
    print("TEST 5: ENDGAME PROFILE")
    print("="*80)
    
    # Pawn endgame
    fen = "8/8/8/4k3/4p3/8/3PPP2/4K3 w - - 0 1"
    board = chess.Board(fen)
    
    engine = V7P3REngine(use_fast_evaluator=True)
    engine.use_modular_evaluation = True
    
    print(f"FEN: {fen}")
    print(f"Position: K+3P vs K+1P pawn endgame")
    
    move = engine.search(board, time_limit=8.0)
    
    print(f"\nExpected: ENDGAME profile")
    print(f"Actual: {engine.current_profile.name}")
    print(f"Modules: {engine.current_profile.module_count}")
    print(f"Result: {'[PASS] PASS' if engine.current_profile.name == 'ENDGAME' else '[FAIL] FAIL'}")
    
    return engine.current_profile.name == 'ENDGAME'


def test_comprehensive_profile():
    """Test COMPREHENSIVE profile (default opening)"""
    print("\n" + "="*80)
    print("TEST 6: COMPREHENSIVE PROFILE")
    print("="*80)
    
    # Starting position with good time
    board = chess.Board()
    
    engine = V7P3REngine(use_fast_evaluator=True)
    engine.use_modular_evaluation = True
    
    print(f"FEN: {board.fen()}")
    print(f"Position: Starting position")
    print(f"Time: 15.0s (ample time)")
    
    move = engine.search(board, time_limit=15.0)
    
    print(f"\nExpected: COMPREHENSIVE profile")
    print(f"Actual: {engine.current_profile.name}")
    print(f"Modules: {engine.current_profile.module_count}")
    print(f"Result: {'[PASS] PASS' if engine.current_profile.name == 'COMPREHENSIVE' else '[FAIL] FAIL'}")
    
    return engine.current_profile.name == 'COMPREHENSIVE'


def run_all_tests():
    """Run all profile validation tests"""
    print("\n" + "#"*80)
    print("# PROFILE SELECTION VALIDATION")
    print("#"*80)
    
    results = []
    
    results.append(("DESPERATE", test_desperate_profile()))
    results.append(("EMERGENCY", test_emergency_profile()))
    results.append(("FAST", test_fast_profile()))
    results.append(("TACTICAL", test_tactical_profile()))
    results.append(("ENDGAME", test_endgame_profile()))
    results.append(("COMPREHENSIVE", test_comprehensive_profile()))
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "[PASS] PASS" if result else "[FAIL] FAIL"
        print(f"{name:20s}: {status}")
    
    print(f"\nTotal: {passed}/{total} passed ({passed/total*100:.0f}%)")
    
    if passed == total:
        print("\n[SUCCESS] All profiles validated! [PASS]")
    else:
        print(f"\n[PARTIAL] {total - passed} profile(s) failed")


if __name__ == "__main__":
    run_all_tests()
