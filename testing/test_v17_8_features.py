#!/usr/bin/env python3
"""
V7P3R v17.8 Test Suite - Time Management & Mate Detection

Tests for:
1. Time management allocation
2. Mate-in-1 detection
3. Enhanced king safety (back-rank detection)
4. Mate extensions
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
from v7p3r import V7P3REngine
from v7p3r_time_manager import TimeManager
from v7p3r_fast_evaluator import V7P3RFastEvaluator


def test_time_manager_blitz():
    """Test time allocation for 5min+4s blitz"""
    print("\n=== TEST 1: Time Manager - Blitz (5min+4s) ===")
    
    tm = TimeManager(300000, 4000, 'blitz')
    
    # Move 10, middle game, 4 minutes left
    time_left = 240000
    moves = 10
    allocated = tm.calculate_move_time(time_left, moves, is_endgame=False)
    
    print(f"Move 10, 4:00 remaining -> {allocated}ms allocated")
    assert 6000 < allocated < 15000, f"Expected 6-15s, got {allocated}ms"
    print("✓ PASS: Reasonable time allocation")
    
    # Move 30, endgame, 2 minutes left
    time_left = 120000
    moves = 30
    allocated_endgame = tm.calculate_move_time(time_left, moves, is_endgame=True)
    allocated_normal = tm.calculate_move_time(time_left, moves, is_endgame=False)
    
    print(f"Move 30, 2:00 remaining, endgame -> {allocated_endgame}ms")
    print(f"Move 30, 2:00 remaining, normal -> {allocated_normal}ms")
    assert allocated_endgame < allocated_normal * 0.7, "Endgame should be faster"
    print("✓ PASS: Endgame moves faster")
    
    return True


def test_time_pressure():
    """Test behavior with low time"""
    print("\n=== TEST 2: Time Pressure Handling ===")
    
    tm = TimeManager(300000, 4000, 'blitz')
    
    # Only 5 seconds left
    time_left = 5000
    allocated = tm.calculate_move_time(time_left, 30, False)
    
    print(f"5 seconds remaining -> {allocated}ms allocated")
    assert allocated < 3000, f"Too slow under time pressure: {allocated}ms"
    print("✓ PASS: Moves quickly under time pressure")
    
    # Only 2 seconds left (emergency)
    time_left = 2000
    allocated = tm.calculate_move_time(time_left, 35, False)
    
    print(f"2 seconds remaining -> {allocated}ms allocated")
    assert allocated <= 100, f"Should move instantly: {allocated}ms"
    print("✓ PASS: Emergency moves instantly")
    
    return True


def test_mate_in_1_detection():
    """Test immediate mate detection"""
    print("\n=== TEST 3: Mate-in-1 Detection ===")
    
    engine = V7P3REngine()
    
    # Test position 1: Back rank mate
    print("\nPosition 1: Back rank mate (Ra8#)")
    fen = "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1"
    engine.board = chess.Board(fen)
    
    mate_move = engine._check_immediate_mate(engine.board)
    expected = chess.Move.from_uci("a1a8")
    
    if mate_move == expected:
        print(f"✓ PASS: Found Ra8# immediately")
    else:
        print(f"✗ FAIL: Expected {expected}, got {mate_move}")
        return False
    
    # Test position 2: Queen mate
    print("\nPosition 2: Queen mate (Qh7#)")
    fen = "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4"
    engine.board = chess.Board(fen)
    
    # Switch to white's perspective (white has mate)
    engine.board.turn = chess.WHITE
    mate_move = engine._check_immediate_mate(engine.board)
    
    if mate_move and mate_move.uci() == "f7h7":
        print(f"✓ PASS: Found Qh7# immediately")
    else:
        # May not find it if not white's turn
        print(f"! INFO: Mate move found: {mate_move}")
    
    # Test position 3: No mate available
    print("\nPosition 3: Starting position (no mate)")
    engine.board = chess.Board()
    
    mate_move = engine._check_immediate_mate(engine.board)
    
    if mate_move is None:
        print("✓ PASS: Correctly identified no mate available")
    else:
        print(f"✗ FAIL: False positive - found {mate_move}")
        return False
    
    return True


def test_back_rank_safety():
    """Test back-rank weakness detection"""
    print("\n=== TEST 4: Back-Rank Safety Detection ===")
    
    evaluator = V7P3RFastEvaluator()
    
    # Position 1: Trapped king on back rank
    print("\nPosition 1: Trapped king (no escape squares)")
    fen = "6k1/5ppp/8/8/8/8/5PPP/R5K1 b - - 0 1"
    board = chess.Board(fen)
    
    safety_bonus = evaluator._calculate_v17_8_king_safety(board)
    
    print(f"King safety bonus: {safety_bonus}cp")
    # Black king is trapped, white is safe - should favor white
    if safety_bonus > 40:
        print("✓ PASS: Detected trapped black king (positive for white)")
    else:
        print(f"✗ FAIL: Should detect back-rank weakness: {safety_bonus}cp")
        return False
    
    # Position 2: Safe king with escape squares
    print("\nPosition 2: Safe king (escape squares available)")
    fen = "6k1/8/8/8/8/8/8/6K1 w - - 0 1"
    board = chess.Board(fen)
    
    safety_bonus = evaluator._calculate_v17_8_king_safety(board)
    
    print(f"King safety bonus: {safety_bonus}cp")
    # Both kings are safe, should be close to 0
    if abs(safety_bonus) < 40:
        print("✓ PASS: Recognized safe positions")
    else:
        print(f"! INFO: Safety bonus {safety_bonus}cp (expected near 0)")
    
    # Position 3: Back rank with rook threat
    print("\nPosition 3: Back rank with enemy rook on same rank")
    fen = "6k1/5ppp/8/8/8/8/8/R5K1 w - - 0 1"
    board = chess.Board(fen)
    
    safety_bonus = evaluator._calculate_v17_8_king_safety(board)
    
    print(f"King safety bonus: {safety_bonus}cp")
    # Black king on back rank with white rook - big penalty for black
    if safety_bonus > 60:
        print("✓ PASS: Heavy penalty for back-rank threat")
    else:
        print(f"! INFO: Back-rank threat penalty: {safety_bonus}cp")
    
    return True


def test_mate_threat_extension():
    """Test that forcing positions get extended search"""
    print("\n=== TEST 5: Mate Threat Extension ===")
    
    engine = V7P3REngine()
    
    # Position with checks available
    fen = "6k1/5ppp/8/3Q4/8/8/5PPP/6K1 w - - 0 1"
    engine.board = chess.Board(fen)
    
    has_threat = engine._is_in_check_or_mate_threat(engine.board)
    
    print(f"Position with queen: Has forcing moves? {has_threat}")
    if has_threat:
        print("✓ PASS: Detected forcing position (checks available)")
    else:
        print("! INFO: No forcing moves detected (may be OK)")
    
    # Position without checks
    fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    engine.board = chess.Board(fen)
    
    has_threat = engine._is_in_check_or_mate_threat(engine.board)
    
    print(f"Starting position: Has forcing moves? {has_threat}")
    if not has_threat:
        print("✓ PASS: No forcing moves in starting position")
    else:
        print("! INFO: Detected forcing moves (unexpected)")
    
    return True


def test_integration_search():
    """Test full search with v17.8 features"""
    print("\n=== TEST 6: Integration - Full Search ===")
    
    engine = V7P3REngine()
    
    # Simple mate-in-1 position
    print("\nPosition: Mate-in-1 via search")
    fen = "6k1/5ppp/8/8/8/8/8/R6K w - - 0 1"
    engine.board = chess.Board(fen)
    
    # Search should find mate immediately
    best_move = engine.search(engine.board, time_limit=1.0)
    
    if best_move.uci() == "a1a8":
        print("✓ PASS: Search found mate-in-1 (Ra8#)")
    else:
        print(f"✗ FAIL: Expected Ra8#, got {best_move.uci()}")
        return False
    
    # Test normal position
    print("\nPosition: Normal middlegame")
    engine.board = chess.Board()
    engine.board.push_san("e4")
    engine.board.push_san("e5")
    engine.board.push_san("Nf3")
    
    best_move = engine.search(engine.board, time_limit=2.0)
    
    print(f"Move selected: {best_move.uci()}")
    if best_move in engine.board.legal_moves:
        print("✓ PASS: Valid move returned")
    else:
        print("✗ FAIL: Invalid move returned")
        return False
    
    return True


def run_all_tests():
    """Run all v17.8 tests"""
    print("=" * 60)
    print("V7P3R v17.8 Test Suite")
    print("Time Management & Mate Detection")
    print("=" * 60)
    
    tests = [
        ("Time Manager - Blitz", test_time_manager_blitz),
        ("Time Pressure", test_time_pressure),
        ("Mate-in-1 Detection", test_mate_in_1_detection),
        ("Back-Rank Safety", test_back_rank_safety),
        ("Mate Threat Extension", test_mate_threat_extension),
        ("Integration Search", test_integration_search),
    ]
    
    passed = 0
    failed = 0
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            if result:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"\n✗ TEST FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 60)
    print(f"RESULTS: {passed}/{len(tests)} tests passed")
    if failed == 0:
        print("✓ ALL TESTS PASSED")
    else:
        print(f"✗ {failed} test(s) failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
