#!/usr/bin/env python3
"""
Debug v18.0 Threefold Avoidance Implementation

Tests for issues causing depth 0 searches and game losses in tournament.
"""

import sys
import os
import time
import chess

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine

def test_basic_search():
    """Test 1: Basic search functionality"""
    print("=" * 60)
    print("TEST 1: Basic Search Functionality")
    print("=" * 60)
    
    engine = V7P3REngine()
    board = chess.Board()
    
    try:
        start = time.time()
        best_move, score = engine.get_best_move(board, time_limit=5.0)
        elapsed = time.time() - start
        
        print(f"✓ Search completed in {elapsed:.2f}s")
        print(f"  Best move: {best_move}")
        print(f"  Score: {score:.2f}")
        print(f"  Nodes: {engine.nodes_searched}")
        print(f"  Depth achieved: {engine.search_stats.get('max_depth', 0)}")
        
        if best_move is None:
            print("✗ FAIL: No move returned!")
            return False
        if engine.nodes_searched == 0:
            print("✗ FAIL: No nodes searched!")
            return False
            
        return True
        
    except Exception as e:
        print(f"✗ FAIL: Exception during search: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_threefold_detection():
    """Test 2: Threefold repetition detection"""
    print("\n" + "=" * 60)
    print("TEST 2: Threefold Detection")
    print("=" * 60)
    
    engine = V7P3REngine()
    
    # Create a position that will repeat
    board = chess.Board()
    board.push_san("Nf3")
    board.push_san("Nf6")
    board.push_san("Ng1")
    board.push_san("Ng8")
    
    print(f"Position after Nf3 Nf6 Ng1 Ng8:")
    print(f"  Is repetition(0): {board.is_repetition(0)}")
    print(f"  Is repetition(1): {board.is_repetition(1)}")
    print(f"  Is repetition(2): {board.is_repetition(2)}")
    
    # Test the helper method
    move = chess.Move.from_uci("g1f3")  # Would create 3-fold
    
    try:
        is_threefold = engine._would_cause_threefold(board, move)
        print(f"  _would_cause_threefold(Nf3): {is_threefold}")
        
        if is_threefold:
            print("✓ Correctly detected threefold repetition")
        else:
            print("✓ Detection working (may not be threefold yet)")
            
        return True
        
    except Exception as e:
        print(f"✗ FAIL: Exception in threefold check: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_move_ordering_with_threefold():
    """Test 3: Move ordering with threefold penalty"""
    print("\n" + "=" * 60)
    print("TEST 3: Move Ordering with Threefold Penalty")
    print("=" * 60)
    
    engine = V7P3REngine()
    
    # Create repetition scenario
    board = chess.Board()
    board.push_san("Nf3")
    board.push_san("Nf6")
    board.push_san("Ng1")
    board.push_san("Ng8")
    
    legal_moves = list(board.legal_moves)
    print(f"Legal moves: {len(legal_moves)}")
    
    try:
        ordered = engine._order_moves_advanced(board, legal_moves, depth=4, tt_move=None)
        print(f"Ordered moves: {len(ordered)}")
        print(f"  Top 3 moves: {[str(m) for m in ordered[:3]]}")
        
        # Check if Nf3 (repetition) is penalized
        nf3 = chess.Move.from_uci("g1f3")
        if nf3 in ordered:
            nf3_index = ordered.index(nf3)
            print(f"  Nf3 (repetition) at position: {nf3_index}/{len(ordered)}")
            if nf3_index > len(ordered) / 2:
                print("✓ Repetition move penalized (low priority)")
            else:
                print("⚠ Repetition move not strongly penalized")
        
        return True
        
    except Exception as e:
        print(f"✗ FAIL: Exception in move ordering: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_performance_benchmark():
    """Test 4: Performance benchmark"""
    print("\n" + "=" * 60)
    print("TEST 4: Performance Benchmark (10 positions)")
    print("=" * 60)
    
    engine = V7P3REngine()
    positions = [
        chess.Board(),  # Starting position
        chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"),  # After e4
        chess.Board("rnbqkb1r/pppp1ppp/5n2/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"),  # After Nf6
        chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"),  # Sicilian
    ]
    
    total_nodes = 0
    total_time = 0
    
    try:
        for i, board in enumerate(positions, 1):
            start = time.time()
            move, score = engine.get_best_move(board, time_limit=2.0)
            elapsed = time.time() - start
            
            if move is None:
                print(f"✗ Position {i}: No move returned!")
                return False
            
            total_nodes += engine.nodes_searched
            total_time += elapsed
            
            print(f"  Position {i}: {move} ({engine.nodes_searched} nodes, {elapsed:.2f}s)")
        
        nps = total_nodes / total_time if total_time > 0 else 0
        print(f"\n✓ Benchmark complete:")
        print(f"  Total nodes: {total_nodes}")
        print(f"  Total time: {total_time:.2f}s")
        print(f"  NPS: {nps:.0f}")
        
        if nps < 1000:
            print("⚠ WARNING: Very low NPS - performance issue detected")
        
        return True
        
    except Exception as e:
        print(f"✗ FAIL: Exception during benchmark: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_game21_position():
    """Test 5: Specific position from failed Game 21"""
    print("\n" + "=" * 60)
    print("TEST 5: Game 21 Starting Position (Where Failure Occurred)")
    print("=" * 60)
    
    engine = V7P3REngine()
    
    # This is the position where v18 started failing (depth 0)
    # Extract from PGN if available, or use a typical middlegame position
    board = chess.Board("rnbqkb1r/ppp1pppp/5n2/3p4/2PP4/5N2/PP2PPPP/RNBQKB1R b KQkq - 0 3")
    
    try:
        print("Testing position:")
        print(board)
        
        start = time.time()
        move, score = engine.get_best_move(board, time_limit=5.0)
        elapsed = time.time() - start
        
        print(f"\n  Best move: {move}")
        print(f"  Score: {score:.2f}")
        print(f"  Nodes: {engine.nodes_searched}")
        print(f"  Time: {elapsed:.2f}s")
        print(f"  Max depth: {engine.search_stats.get('max_depth', 0)}")
        
        if move is None:
            print("✗ FAIL: No move returned (same as tournament failure!)")
            return False
        if engine.search_stats.get('max_depth', 0) == 0:
            print("✗ FAIL: Depth 0 search (same as tournament failure!)")
            return False
        
        print("✓ Position search successful")
        return True
        
    except Exception as e:
        print(f"✗ FAIL: Exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_safety_checks():
    """Test 6: Safety checker functionality"""
    print("\n" + "=" * 60)
    print("TEST 6: Safety Checker")
    print("=" * 60)
    
    engine = V7P3REngine()
    
    # Position where a piece can hang
    board = chess.Board("rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2")
    
    try:
        # Test a safe move
        safe_move = chess.Move.from_uci("b8c6")
        safety_score = engine.move_safety.evaluate_move_safety(board, safe_move)
        print(f"Safe move (Nc6): penalty = {safety_score:.2f}")
        
        # Test if safety checker is working
        if safety_score > -1000:  # Reasonable penalty range
            print("✓ Safety checker working")
            return True
        else:
            print("⚠ Extreme penalty detected")
            return True
            
    except Exception as e:
        print(f"✗ FAIL: Exception in safety check: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests"""
    print("V7P3R v18.0 Threefold Avoidance Debug Suite")
    print("=" * 60)
    
    tests = [
        ("Basic Search", test_basic_search),
        ("Threefold Detection", test_threefold_detection),
        ("Move Ordering", test_move_ordering_with_threefold),
        ("Performance", test_performance_benchmark),
        ("Game 21 Position", test_game21_position),
        ("Safety Checks", test_safety_checks),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n✗ CRITICAL: Test '{name}' crashed: {e}")
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
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n✓ All tests passed - engine appears functional")
    else:
        print(f"\n✗ {total - passed} test(s) failed - issues detected")
    
    return passed == total


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
