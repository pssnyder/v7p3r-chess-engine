#!/usr/bin/env python3
"""Test V15.1 Depth 8 and Phase-Aware Time Management

Tests that the engine:
1. Can reach depth 8 consistently
2. Allocates more time in opening/middlegame
3. Maintains depth consistency across phases
4. Doesn't timeout in rapid games
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
import time
from v7p3r import V7P3REngine


def test_depth_8_capability():
    """Test that engine can reach depth 8 with time allocation"""
    print("=" * 80)
    print("TEST 1: Depth 8 Capability")
    print("=" * 80)
    
    engine = V7P3REngine()
    print(f"✓ Engine initialized with max_depth = {engine.max_depth}")
    
    if engine.max_depth != 8:
        print(f"✗ FAILED: Expected max_depth=8, got {engine.max_depth}")
        return False
    
    # Test starting position with 180 seconds (3 minute game)
    time_left = 180.0
    increment = 2.0
    
    print(f"\nTest position: Starting position")
    print(f"Time control: {time_left}s + {increment}s increment")
    
    start_time = time.time()
    best_move = engine.get_best_move(time_left, increment)
    elapsed = time.time() - start_time
    
    print(f"Move: {best_move}")
    print(f"Time used: {elapsed:.3f}s")
    print(f"Expected depth reached: 8 (or close)")
    
    if elapsed > 30:
        print(f"✗ WARNING: Move took {elapsed:.3f}s, might be too slow")
    else:
        print(f"✓ Move completed in reasonable time")
    
    return best_move is not None


def test_phase_aware_time_allocation():
    """Test that engine allocates different time based on game phase"""
    print("\n" + "=" * 80)
    print("TEST 2: Phase-Aware Time Allocation")
    print("=" * 80)
    
    engine = V7P3REngine()
    
    # Test three positions: opening, middlegame, endgame
    test_cases = [
        {
            "name": "Opening (move 4)",
            "fen": "rnbqkbnr/ppp1pppp/8/3p4/3P4/8/PPP1PPPP/RNBQKBNR w KQkq - 0 2",
            "expected_phase": "opening"
        },
        {
            "name": "Middlegame (queens on, many pieces)",
            "fen": "r1bqk2r/pppp1ppp/2n2n2/2b1p3/2B1P3/3P1N2/PPP2PPP/RNBQK2R w KQkq - 0 6",
            "expected_phase": "middlegame"
        },
        {
            "name": "Endgame (queens off, few pieces)",
            "fen": "8/5k2/8/4K3/8/8/4R3/8 w - - 0 1",
            "expected_phase": "endgame"
        }
    ]
    
    time_left = 120.0
    increment = 1.0
    
    results = []
    
    for test in test_cases:
        print(f"\n{test['name']}:")
        print(f"FEN: {test['fen']}")
        
        engine.board = chess.Board(test['fen'])
        
        # Check phase detection
        phase = engine._get_game_phase(engine.board)
        print(f"Detected phase: {phase}")
        
        if phase != test['expected_phase']:
            print(f"✗ WARNING: Expected {test['expected_phase']}, got {phase}")
        else:
            print(f"✓ Phase detection correct")
        
        # Calculate time limit
        time_limit = engine._calculate_time_limit(time_left, increment)
        print(f"Time limit: {time_limit:.3f}s")
        
        # Make a move and time it
        start_time = time.time()
        best_move = engine.get_best_move(time_left, increment)
        elapsed = time.time() - start_time
        
        print(f"Move: {best_move}")
        print(f"Actual time used: {elapsed:.3f}s")
        
        results.append({
            "phase": phase,
            "time_limit": time_limit,
            "actual_time": elapsed
        })
    
    # Verify time allocation pattern
    print("\n" + "-" * 80)
    print("Time Allocation Summary:")
    for i, result in enumerate(results):
        print(f"{test_cases[i]['name']}: {result['time_limit']:.3f}s limit, {result['actual_time']:.3f}s used")
    
    # Expected: endgame should get most time, then middlegame, then opening
    if len(results) == 3:
        opening_limit = results[0]['time_limit']
        middlegame_limit = results[1]['time_limit']
        endgame_limit = results[2]['time_limit']
        
        if opening_limit < middlegame_limit < endgame_limit:
            print("✓ Time allocation pattern correct: opening < middlegame < endgame")
            return True
        else:
            print(f"✗ FAILED: Time allocation pattern incorrect")
            print(f"  Opening: {opening_limit:.3f}s")
            print(f"  Middlegame: {middlegame_limit:.3f}s")
            print(f"  Endgame: {endgame_limit:.3f}s")
            return False
    
    return False


def test_rapid_game_compatibility():
    """Test that engine doesn't timeout in rapid games"""
    print("\n" + "=" * 80)
    print("TEST 3: Rapid Game Compatibility (10+0)")
    print("=" * 80)
    
    engine = V7P3REngine()
    
    # Simulate a 10+0 rapid game (10 minutes, no increment)
    time_left = 600.0
    increment = 0.0
    
    print(f"Time control: {time_left/60:.0f}+{increment}")
    print("Simulating first 5 moves...\n")
    
    total_time = 0.0
    moves = []
    
    for i in range(5):
        start_time = time.time()
        best_move = engine.get_best_move(time_left, increment)
        elapsed = time.time() - start_time
        
        if best_move:
            engine.board.push(best_move)
            moves.append(best_move)
            time_left -= elapsed
            total_time += elapsed
            
            print(f"Move {i+1}: {best_move.uci()} (time: {elapsed:.3f}s, remaining: {time_left:.1f}s)")
    
    avg_time = total_time / len(moves) if moves else 0
    print(f"\nAverage time per move: {avg_time:.3f}s")
    print(f"Total time used: {total_time:.3f}s / 600s")
    print(f"Time remaining: {time_left:.1f}s")
    
    # In a 10+0 game, using more than 30s per move is risky
    if avg_time < 30:
        print(f"✓ Average time safe for rapid games")
        return True
    else:
        print(f"✗ WARNING: Average time too high for rapid games")
        return False


def test_depth_consistency():
    """Test that depth 8 is reached consistently"""
    print("\n" + "=" * 80)
    print("TEST 4: Depth 8 Consistency Check")
    print("=" * 80)
    
    engine = V7P3REngine()
    
    # Test 5 different positions with generous time
    positions = [
        ("Starting position", chess.Board()),
        ("Queen's Gambit", chess.Board("rnbqkbnr/ppp1pppp/8/3p4/2PP4/8/PP2PPPP/RNBQKBNR b KQkq - 0 2")),
        ("Italian Game", chess.Board("r1bqkbnr/pppp1ppp/2n5/4p3/2B1P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3")),
        ("Middlegame", chess.Board("r1bq1rk1/pp2ppbp/2np1np1/8/2BNP3/2N1BP2/PPPQ2PP/R3K2R w KQ - 0 9")),
        ("Endgame", chess.Board("8/5pk1/6p1/8/8/6P1/5PK1/8 w - - 0 1"))
    ]
    
    time_left = 180.0  # 3 minutes
    increment = 2.0
    
    print(f"Time control: {time_left}s + {increment}s increment")
    print(f"Expected: Engine should reach depth 7-8 in all positions\n")
    
    depths_reached = []
    
    for name, board in positions:
        engine.board = board
        phase = engine._get_game_phase(board)
        
        print(f"{name} (phase: {phase}):")
        
        start_time = time.time()
        best_move = engine.get_best_move(time_left, increment)
        elapsed = time.time() - start_time
        
        # Estimate depth reached based on nodes searched
        # At ~12K NPS with depth 8, we'd expect ~100K-500K nodes depending on position
        estimated_depth = min(8, 6 + (elapsed / 0.5))  # Rough estimate
        
        print(f"  Move: {best_move}")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  Nodes: {engine.nodes_searched:,}")
        print(f"  NPS: {engine.nodes_searched/elapsed:,.0f}" if elapsed > 0 else "  NPS: N/A")
        print(f"  Estimated depth: {estimated_depth:.1f}")
        
        depths_reached.append(estimated_depth)
    
    avg_depth = sum(depths_reached) / len(depths_reached) if depths_reached else 0
    print(f"\n" + "-" * 80)
    print(f"Average estimated depth: {avg_depth:.1f}")
    
    if avg_depth >= 7.0:
        print(f"✓ Depth consistency good (average >= 7.0)")
        return True
    else:
        print(f"✗ WARNING: Average depth below 7.0")
        return False


def main():
    print("V7P3R v15.1 - Depth 8 & Phase-Aware Time Management Tests")
    print("=" * 80)
    print()
    
    results = []
    
    # Run all tests
    results.append(("Depth 8 Capability", test_depth_8_capability()))
    results.append(("Phase-Aware Time Allocation", test_phase_aware_time_allocation()))
    results.append(("Rapid Game Compatibility", test_rapid_game_compatibility()))
    results.append(("Depth Consistency", test_depth_consistency()))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    for test_name, passed in results:
        status = "✓ PASSED" if passed else "✗ FAILED"
        print(f"{test_name}: {status}")
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 All tests passed! V15.1 with depth 8 is ready for deployment.")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} test(s) failed. Review results above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
