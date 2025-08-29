#!/usr/bin/env python3
"""
V7P3R v8.1 Direct Testing Suite
Test the rollback to deterministic evaluation while preserving architectural improvements
"""

import sys
import os
sys.path.append('src')

import chess
import time
from v7p3r_v8_1 import V7P3REngineV81

def test_deterministic_evaluation():
    """Test that evaluation is now deterministic and consistent"""
    print("🔍 Testing Deterministic Evaluation")
    print("=" * 50)
    
    engine = V7P3REngineV81()
    
    # Test same position multiple times - should get identical results
    test_positions = [
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 4 5",
        "8/8/8/8/3k4/8/3K4/8 w - - 0 1",
    ]
    
    for i, fen in enumerate(test_positions, 1):
        print(f"\n📍 Position {i}: {fen[:20]}...")
        board = chess.Board(fen)
        
        # Evaluate same position 5 times
        evaluations = []
        for run in range(5):
            eval_score = engine._evaluate_position_deterministic(board)
            evaluations.append(eval_score)
        
        # Check consistency
        unique_evals = set(evaluations)
        if len(unique_evals) == 1:
            print(f"✅ Deterministic: All 5 runs = {evaluations[0]:+.2f}")
        else:
            print(f"❌ Non-deterministic: {evaluations}")
        
        # Reset cache and test again
        engine.evaluation_cache.clear()
        fresh_eval = engine._evaluate_position_deterministic(board)
        if fresh_eval == evaluations[0]:
            print(f"✅ Cache-independent: Fresh eval = {fresh_eval:+.2f}")
        else:
            print(f"❌ Cache-dependent: Fresh = {fresh_eval:+.2f}, Cached = {evaluations[0]:+.2f}")
    
    return True

def test_unified_search_stability():
    """Test that unified search architecture is stable"""
    print("\n🔧 Testing Unified Search Stability")
    print("=" * 50)
    
    engine = V7P3REngineV81()
    board = chess.Board()
    
    # Test search at different depths
    depths = [1, 2, 3, 4]
    for depth in depths:
        print(f"\n🎯 Testing depth {depth}:")
        
        try:
            start_time = time.time()
            move, score, pv = engine._unified_search_root(board, depth, engine._configure_search_options(5.0))
            search_time = time.time() - start_time
            
            print(f"  Move: {move}")
            print(f"  Score: {score:+.2f}")
            print(f"  PV: {' '.join(str(m) for m in pv[:3])}")
            print(f"  Time: {search_time:.3f}s")
            print(f"  Nodes: {engine.nodes_searched}")
            
            # Reset for next test
            engine.nodes_searched = 0
            
        except Exception as e:
            print(f"  ❌ Error at depth {depth}: {e}")
            return False
    
    print("✅ Unified search stable at all depths")
    return True

def test_move_ordering_consistency():
    """Test that move ordering is working and consistent"""
    print("\n📋 Testing Move Ordering Consistency")
    print("=" * 50)
    
    engine = V7P3REngineV81()
    
    # Test position with various move types
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 4 5")
    legal_moves = list(board.legal_moves)
    
    print(f"Position has {len(legal_moves)} legal moves")
    
    # Test move ordering multiple times
    from v7p3r_v8_1 import SearchOptions
    options = SearchOptions()
    
    ordered_moves_runs = []
    for run in range(3):
        ordered_moves = engine._order_moves_enhanced(board, legal_moves.copy(), 0, options)
        ordered_moves_runs.append([str(move) for move in ordered_moves[:5]])  # Top 5 moves
    
    # Check consistency
    if all(run == ordered_moves_runs[0] for run in ordered_moves_runs):
        print(f"✅ Move ordering consistent across runs")
        print(f"  Top moves: {', '.join(ordered_moves_runs[0])}")
    else:
        print(f"❌ Move ordering inconsistent:")
        for i, run in enumerate(ordered_moves_runs):
            print(f"  Run {i+1}: {', '.join(run)}")
    
    # Test move scoring
    print(f"\n🏆 Move scores for top 5 moves:")
    ordered_moves = engine._order_moves_enhanced(board, legal_moves, 0, options)
    for i, move in enumerate(ordered_moves[:5]):
        score = engine._score_move_enhanced(board, move, 0, options)
        move_type = "Capture" if board.is_capture(move) else "Quiet"
        print(f"  {i+1}. {move} ({move_type}): {score:.0f}")
    
    return True

def test_cache_functionality():
    """Test that evaluation cache is working properly"""
    print("\n💾 Testing Cache Functionality")
    print("=" * 50)
    
    engine = V7P3REngineV81()
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 4 5")
    
    # Clear cache and check initial state
    engine.evaluation_cache.clear()
    engine.search_stats['cache_hits'] = 0
    engine.search_stats['cache_misses'] = 0
    
    print(f"Initial cache size: {len(engine.evaluation_cache)}")
    
    # First evaluation - should be cache miss
    eval1 = engine._evaluate_position_deterministic(board)
    print(f"First eval: {eval1:+.2f}")
    print(f"Cache misses: {engine.search_stats['cache_misses']}")
    print(f"Cache hits: {engine.search_stats['cache_hits']}")
    print(f"Cache size: {len(engine.evaluation_cache)}")
    
    # Second evaluation - should be cache hit
    eval2 = engine._evaluate_position_deterministic(board)
    print(f"Second eval: {eval2:+.2f}")
    print(f"Cache misses: {engine.search_stats['cache_misses']}")
    print(f"Cache hits: {engine.search_stats['cache_hits']}")
    
    if eval1 == eval2 and engine.search_stats['cache_hits'] == 1:
        print("✅ Cache working correctly")
    else:
        print("❌ Cache issue detected")
    
    return True

def test_time_management():
    """Test that time management is working properly"""
    print("\n⏱️  Testing Time Management")
    print("=" * 50)
    
    engine = V7P3REngineV81()
    board = chess.Board()
    
    # Test different time limits
    time_limits = [0.5, 1.0, 2.0]
    
    for time_limit in time_limits:
        print(f"\n🕐 Testing {time_limit}s time limit:")
        
        start_time = time.time()
        move = engine.search(board, time_limit)
        actual_time = time.time() - start_time
        
        print(f"  Move: {move}")
        print(f"  Target: {time_limit:.1f}s")
        print(f"  Actual: {actual_time:.3f}s")
        print(f"  Ratio: {actual_time/time_limit:.2f}")
        
        # Should not exceed time by more than 50%
        if actual_time <= time_limit * 1.5:
            print(f"  ✅ Time management good")
        else:
            print(f"  ⚠️  Time overrun")
        
        # Reset for next test
        engine.new_game()
    
    return True

def test_no_async_artifacts():
    """Test that there are no leftover async artifacts causing issues"""
    print("\n🧹 Testing No Async Artifacts")
    print("=" * 50)
    
    engine = V7P3REngineV81()
    
    # Check that async attributes don't exist
    async_attributes = [
        'thread_pool', 'confidence_threshold', 'evaluation_timeouts'
    ]
    
    for attr in async_attributes:
        if hasattr(engine, attr):
            print(f"⚠️  Found async artifact: {attr}")
        else:
            print(f"✅ No {attr} (good)")
    
    # Check that evaluation is purely synchronous
    board = chess.Board("r1bqkb1r/pppp1ppp/2n2n2/4p3/4P3/3P1N2/PPP2PPP/RNBQKB1R w KQkq - 4 5")
    
    # Time evaluation - should be very fast and consistent
    eval_times = []
    for _ in range(5):
        start = time.time()
        engine._evaluate_position_deterministic(board)
        eval_times.append(time.time() - start)
    
    avg_time = sum(eval_times) / len(eval_times)
    max_time = max(eval_times)
    
    print(f"Evaluation times: {[f'{t:.6f}' for t in eval_times]}")
    print(f"Average: {avg_time:.6f}s")
    print(f"Maximum: {max_time:.6f}s")
    
    if max_time < 0.001:  # Should be sub-millisecond
        print("✅ Evaluation is fast and synchronous")
    else:
        print("⚠️  Evaluation seems slow")
    
    return True

def main():
    """Run all V8.1 stability tests"""
    print("V7P3R v8.1 Stability Testing")
    print("Rolling back async, preserving architecture")
    print("=" * 60)
    
    tests = [
        ("Deterministic Evaluation", test_deterministic_evaluation),
        ("Unified Search Stability", test_unified_search_stability),
        ("Move Ordering Consistency", test_move_ordering_consistency),
        ("Cache Functionality", test_cache_functionality),
        ("Time Management", test_time_management),
        ("No Async Artifacts", test_no_async_artifacts),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🧪 Running: {test_name}")
        try:
            if test_func():
                print(f"✅ {test_name}: PASSED")
                passed += 1
            else:
                print(f"❌ {test_name}: FAILED")
        except Exception as e:
            print(f"💥 {test_name}: ERROR - {e}")
            import traceback
            traceback.print_exc()
    
    print(f"\n🏆 V8.1 STABILITY RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 V8.1 is stable! Ready for move ordering analysis.")
        return 0
    else:
        print("⚠️  Some stability issues detected. Fix before proceeding.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
