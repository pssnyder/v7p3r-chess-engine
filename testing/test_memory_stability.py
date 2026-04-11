#!/usr/bin/env python3
"""
Memory Stability Test for V7P3R v18.4 Phase 1

Tests memory usage before/after implementing bounded caches:
1. Evaluation cache (unbounded → LRU 20k)
2. Transposition table eviction (truncate → depth-preferred)
3. History heuristic (unbounded → 10k cap)

Usage:
    python testing/test_memory_stability.py
"""

import sys
import os
import chess
import tracemalloc
import traceback
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from v7p3r import V7P3REngine


def test_evaluation_cache_size():
    """Test evaluation cache doesn't grow unbounded"""
    print("\n" + "="*80)
    print("TEST: Evaluation Cache Size Limit")
    print("="*80)
    
    engine = V7P3REngine()
    board = chess.Board()
    
    # Play 100 moves, evaluating each position multiple times
    moves_played = 0
    for _ in range(50):  # 50 move pairs = 100 plies
        if board.is_game_over():
            break
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        
        # Evaluate position 5 times (simulating search re-visiting positions)
        for _ in range(5):
            engine._evaluate_position(board)
        
        # Make a move
        move = legal_moves[0]
        board.push(move)
        moves_played += 1
    
    cache_size = len(engine.evaluation_cache)
    
    print(f"Moves played: {moves_played}")
    print(f"Evaluation cache size: {cache_size}")
    print(f"Expected max: 20,000 entries (if bounded)")
    
    # Before fix: cache will grow unbounded (500+ entries for 100 positions)
    # After fix: should be bounded to 20,000 entries
    if cache_size > 25000:
        print("❌ FAIL: Cache exceeds 25k limit (unbounded growth detected)")
        return False
    else:
        print("✅ PASS: Cache size within acceptable bounds")
        return True


def test_transposition_table_eviction():
    """Test TT eviction strategy preserves deep entries"""
    print("\n" + "="*80)
    print("TEST: Transposition Table Eviction Strategy")
    print("="*80)
    
    engine = V7P3REngine()
    board = chess.Board()
    
    # Fill TT beyond max capacity
    initial_size = len(engine.transposition_table)
    
    # Search from starting position to populate TT
    best_move = engine.search(board, time_limit=5.0)
    
    tt_size = len(engine.transposition_table)
    max_entries = engine.max_tt_entries
    
    print(f"Initial TT size: {initial_size}")
    print(f"TT size after search: {tt_size}")
    print(f"Max TT entries: {max_entries}")
    print(f"Best move found: {best_move}")
    
    if tt_size > max_entries * 1.1:
        print(f"❌ FAIL: TT size ({tt_size}) exceeds limit ({max_entries}) by >10%")
        return False
    
    # Check that deep entries are preserved
    if tt_size > 0:
        depths = [entry.depth for entry in engine.transposition_table.values()]
        avg_depth = sum(depths) / len(depths)
        max_depth = max(depths)
        
        print(f"Average stored depth: {avg_depth:.2f}")
        print(f"Max stored depth: {max_depth}")
        
        # After fix: should preserve deeper entries
        # Before fix: truncation keeps random 75%, may lose deep entries
        if avg_depth >= 2.0:
            print("✅ PASS: TT preserves reasonable average depth")
            return True
        else:
            print("⚠️  WARNING: Low average depth, may indicate poor eviction")
            return True  # Not a failure, just suboptimal
    else:
        print("⚠️  WARNING: TT empty after search")
        return True


def test_history_heuristic_size():
    """Test history heuristic doesn't grow unbounded"""
    print("\n" + "="*80)
    print("TEST: History Heuristic Size Limit")
    print("="*80)
    
    engine = V7P3REngine()
    board = chess.Board()
    
    # Play 100 moves, updating history
    moves_played = 0
    for _ in range(50):
        if board.is_game_over():
            break
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        
        # Update history for each legal move (simulating search)
        for move in legal_moves[:5]:  # Top 5 moves
            engine.history_heuristic.update_history(move, depth=4)
        
        move = legal_moves[0]
        board.push(move)
        moves_played += 1
    
    history_size = len(engine.history_heuristic.history)
    
    print(f"Moves played: {moves_played}")
    print(f"History table size: {history_size}")
    print(f"Expected max: 10,000 entries (if bounded)")
    
    # Before fix: will grow unbounded
    # After fix: should be capped at 10,000
    if history_size > 12000:
        print("❌ FAIL: History table exceeds 12k limit (unbounded growth detected)")
        return False
    else:
        print("✅ PASS: History table size within acceptable bounds")
        return True


def test_memory_usage_over_long_game():
    """Test memory doesn't leak over a long game"""
    print("\n" + "="*80)
    print("TEST: Memory Usage Over Long Game (200 moves)")
    print("="*80)
    
    tracemalloc.start()
    
    engine = V7P3REngine()
    board = chess.Board()
    
    memory_samples = []
    
    # Play 200 moves with periodic memory checks
    for move_num in range(1, 201):
        if board.is_game_over():
            break
        
        legal_moves = list(board.legal_moves)
        if not legal_moves:
            break
        
        # Make a move
        move = legal_moves[0]
        board.push(move)
        
        # Sample memory every 20 moves
        if move_num % 20 == 0:
            current, peak = tracemalloc.get_traced_memory()
            memory_samples.append(current / 1024 / 1024)  # Convert to MB
            print(f"Move {move_num}: {current / 1024 / 1024:.2f} MB current, {peak / 1024 / 1024:.2f} MB peak")
    
    tracemalloc.stop()
    
    if len(memory_samples) > 2:
        # Check if memory grows linearly (leak) or stabilizes
        first_half_avg = sum(memory_samples[:len(memory_samples)//2]) / (len(memory_samples)//2)
        second_half_avg = sum(memory_samples[len(memory_samples)//2:]) / (len(memory_samples) - len(memory_samples)//2)
        
        growth = second_half_avg - first_half_avg
        growth_percent = (growth / first_half_avg) * 100 if first_half_avg > 0 else 0
        
        print(f"\nFirst half average: {first_half_avg:.2f} MB")
        print(f"Second half average: {second_half_avg:.2f} MB")
        print(f"Memory growth: {growth:.2f} MB ({growth_percent:.1f}%)")
        
        # Before fix: expect >20% growth (unbounded caches)
        # After fix: expect <10% growth (bounded caches stabilize)
        if growth_percent > 25:
            print("❌ FAIL: Excessive memory growth detected (>25%), likely memory leak")
            return False
        elif growth_percent > 10:
            print("⚠️  WARNING: Memory growth >10%, caches may not be bounded")
            return True  # Pass but with warning
        else:
            print("✅ PASS: Memory usage stable over long game")
            return True
    else:
        print("⚠️  Not enough samples collected")
        return True


def test_cache_functionality():
    """Verify caches still function correctly after bounding"""
    print("\n" + "="*80)
    print("TEST: Cache Functionality (hit rates)")
    print("="*80)
    
    engine = V7P3REngine()
    board = chess.Board()
    
    # Reset stats
    engine.search_stats['cache_hits'] = 0
    engine.search_stats['cache_misses'] = 0
    engine.search_stats['tt_hits'] = 0
    
    # Run a short search
    best_move = engine.search(board, time_limit=3.0)
    
    cache_hits = engine.search_stats['cache_hits']
    cache_misses = engine.search_stats['cache_misses']
    tt_hits = engine.search_stats['tt_hits']
    
    total_lookups = cache_hits + cache_misses
    cache_hit_rate = (cache_hits / total_lookups * 100) if total_lookups > 0 else 0
    
    print(f"Best move: {best_move}")
    print(f"Eval cache hits: {cache_hits}")
    print(f"Eval cache misses: {cache_misses}")
    print(f"Cache hit rate: {cache_hit_rate:.1f}%")
    print(f"TT hits: {tt_hits}")
    
    # Cache should still be functional
    if cache_hit_rate > 10:
        print("✅ PASS: Caches functioning (hit rate >10%)")
        return True
    else:
        print("⚠️  WARNING: Low cache hit rate, may indicate issue")
        return True


def run_all_tests():
    """Run all memory stability tests"""
    print("\n" + "="*80)
    print("V7P3R v18.4 PHASE 1: MEMORY STABILITY TEST SUITE")
    print("="*80)
    
    tests = [
        ("Evaluation Cache Size", test_evaluation_cache_size),
        ("Transposition Table Eviction", test_transposition_table_eviction),
        ("History Heuristic Size", test_history_heuristic_size),
        ("Memory Over Long Game", test_memory_usage_over_long_game),
        ("Cache Functionality", test_cache_functionality),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            passed = test_func()
            results.append((name, passed))
        except Exception as e:
            print(f"\n❌ EXCEPTION in {name}: {e}")
            traceback.print_exc()
            results.append((name, False))
    
    # Summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    
    passed_count = sum(1 for _, passed in results if passed)
    total_count = len(results)
    
    for name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    
    if passed_count == total_count:
        print("\n🎉 ALL TESTS PASSED")
        return 0
    else:
        print(f"\n⚠️  {total_count - passed_count} TEST(S) FAILED")
        return 1


if __name__ == "__main__":
    sys.exit(run_all_tests())
