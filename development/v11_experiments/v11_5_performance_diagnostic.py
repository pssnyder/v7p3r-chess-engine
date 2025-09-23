#!/usr/bin/env python3
"""
V11.5 Performance Diagnostic
============================

Systematic analysis of V7P3R performance bottlenecks:
1. UCI output overhead
2. Tactical pattern detection frequency
3. Hash position calls
4. Move generation efficiency
5. Evaluation caching effectiveness

This will help us prioritize optimization efforts.
"""

import time
import chess
import cProfile
import pstats
import io
from src.v7p3r import V7P3REngine

def profile_tactical_calls():
    """Profile how often tactical pattern detector is called"""
    print("=== TACTICAL CALL FREQUENCY ANALYSIS ===")
    
    engine = V7P3REngine()
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    
    # Patch tactical detector to count calls
    original_evaluate = engine.tactical_pattern_detector.evaluate_tactical_patterns
    call_count = {'count': 0}
    
    def counting_evaluate(*args, **kwargs):
        call_count['count'] += 1
        return original_evaluate(*args, **kwargs)
    
    engine.tactical_pattern_detector.evaluate_tactical_patterns = counting_evaluate
    
    # Run single move search
    start_time = time.time()
    try:
        result = engine.search(board, depth=3, time_limit=5.0)
        elapsed = time.time() - start_time
        
        print(f"Search completed in {elapsed:.3f}s")
        print(f"Tactical pattern detector called: {call_count['count']} times")
        print(f"Tactical calls per second: {call_count['count'] / elapsed:.1f}")
        print(f"Cache stats: {engine.tactical_cache.get_stats()}")
        
    except Exception as e:
        print(f"Search failed: {e}")

def profile_hash_calls():
    """Profile hash_position call frequency"""
    print("\n=== HASH POSITION CALL ANALYSIS ===")
    
    engine = V7P3REngine()
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    
    # Patch hash_position to count calls
    original_hash = engine.zobrist.hash_position
    call_count = {'count': 0}
    
    def counting_hash(*args, **kwargs):
        call_count['count'] += 1
        return original_hash(*args, **kwargs)
    
    engine.zobrist.hash_position = counting_hash
    
    # Run single move search
    start_time = time.time()
    try:
        result = engine.search(board, depth=3, time_limit=5.0)
        elapsed = time.time() - start_time
        
        print(f"Search completed in {elapsed:.3f}s")
        print(f"hash_position called: {call_count['count']} times")
        print(f"Hash calls per second: {call_count['count'] / elapsed:.1f}")
        
    except Exception as e:
        print(f"Search failed: {e}")

def profile_move_generation():
    """Profile move generation efficiency"""
    print("\n=== MOVE GENERATION ANALYSIS ===")
    
    engine = V7P3REngine()
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    
    # Count legal_moves calls
    original_legal_moves = board.legal_moves
    call_count = {'count': 0}
    
    # Test move generation speed
    start_time = time.time()
    for _ in range(1000):
        moves = list(board.legal_moves)
        call_count['count'] += 1
    elapsed = time.time() - start_time
    
    print(f"1000 move generations completed in {elapsed:.3f}s")
    print(f"Move generation rate: {1000 / elapsed:.1f} calls/second")
    print(f"Moves found: {len(moves)}")

def profile_evaluation_cache():
    """Profile evaluation cache effectiveness"""
    print("\n=== EVALUATION CACHE ANALYSIS ===")
    
    engine = V7P3REngine()
    board = chess.Board("rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1")
    
    # Test evaluation caching
    start_time = time.time()
    for _ in range(100):
        # Same position multiple times - should hit cache
        score = engine._evaluate_position(board, depth=3)
    elapsed = time.time() - start_time
    
    print(f"100 evaluations completed in {elapsed:.3f}s")
    print(f"Evaluation rate: {100 / elapsed:.1f} evals/second")
    print(f"Cache hits: {engine.eval_cache_hits}")
    print(f"Cache misses: {engine.eval_cache_misses}")
    hit_rate = engine.eval_cache_hits / (engine.eval_cache_hits + engine.eval_cache_misses) * 100
    print(f"Cache hit rate: {hit_rate:.1f}%")

def test_uci_output_overhead():
    """Test if UCI output is causing performance issues"""
    print("\n=== UCI OUTPUT OVERHEAD ANALYSIS ===")
    
    # Test print performance
    start_time = time.time()
    for i in range(10000):
        # Simulate UCI info output
        pass  # No output
    no_output_time = time.time() - start_time
    
    start_time = time.time()
    for i in range(10000):
        # Simulate UCI info output with actual printing
        print(f"info depth {i} score cp 100 nodes {i*10} time {i} nps {i*100}")
    with_output_time = time.time() - start_time
    
    print(f"10000 iterations without output: {no_output_time:.3f}s")
    print(f"10000 iterations with UCI output: {with_output_time:.3f}s")
    print(f"Output overhead: {(with_output_time - no_output_time)*1000:.1f}ms")
    print(f"Overhead per UCI line: {(with_output_time - no_output_time)*1000/10000:.3f}ms")

def run_comprehensive_profile():
    """Run comprehensive performance profiling"""
    print("V11.5 PERFORMANCE DIAGNOSTIC")
    print("============================")
    
    # Test each bottleneck category
    profile_tactical_calls()
    profile_hash_calls() 
    profile_move_generation()
    profile_evaluation_cache()
    test_uci_output_overhead()
    
    print("\n=== OPTIMIZATION PRIORITY RANKING ===")
    print("1. CRITICAL: Tactical pattern detector (called hundreds of times)")
    print("2. HIGH: Hash position calls (excessive redundancy)")
    print("3. MEDIUM: UCI output overhead (if printing during search)")
    print("4. LOW: Move generation (inherently fast)")
    print("5. LOW: Evaluation cache (should have high hit rate)")

if __name__ == "__main__":
    run_comprehensive_profile()