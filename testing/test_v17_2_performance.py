#!/usr/bin/env python3
"""
V17.2 Performance Testing Suite
Tests NPS improvements and validates tactical strength preservation
"""

import sys
import os
import time
import chess

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from v7p3r import V7P3REngine


def test_nps_benchmark():
    """Measure NPS with perft test"""
    print("\n=== NPS Benchmark (Perft) ===")
    engine = V7P3REngine(use_fast_evaluator=True)
    board = chess.Board()
    
    print("Running perft(5) from starting position...")
    start = time.time()
    nodes = engine.perft(board, depth=5, divide=False, root_call=True)
    elapsed = time.time() - start
    nps = int(nodes / elapsed)
    
    print(f"Nodes: {nodes:,}")
    print(f"Time: {elapsed:.2f}s")
    print(f"NPS: {nps:,}")
    print(f"Target: 200,000+ NPS for pure move generation")
    
    return nps


def test_depth_improvement():
    """Test depth reached in standard positions"""
    print("\n=== Depth Test (5 seconds per position) ===")
    engine = V7P3REngine(use_fast_evaluator=True)
    
    test_positions = [
        ("Starting position", "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"),
        ("Open middlegame", "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3"),
        ("Tactical position", "rnbqkb1r/pp2pppp/5n2/2pp4/2PP4/2N2N2/PP2PPPP/R1BQKB1R w KQkq - 0 5"),
        ("Complex middlegame", "r2qkb1r/pp2pppp/2n2n2/3p1b2/3P4/2N1PN2/PP3PPP/R1BQKB1R w KQkq - 4 7"),
        ("Endgame", "8/5k2/8/5P2/8/3K4/8/8 w - - 0 1"),
    ]
    
    depths = []
    for name, fen in test_positions:
        board = chess.Board(fen)
        print(f"\n{name}:")
        print(f"FEN: {fen}")
        
        start = time.time()
        best_move = engine.search(board, time_limit=5.0)
        elapsed = time.time() - start
        
        # Get depth info from last search
        depth = engine.default_depth
        nps = int(engine.nodes_searched / max(elapsed, 0.001))
        
        print(f"Best move: {best_move}")
        print(f"Nodes: {engine.nodes_searched:,}")
        print(f"Time: {elapsed:.2f}s")
        print(f"NPS: {nps:,}")
        print(f"Depth: {depth}")
        
        depths.append(depth)
    
    avg_depth = sum(depths) / len(depths)
    print(f"\n=== Average Depth: {avg_depth:.1f} ===")
    print(f"Target: 6.5+ (v17.2.0), 7.0+ (v17.2.1), 8.5+ (v17.2.2)")
    
    return avg_depth


def test_tactical_positions():
    """Test tactical strength on known positions"""
    print("\n=== Tactical Validation ===")
    engine = V7P3REngine(use_fast_evaluator=True)
    
    # Tactical test suite (position, best move, description)
    tactical_tests = [
        ("r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - 4 4", 
         "h5f7", "Scholar's mate"),
        ("r1bqkbnr/pppp1ppp/2n5/1B2p3/4P3/5N2/PPPP1PPP/RNBQK2R b KQkq - 3 3",
         "d8g5", "Pin the knight"),
        ("rnbqkb1r/pppp1ppp/5n2/4p3/2B1P3/8/PPPP1PPP/RNBQK1NR w KQkq - 2 3",
         "c4f7", "Fried liver attack"),
    ]
    
    passed = 0
    failed = 0
    
    for fen, expected_move, description in tactical_tests:
        board = chess.Board(fen)
        print(f"\n{description}:")
        print(f"FEN: {fen}")
        
        best_move = engine.search(board, time_limit=3.0)
        best_move_str = str(best_move)
        
        if best_move_str == expected_move:
            print(f"✓ PASS - Found {best_move_str}")
            passed += 1
        else:
            print(f"✗ FAIL - Expected {expected_move}, got {best_move_str}")
            failed += 1
    
    print(f"\n=== Tactical Results: {passed}/{len(tactical_tests)} passed ===")
    return passed, failed


def test_cache_statistics():
    """Test unified TT cache performance"""
    print("\n=== Cache Statistics ===")
    engine = V7P3REngine(use_fast_evaluator=True)
    board = chess.Board()
    
    # Run a search to populate cache
    print("Running 5-second search from starting position...")
    engine.search(board, time_limit=5.0)
    
    stats = engine.search_stats
    print(f"\nCache hits: {stats['cache_hits']:,}")
    print(f"Cache misses: {stats['cache_misses']:,}")
    total = stats['cache_hits'] + stats['cache_misses']
    if total > 0:
        hit_rate = (stats['cache_hits'] / total) * 100
        print(f"Hit rate: {hit_rate:.1f}%")
        print(f"Target: 60%+ hit rate")
    
    print(f"\nTT hits: {stats['tt_hits']:,}")
    print(f"TT stores: {stats['tt_stores']:,}")
    print(f"Killer hits: {stats['killer_hits']:,}")
    
    return stats


def main():
    """Run all v17.2 performance tests"""
    print("=" * 60)
    print("V7P3R v17.2.0 Performance Testing Suite")
    print("=" * 60)
    
    # Test 1: NPS benchmark
    nps = test_nps_benchmark()
    
    # Test 2: Depth improvement
    avg_depth = test_depth_improvement()
    
    # Test 3: Tactical validation
    passed, failed = test_tactical_positions()
    
    # Test 4: Cache statistics
    stats = test_cache_statistics()
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"NPS (perft): {nps:,}")
    print(f"Average depth: {avg_depth:.1f}")
    print(f"Tactical tests: {passed}/{passed+failed} passed")
    
    cache_total = stats['cache_hits'] + stats['cache_misses']
    if cache_total > 0:
        hit_rate = (stats['cache_hits'] / cache_total) * 100
        print(f"Cache hit rate: {hit_rate:.1f}%")
    
    print("\n" + "=" * 60)
    print("EXPECTED IMPROVEMENTS (vs v17.1.1)")
    print("=" * 60)
    print("Phase 1 (TT replacement): +20-25% NPS")
    print("Phase 2 (Unified cache): +30-35% NPS")
    print("Phase 3 (Quiescence opt): +5-8% NPS")
    print("Phase 4 (Buffer reuse): +8-12% NPS")
    print("Total expected: +68% NPS cumulative")
    print("\nTarget: 5,000 NPS → 8,400 NPS")
    print("Target depth: 5.2 → 7.0")


if __name__ == "__main__":
    main()
